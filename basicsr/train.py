# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import argparse
import datetime
import logging
import math
import random
import time
import torch
import os
import numpy as np
from os import path as osp
from copy import deepcopy

from basicsr.data import create_dataloader, create_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import create_model
from basicsr.utils import (MessageLogger, check_resume, get_env_info,
                           get_root_logger, get_time_str, init_tb_logger,
                           init_wandb_logger, make_exp_dirs, mkdir_and_rename,
                           set_random_seed)
from basicsr.utils.dist_util import get_dist_info, init_dist
from basicsr.utils.options import dict2str, parse

# torch.autograd.set_detect_anomaly(True)

# --- MODIFICATION START: Helper function for creating dynamic dataloader ---
def create_progressive_dataloader(opt, logger, stage_idx, gt_sizes, batch_sizes):
    """
    Dynamically create a training dataloader for a specific progressive learning stage.
    """
    # Deep copy the dataset options to avoid modifying the original opt
    dataset_opt = opt['datasets']['train'].copy()

    # Update patch size and batch size for the current stage
    current_gt_size = gt_sizes[stage_idx]
    current_batch_size = batch_sizes[stage_idx]
    dataset_opt['gt_size'] = current_gt_size
    dataset_opt['batch_size_per_gpu'] = current_batch_size

    logger.info(f"Creating Dataloader for Stage {stage_idx + 1}: "
                f"Patch Size = {current_gt_size}, "
                f"Batch Size per GPU = {current_batch_size}")

    dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
    train_set = create_dataset(dataset_opt)
    train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                    opt['rank'], dataset_enlarge_ratio)
    train_loader = create_dataloader(
        train_set,
        dataset_opt,
        num_gpu=opt['num_gpu'],
        dist=opt['dist'],
        sampler=train_sampler,
        seed=opt['manual_seed'] + stage_idx  # Use different seed for each stage's dataloader
    )

    # Calculate iterations per epoch for this specific stage's dataloader
    num_iter_per_epoch = math.ceil(
        len(train_set) * dataset_enlarge_ratio /
        (current_batch_size * opt['world_size']))

    return train_loader, train_sampler, num_iter_per_epoch


# --- MODIFICATION END ---

def parse_options(is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--input_path', type=str, required=False, help='The path to the input image. For single image inference only.')
    parser.add_argument('--output_path', type=str, required=False, help='The path to the output image. For single image inference only.')

    args = parser.parse_args()
    opt = parse(args.opt, is_train=is_train)

    # distributed training settings
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
            print('init dist .. ', args.launcher)

    opt['rank'], opt['world_size'] = get_dist_info()

    # random seed
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])

    if args.input_path is not None and args.output_path is not None:
        opt['img_path'] = {
            'input_img': args.input_path,
            'output_img': args.output_path
        }


    return opt


def init_loggers(opt):
    log_file = osp.join(opt['path']['log'],
                        f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # initialize wandb logger
    if (opt['logger'].get('wandb')
            is not None) and (opt['logger']['wandb'].get('project')
                              is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, (
            'should turn on tensorboard when using wandb')
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        tb_logger = init_tb_logger(log_dir=osp.join('tb_logger', opt['name']))
    return logger, tb_logger


def create_train_val_dataloader(opt, logger):
    # create train and val dataloaders
    train_loader, val_loader = None, None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
            train_set = create_dataset(dataset_opt)
            train_sampler = EnlargedSampler(train_set, opt['world_size'],
                                            opt['rank'], dataset_enlarge_ratio)
            train_loader = create_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed'])

            logger.info(
                f'Number of train images: {len(train_set)}, '
                f'Dataset enlarge ratio: {dataset_enlarge_ratio}')
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            logger.info(
                f'Number of val images/folders in {dataset_opt["name"]}: '
                f'{len(val_set)}')
    return train_loader, val_loader

# def create_val_dataloader(opt, logger):
#     val_loader = None
#     if 'val' in opt['datasets']:
#         dataset_opt = opt['datasets']['val']
#         val_set = create_dataset(dataset_opt)
#         val_loader = create_dataloader(
#             val_set,
#             dataset_opt,
#             num_gpu=opt['num_gpu'],
#             dist=opt['dist'],
#             sampler=None,
#             seed=opt['manual_seed'])
#         logger.info(
#             f'Number of val images/folders in {dataset_opt["name"]}: '
#             f'{len(val_set)}')
#     return val_loader

def create_val_dataloaders(opt, logger):
    val_loaders = {}
    
    for phase, dataset_opt in sorted(opt['datasets'].items()):
        if 'val' in phase:
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None,
                seed=opt['manual_seed'])
            
            dataset_name = dataset_opt["name"]
            logger.info(
                f"Found validation dataset '{dataset_name}' under key '{phase}'. "
                f"Number of images: {len(val_set)}")
            val_loaders[dataset_name] = val_loader
            
    return val_loaders


def log_training_statistics(opt, logger):
    """Calculate and log training statistics, including estimated epochs."""

    # This is a common practice for pre-calculating statistics
    train_set = create_dataset(opt['datasets']['train'])
    dataset_enlarge_ratio = opt['datasets']['train'].get('dataset_enlarge_ratio', 1)

    logger.info(f'Number of train images: {len(train_set)}, '
                f'Dataset enlarge ratio: {dataset_enlarge_ratio}')

    is_progressive = opt['datasets']['train'].get('iters') is not None

    if is_progressive:
        iters_per_stage = opt['datasets']['train']['iters']
        batch_sizes = opt['datasets']['train']['mini_batch_sizes']

        total_estimated_epochs = 0
        logger.info('--- Progressive Training Statistics ---')
        for i, (stage_iters, stage_bs) in enumerate(zip(iters_per_stage, batch_sizes)):
            # Calculate iterations per epoch for this specific stage
            num_iter_per_epoch_stage = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (stage_bs * opt['world_size']))

            # Calculate how many epochs this stage will approximately run for
            epochs_in_stage = stage_iters / num_iter_per_epoch_stage
            total_estimated_epochs += epochs_in_stage

            logger.info(
                f'  [Stage {i + 1}] Iters: {stage_iters}, Batch Size: {stage_bs}, '
                f'Iters/Epoch: {num_iter_per_epoch_stage}, '
                f'Est. Epochs for this stage: {epochs_in_stage:.2f}'
            )
        logger.info(
            f'--- Total Estimated Epochs: {total_estimated_epochs:.2f} ---')

    else:  # Standard training
        dataset_opt = opt['datasets']['train']
        num_iter_per_epoch = math.ceil(
            len(train_set) * dataset_enlarge_ratio /
            (dataset_opt['batch_size_per_gpu'] * opt['world_size']))
        total_iters = int(opt['train']['total_iter'])
        total_epochs = math.ceil(total_iters / num_iter_per_epoch)
        logger.info(
            '--- Standard Training Statistics ---'
            f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
            f'\n\tWorld size (gpu number): {opt["world_size"]}'
            f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
            f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')



def main():
    # parse options, set distributed setting, set ramdom seed
    # torch.autograd.set_detect_anomaly(True)
    opt = parse_options(is_train=True)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # automatic resume ..
    state_folder_path = 'experiments/{}/training_states/'.format(opt['name'])
    try:
        states = os.listdir(state_folder_path)
    except:
        states = []

    resume_state = None
    if len(states) > 0:
        max_state_file = '{}.state'.format(max([int(x.split('.')[0]) for x in states]))
        resume_state = os.path.join(state_folder_path, max_state_file)
        opt['path']['resume_state'] = resume_state

    # load resume states if necessary
    if opt['path'].get('resume_state'):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt['path']['resume_state'],
            map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # mkdir for experiments and logger
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt[
                'name'] and opt['rank'] == 0:
            mkdir_and_rename(osp.join('tb_logger', opt['name']))

    # initialize loggers
    logger, tb_logger = init_loggers(opt)

    # create validation dataloader (created once)
    val_loaders = create_val_dataloaders(opt, logger)

    # --- MODIFICATION START: Conditional training loop based on user config ---
    update_schedule_by = opt['train'].get('update_schedule_by', 'iter').lower()
    is_progressive = 'iters' in opt['datasets']['train']

    # create model
    if resume_state:  # resume training
        update_schedule_by = opt['train'].get('update_schedule_by', 'iter').lower()
        is_progressive = 'iters' in opt['datasets']['train']
        training_mode = resume_state.get('training_mode', 'iter')

        if training_mode == 'epoch' and update_schedule_by == 'epoch' and not is_progressive:
            start_epoch = resume_state['epoch']
            current_iter = start_epoch
            check_resume(opt, start_epoch)
        else:
            # --- Iteration-centric resume logic ---
            start_epoch = resume_state['epoch']
            current_iter = resume_state['iter']
            check_resume(opt, current_iter)

        model = create_model(opt)

        model.resume_training(resume_state)
        
        logger.info(f"Resuming training from epoch: {start_epoch}, "
                    f"iter: {current_iter}.")

    else:
        model = create_model(opt)
        start_epoch = 0
        current_iter = 0
    
    
    # create message logger
    msg_logger = MessageLogger(opt, current_iter, tb_logger)

    if update_schedule_by == 'epoch' and not is_progressive:
        logger.info(f'Start training from epoch: {start_epoch}')
    else:
        logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    start_time = time.time()
    
    net_g_instance = model.net_g.module if hasattr(model.net_g, 'module') else model.net_g
    
    val_params = opt.get('val', {})
    val_save_img = val_params.get('save_img', False)
    val_rgb2bgr = val_params.get('rgb2bgr', True)
    val_use_image = val_params.get('use_image', False)

    if update_schedule_by == 'epoch' and not is_progressive:
        # --- Epoch-based training loop (similar to iteration-based) ---
        logger.info("Starting training with epoch-based scheduler updates.")

        # Create dataloader for standard training
        dataset_opt = opt['datasets']['train']
        train_set = create_dataset(dataset_opt)
        train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'])
        train_loader = create_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'])
        
        # Get total epochs from user config
        total_epochs = int(opt['train']['total_epoch'])
        logger.info(f"Training will run for {total_epochs} epochs.")

        # Dataloader prefetcher
        prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
        if prefetch_mode is None or prefetch_mode == 'cpu':
            prefetcher = CPUPrefetcher(train_loader)
        elif prefetch_mode == 'cuda':
            prefetcher = CUDAPrefetcher(train_loader, opt)
            logger.info(f'Use {prefetch_mode} prefetch dataloader')
            if opt['datasets']['train'].get('pin_memory') is not True:
                raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
        else:
            raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                             "Supported ones are: None, 'cuda', 'cpu'.")

        # Initialize time tracking for epoch mode
        data_time, iter_time = time.time(), time.time()
        
        epoch = start_epoch
        while epoch < total_epochs:   
            epoch += 1  # Increment epoch at the beginning to start from 1
            
            if hasattr(net_g_instance, 'grid'):
                net_g_instance.grid = False
            
            if hasattr(model, 'reset_epoch_stats'):
                model.reset_epoch_stats()
     
            # Calculate data preparation time (from end of last epoch to start of current epoch)
            data_time = time.time() - data_time
            
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()

            # Start timing for this epoch
            iter_time = time.time()

            # Process all batches in this epoch
            while train_data is not None:
                # training
                model.feed_data(train_data, is_val=False)
                
                model.optimize_parameters(epoch, tb_logger)
                
                train_data = prefetcher.next()
                
            model.update_learning_rate(epoch + 1, warmup_iter=opt['train'].get('warmup_iter', -1))
            
            # Calculate epoch time
            epoch_time = time.time() - iter_time
            
            # Log at the end of each epoch
            if epoch % opt['logger']['print_freq'] == 0:
                log_vars = {
                    'epoch': epoch, 
                    'total_epochs': total_epochs,
                    'time': epoch_time,
                    'data_time': data_time
                }
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            
            # Reset time variables for next epoch
            data_time = time.time()
            iter_time = time.time()

            # Save models and training states at the end of each epoch
            if epoch % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, epoch)

            # Validation at the end of each epoch
            if opt.get('val') is not None and (epoch % opt['val']['val_freq'] == 0):
                
                if hasattr(net_g_instance, 'grid'):
                    net_g_instance.grid = True
                
                for val_name, val_loader in val_loaders.items():
                    logger.info(f"Validating on {val_name}...")
                    model.validation(
                        val_loader,
                        epoch,
                        tb_logger,
                        save_img=val_save_img,
                        rgb2bgr=val_rgb2bgr,
                        use_image=val_use_image
                    )
                    if opt['dist']:
                        device_id = torch.cuda.current_device()
                        torch.distributed.barrier(device_ids=[device_id])
            
    else:
        # --- ORIGINAL: Iteration-based training loop ---
        # --- MOVE setup logic here for self-containment ---
        if is_progressive:
            iters_per_stage = opt['datasets']['train']['iters']
            gt_sizes = opt['datasets']['train']['gt_sizes']
            batch_sizes = opt['datasets']['train']['mini_batch_sizes']
            stage_thresholds = np.cumsum(iters_per_stage)
            total_iters = int(stage_thresholds[-1])
            opt['train']['total_iter'] = total_iters
            logger.info("Starting training with iteration-based scheduler updates (Progressive Learning Mode).")
        else:
            total_iters = int(opt['train']['total_iter'])
            logger.info("Starting training with iteration-based scheduler updates (Default Mode).")
        
        log_training_statistics(opt, logger)
        # --- End of moved logic ---

        # Determine starting stage for progressive learning
        if resume_state and is_progressive:
            start_stage_idx = np.searchsorted(stage_thresholds, current_iter, side='right')
        else:
            start_stage_idx = 0

        # Initialize time tracking for iteration mode
        data_time, iter_time = time.time(), time.time()
        
        # Main training loop
        for stage_idx in range(start_stage_idx, len(opt['datasets']['train'].get('iters', [1]))):
            
            # Setup for the current stage if progressive learning is enabled
            if is_progressive:
                train_loader, train_sampler, num_iter_per_epoch = create_progressive_dataloader(
                    opt, logger, stage_idx, gt_sizes, batch_sizes
                )
                stage_end_iter = stage_thresholds[stage_idx]
            else: # Standard training setup
                dataset_opt = opt['datasets']['train']
                train_set = create_dataset(dataset_opt)
                train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'])
                train_loader = create_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=train_sampler, seed=opt['manual_seed'])
                stage_end_iter = total_iters
            
            # Dataloader prefetcher
            prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
            if prefetch_mode is None or prefetch_mode == 'cpu':
                prefetcher = CPUPrefetcher(train_loader)
            elif prefetch_mode == 'cuda':
                prefetcher = CUDAPrefetcher(train_loader, opt)
                logger.info(f'Use {prefetch_mode} prefetch dataloader')
                if opt['datasets']['train'].get('pin_memory') is not True:
                    raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
            else:
                raise ValueError(f'Wrong prefetch_mode {prefetch_mode}.'
                                 "Supported ones are: None, 'cuda', 'cpu'.")

            # Resume epoch count if needed, or start from 0 for the new stage
            epoch = resume_state['epoch'] if stage_idx == start_stage_idx and resume_state else 0

            # Loop for the current stage
            while current_iter <= stage_end_iter:
                train_sampler.set_epoch(epoch)
                prefetcher.reset()
                train_data = prefetcher.next()
                
                while train_data is not None:
                    data_time = time.time() - data_time

                    current_iter += 1
                    
                    if hasattr(net_g_instance, 'grid'):
                        net_g_instance.grid = False
                    model.net_g.train()
            
                    if current_iter > total_iters:
                        break

                    # training
                    model.feed_data(train_data, is_val=False)
                    model.optimize_parameters(current_iter, tb_logger)
                    model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

                    iter_time = time.time() - iter_time

                    # log
                    if current_iter % opt['logger']['print_freq'] == 0:
                        log_vars = {'epoch': epoch, 'iter': current_iter, 'total_iter': total_iters}
                        log_vars.update({'lrs': model.get_current_learning_rate()})
                        log_vars.update({'time': iter_time, 'data_time': data_time})
                        log_vars.update(model.get_current_log())
                        msg_logger(log_vars)

                    # save models and training states
                    if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                        logger.info('Saving models and training states.')
                        model.save(epoch, current_iter)

                    # validation
                    if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0):
                        model.net_g.eval()
                        if hasattr(net_g_instance, 'grid'):
                            net_g_instance.grid = True
                        
                        for val_name, val_loader in val_loaders.items():
                            logger.info(f"Validating on {val_name}...")
                            model.validation(
                                val_loader,
                                current_iter,
                                tb_logger,
                                save_img=val_save_img,
                                rgb2bgr=val_rgb2bgr,
                                use_image=val_use_image
                            )
                            if opt['dist']:
                                device_id = torch.cuda.current_device()
                                torch.distributed.barrier(device_ids=[device_id])
                        
                        if hasattr(net_g_instance, 'grid'):
                            net_g_instance.grid = False
                        model.net_g.train()

                    data_time = time.time()
                    iter_time = time.time()
                    train_data = prefetcher.next()

                epoch += 1
                if current_iter > stage_end_iter and is_progressive:
                    break
                
            if current_iter > total_iters:
                break
    # --- MODIFICATION END ---


    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    
    # ------------------ START: Final Save & Clean Validation Logic ------------------
    logger.info("\n\n" + "="*80)
    logger.info("====== Final Save and Clean Validation in Distributed Mode ======")
    
    if opt['rank'] == 0:
        logger.info("Rank 0: Saving the final model state as 'latest'...")
        model.save(epoch=-1, current_iter=-1) # 这会创建 net_g_latest.pth 和 training_state_latest.state
    
    # 使用 barrier 确保所有进程都等待 rank 0 保存完毕
    if opt['dist']:
        torch.distributed.barrier()

    final_model_path = osp.join(opt['path']['models'], 'net_g_latest.pth')

    if not osp.exists(final_model_path):
        # 只有 rank 0 打印警告，避免信息刷屏
        if opt['rank'] == 0:
            logger.warning(f"Failed to find the final model at {final_model_path}. Skipping final validation.")
    else:
        if opt['rank'] == 0:
            logger.info(f"All ranks will now validate using model: {final_model_path}")
        
        val_opt = deepcopy(opt)
        val_opt['is_train'] = False

        val_opt['path']['pretrain_network_g'] = final_model_path
        val_opt['path']['resume_state'] = None

        try:
            if opt['rank'] == 0:
                logger.info('Creating a clean model instance for final validation...')
            validation_model = create_model(val_opt)
            if opt['rank'] == 0:
                logger.info('Clean validation model created successfully.')

            if val_loaders:
                if opt['rank'] == 0:
                    logger.info('Performing final validation on the clean model...')
                
                validation_model.net_g.eval()
                val_net_g_instance = validation_model.net_g.module if hasattr(validation_model.net_g, 'module') else validation_model.net_g
                if hasattr(val_net_g_instance, 'grid'):
                    val_net_g_instance.grid = True

                validation_iter_label = 'final_clean_validation'
                
                for val_name, val_loader in val_loaders.items():
                    logger.info(f"Final validation on {val_name}...")
                    validation_model.validation(
                        val_loader,
                        validation_iter_label,
                        tb_logger,
                        save_img=val_save_img,
                        rgb2bgr=val_rgb2bgr,
                        use_image=val_use_image
                    )
                    if opt['dist']:
                        device_id = torch.cuda.current_device()
                        torch.distributed.barrier(device_ids=[device_id])
            
            del validation_model
            torch.cuda.empty_cache()
            if opt['rank'] == 0:
                logger.info("====== Final Clean Validation Completed Successfully ======")

        except Exception as e:
            if opt['rank'] == 0:
                logger.error(f"====== Final Clean Validation Failed with an exception: {e} ======")
                import traceback
                logger.error(traceback.format_exc())
    
    if tb_logger:
        tb_logger.close()


if __name__ == '__main__':
    os.environ['GRPC_POLL_STRATEGY'] = 'epoll1'
    main()
