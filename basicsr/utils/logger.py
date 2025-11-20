# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import datetime
import logging
import time

from .dist_util import get_dist_info, master_only

_file_handler_added = False


class MessageLogger():
    """Message logger for printing.

    Args:
        opt (dict): Config. It contains the following keys:
            name (str): Exp name.
            logger (dict): Contains 'print_freq' (str) for logger interval.
            train (dict): Contains 'total_iter' (int) for total iters.
            use_tb_logger (bool): Use tensorboard logger.
        start_iter (int): Start iter. Default: 1.
        tb_logger (obj:`tb_logger`): Tensorboard logger. Default： None.
    """

    def __init__(self, opt, start_iter=1, tb_logger=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        
        # Check training mode
        update_schedule_by = opt['train'].get('update_schedule_by', 'iter').lower()
        is_progressive = 'iters' in opt['datasets']['train']
        
        if update_schedule_by == 'epoch' and not is_progressive:
            # Epoch mode: use total_epochs from config
            self.max_iters = int(opt['train']['total_epoch'])
            self.training_mode = 'epoch'
        else:
            # Iter mode: use total_iter
            self.max_iters = opt['train']['total_iter']
            self.training_mode = 'iter'
        
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        """Format logging message.

        Args:
            log_vars (dict): It contains the following keys:
                epoch (int): Epoch number.
                iter (int): Current iter (optional in epoch mode).
                total_iter (int): Total iterations (optional in epoch mode).
                total_epochs (int): Total epochs (optional in epoch mode).
                lrs (list): List for learning rates.
                time (float): Iter time.
                data_time (float): Data time for each iter.
        """
        # epoch, learning rates
        epoch = log_vars.pop('epoch')
        lrs = log_vars.pop('lrs')

        # Handle different training modes
        if self.training_mode == 'epoch':
            # Epoch mode: use epoch-based logging
            if 'total_epochs' in log_vars:
                total_epochs = log_vars.pop('total_epochs')
            else:
                total_epochs = self.max_iters
            
            message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}/{total_epochs:3d}, '
                       f'lr:(')
            for v in lrs:
                message += f'{v:.3e},'
            message += ')] '

            # time and estimated time for epoch mode
            if 'time' in log_vars.keys():
                iter_time = log_vars.pop('time')
                data_time = log_vars.pop('data_time')

                total_time = time.time() - self.start_time
                time_sec_avg = total_time / (epoch - self.start_iter + 1)
                eta_sec = time_sec_avg * (total_epochs - epoch - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                message += f'[eta: {eta_str}, '
                message += f'time (data): {iter_time:.1f}s ({data_time:.1f}s)] '

            # other items, especially losses
            for k, v in log_vars.items():
                message += f'{k}: {v:.8f} '
                # tensorboard logger for epoch mode
                if self.use_tb_logger and 'debug' not in self.exp_name:
                    normed_step = 10000 * (epoch / total_epochs)
                    normed_step = int(normed_step)

                    if k.startswith('l_'):
                        self.tb_logger.add_scalar(f'losses/{k}', v, normed_step)
                    elif k.startswith('m_'):
                        self.tb_logger.add_scalar(f'metrics/{k}', v, normed_step)
                    else:
                        assert 1 == 0

        else:
            # Iter mode: use iteration-based logging
            current_iter = log_vars.pop('iter')
            total_iter = log_vars.pop('total_iter')

            message = (f'[{self.exp_name[:5]}..][epoch:{epoch:3d}, '
                       f'iter:{current_iter:8,d}, lr:(')
            for v in lrs:
                message += f'{v:.3e},'
            message += ')] '

            # time and estimated time for iter mode
            if 'time' in log_vars.keys():
                iter_time = log_vars.pop('time')
                data_time = log_vars.pop('data_time')

                total_time = time.time() - self.start_time
                time_sec_avg = total_time / (current_iter - self.start_iter + 1)
                eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
                eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
                message += f'[eta: {eta_str}, '
                message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '

            # other items, especially losses
            for k, v in log_vars.items():
                message += f'{k}: {v:.8f} '
                # tensorboard logger for iter mode
                if self.use_tb_logger and 'debug' not in self.exp_name:
                    normed_step = 10000 * (current_iter / total_iter)
                    normed_step = int(normed_step)

                    if k.startswith('l_'):
                        self.tb_logger.add_scalar(f'losses/{k}', v, normed_step)
                    elif k.startswith('m_'):
                        self.tb_logger.add_scalar(f'metrics/{k}', v, normed_step)
                    else:
                        assert 1 == 0

        self.logger.info(message)


@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger


@master_only
def init_wandb_logger(opt):
    """We now only use wandb to sync tensorboard log."""
    import wandb
    logger = logging.getLogger('basicsr')

    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'

    wandb.init(
        id=wandb_id,
        resume=resume,
        name=opt['name'],
        config=opt,
        project=project,
        sync_tensorboard=True)

    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')


def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    
    global _file_handler_added

    logger = logging.getLogger()

    if not logger.hasHandlers() or _file_handler_added is False:
        logger.handlers.clear()
        logger.setLevel(log_level)
        format_str = '%(asctime)s %(levelname)s: %(message)s'
        formatter = logging.Formatter(format_str)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    rank, _ = get_dist_info()
    if rank == 0 and log_file is not None and not _file_handler_added:
        format_str = '%(asctime)s %(levelname)s: %(message)s'
        formatter = logging.Formatter(format_str)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
        _file_handler_added = True

    if rank != 0:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(log_level)

    return logging.getLogger(logger_name)


def get_env_info():
    """Get environment information.

    Currently, only log the software version.
    """
    import torch
    import torchvision

    from basicsr.version import __version__
    msg = r"""
                ____                _       _____  ____
               / __ ) ____ _ _____ (_)_____/ ___/ / __ \
              / __  |/ __ `// ___// // ___/\__ \ / /_/ /
             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/
            /_____/ \__,_//____//_/ \___//____//_/ |_|
     ______                   __   __                 __      __
    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /
   / / __ / __ \ / __ \ / __  /  / /   / / / // ___// //_/  / /
  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/
  \____/ \____/ \____/ \____/  /_____/\____/ \___//_/|_|  (_)
    """
    msg += ('\nVersion Information: '
            f'\n\tBasicSR: {__version__}'
            f'\n\tPyTorch: {torch.__version__}'
            f'\n\tTorchVision: {torchvision.__version__}')
    return msg
