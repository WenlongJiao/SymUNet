# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import cv2
import numpy as np
import os

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img, augment_images, tensor2np, np2tensor, image_align
from basicsr.utils.dist_util import get_dist_info
from basicsr.models.optimizers import Muon, MuonWithAuxAdam

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')


# Function for grid inverse with kernel fusion
@torch.jit.script
def grids_inverse_kernel(
    outs: torch.Tensor,
    preds: torch.Tensor,
    indices_i: torch.Tensor,
    indices_j: torch.Tensor,
    fuse_matrix_h1: torch.Tensor,
    fuse_matrix_h2: torch.Tensor,
    fuse_matrix_eh1: torch.Tensor,
    fuse_matrix_eh2: torch.Tensor,
    fuse_matrix_w1: torch.Tensor,
    fuse_matrix_w2: torch.Tensor,
    fuse_matrix_ew1: torch.Tensor,
    fuse_matrix_ew2: torch.Tensor,
    k1: int, k2: int, h: int, w: int,
    grid_overlap_size_h: int,
    grid_overlap_size_w: int,
    ek1: int, ek2: int
) -> torch.Tensor:
    
    for cnt in range(outs.shape[0]):
        i = indices_i[cnt].item()
        j = indices_j[cnt].item()
        
        part = outs[cnt:cnt+1].clone() 

        if i != 0 and i + k1 != h:
            part[:, :, :grid_overlap_size_h, :] *= fuse_matrix_h2
        if i + k1 * 2 - ek1 < h:
            part[:, :, -grid_overlap_size_h:, :] *= fuse_matrix_h1
        if ek1 > 0 and i + k1 == h:
            part[:, :, :ek1, :] *= fuse_matrix_eh2
        if ek1 > 0 and i + k1 * 2 - ek1 == h:
            part[:, :, -ek1:, :] *= fuse_matrix_eh1
            
        if j != 0 and j + k2 != w:
            part[:, :, :, :grid_overlap_size_w] *= fuse_matrix_w2
        if j + k2 * 2 - ek2 < w:
            part[:, :, :, -grid_overlap_size_w:] *= fuse_matrix_w1

        if ek2 > 0 and j + k2 == w:
            part[:, :, :, :ek2] *= fuse_matrix_ew2
        if ek2 > 0 and j + k2 * 2 - ek2 == w:
            part[:, :, :, -ek2:] *= fuse_matrix_ew1
            
        preds[0, :, i:i + k1, j:j + k2] += part.squeeze(0)

    return preds


class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), 
                              param_key=self.opt['path'].get('param_key', 'params'),
                              remove_prefix=self.opt['path'].get('remove_prefix', None))

        # for lightfrenet
        if not self.is_train:
            model_to_deploy = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g

            if hasattr(model_to_deploy, 'switch_to_deploy'):
                print("Model is being switched to deploy mode...")

                model_to_deploy.switch_to_deploy()

                logger = get_root_logger()
                logger.info('Network [net_g] has been successfully switched to deploy mode for fast inference.')

        if self.is_train:
            self.init_training_settings()
            self.epoch_loss_tracker = OrderedDict()
            self.batch_count_in_epoch = 0

        self.scale = int(opt['scale'])
        self.color_augs = self.opt.get('train', {}).get('color_augs', False)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            percep_type = train_opt['perceptual_opt'].pop('type')
            cri_perceptual_cls = getattr(loss_module, percep_type)
            self.cri_perceptual = cri_perceptual_cls(
                **train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        if train_opt.get('fft_loss_opt'):
            fft_type = train_opt['fft_loss_opt'].pop('type')
            cri_fft_cls = getattr(loss_module, fft_type)
            self.cri_fft = cri_fft_cls(**train_opt['fft_loss_opt']).to(
                self.device)
        else:
            self.cri_fft = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def reset_epoch_stats(self):
        """Called at the beginning of each new epoch to reset accumulators."""
        self.epoch_loss_tracker.clear()
        self.batch_count_in_epoch = 0

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1
        
        optim_config = train_opt['optim_g'].copy()
        optim_type = optim_config.pop('type')
        
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **optim_config)
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **optim_config)
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **optim_config)
        elif optim_type == 'RAdam':
            self.optimizer_g = torch.optim.RAdam([{'params': optim_params}], **optim_config)
        elif optim_type == 'NAdam':
            self.optimizer_g = torch.optim.NAdam([{'params': optim_params}], **optim_config)
        elif optim_type == 'MuonWithAuxAdam':
            muon_params = []
            adam_params = []
            for n, p in self.net_g.named_parameters():
                if p.requires_grad:
                    if p.ndim >= 2:
                        muon_params.append(p)
                    else:
                        adam_params.append(p)
                        
            muon_hparams = optim_config.get('muon_hparams', {'lr': 0.02, 'momentum': 0.95})
            adam_hparams = optim_config.get('adam_hparams', {'lr': 3e-4, 'betas': (0.9, 0.99)})
            
            param_groups_for_muon = [
                {'params': muon_params, 'use_muon': True, **muon_hparams},
                {'params': adam_params, 'use_muon': False, **adam_hparams}
            ]
            
            self.optimizer_g = MuonWithAuxAdam(param_groups_for_muon)
            pass
        
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)
        
    def update_optimizer_for_new_params(self):
        logger = get_root_logger()
        net_g_instance = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
        
        current_model_params = {p for p in net_g_instance.parameters() if p.requires_grad}
        
        optimizer_params = set()
        for group in self.optimizer_g.param_groups:
            optimizer_params.update(group['params'])
            
        newly_unfrozen_params = list(current_model_params - optimizer_params)
        
        if newly_unfrozen_params:
            logger.info(f"Found {len(newly_unfrozen_params)} newly unfrozen parameters. Adding them to the optimizer.")
            
            base_lr = self.optimizer_g.param_groups[0]['lr']
            
            new_param_group = {
                'params': newly_unfrozen_params,
                'lr': base_lr
            }
            
            self.optimizer_g.add_param_group(new_param_group)
            
            logger.info(f"Optimizer now has {len(self.optimizer_g.param_groups)} parameter groups.")
        else:
            logger.warning("Optimizer update was triggered, but no new unfrozen parameters were found.")

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.lq.size()
        self.original_size = (b, c, h, w)
        assert b == 1, "Validation grids mode only supports batch_size=1"

        k1 = self.opt['val'].get('crop_size_h', 512)
        k2 = self.opt['val'].get('crop_size_w', 512)

        overlap_h = self.opt['val'].get('overlap_h', 16)
        overlap_w = self.opt['val'].get('overlap_w', 16)
        self.grid_overlap_size = (overlap_h, overlap_w)
        
        k1 = min(h, k1)
        k2 = min(w, k2)
        self.grid_kernel_size = (k1, k2)

        stride_h = k1 - overlap_h
        stride_w = k2 - overlap_w
        self.stride = (stride_h, stride_w)

        self.nr = (h - overlap_h - 1) // stride_h + 1 if stride_h > 0 else 1
        self.nc = (w - overlap_w - 1) // stride_w + 1 if stride_w > 0 else 1

        step_j = k2 if self.nc == 1 else stride_w
        step_i = k1 if self.nr == 1 else stride_h

        # 4. 切割
        parts = []
        idxes = []
        i = 0
        last_i = False

        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                
                # Crop
                parts.append(self.lq[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                
                j = j + step_j
            i = i + step_i
            
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes
        
        self.ek1 = max(0, self.nr * stride_h + overlap_h * 2 - h)
        self.ek2 = max(0, self.nc * stride_w + overlap_w * 2 - w)

    def _init_overlap_matrices(self, device, dtype):
        overlap_h, overlap_w = self.grid_overlap_size
        
        self.fuse_matrix_w1 = torch.linspace(1., 0., overlap_w, device=device, dtype=dtype).view(1, 1, 1, overlap_w)
        self.fuse_matrix_w2 = torch.linspace(0., 1., overlap_w, device=device, dtype=dtype).view(1, 1, 1, overlap_w)
        
        self.fuse_matrix_h1 = torch.linspace(1., 0., overlap_h, device=device, dtype=dtype).view(1, 1, overlap_h, 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., overlap_h, device=device, dtype=dtype).view(1, 1, overlap_h, 1)
        
        if self.ek2 > 0:
            self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2, device=device, dtype=dtype).view(1, 1, 1, self.ek2)
            self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2, device=device, dtype=dtype).view(1, 1, 1, self.ek2)
        else: 
            self.fuse_matrix_ew1 = torch.empty((1,1,1,0), device=device, dtype=dtype)
            self.fuse_matrix_ew2 = torch.empty((1,1,1,0), device=device, dtype=dtype)
            
        if self.ek1 > 0:
            self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1, device=device, dtype=dtype).view(1, 1, self.ek1, 1)
            self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1, device=device, dtype=dtype).view(1, 1, self.ek1, 1)
        else: 
            self.fuse_matrix_eh1 = torch.empty((1,1,0,1), device=device, dtype=dtype)
            self.fuse_matrix_eh2 = torch.empty((1,1,0,1), device=device, dtype=dtype)

    def grids_inverse(self):
        b, _, h, w = self.original_size
        out_c = self.outs.shape[1]
        
        device = self.outs.device
        dtype = self.outs.dtype
        preds = torch.zeros((b, out_c, h, w), device=device, dtype=dtype)
        
        self._init_overlap_matrices(device, dtype)
        
        indices_i = torch.tensor([d['i'] for d in self.idxes], device=device, dtype=torch.long)
        indices_j = torch.tensor([d['j'] for d in self.idxes], device=device, dtype=torch.long)
        
        k1, k2 = self.grid_kernel_size
        overlap_h, overlap_w = self.grid_overlap_size

        preds = grids_inverse_kernel(
            self.outs, preds, indices_i, indices_j,
            self.fuse_matrix_h1, self.fuse_matrix_h2, self.fuse_matrix_eh1, self.fuse_matrix_eh2,
            self.fuse_matrix_w1, self.fuse_matrix_w2, self.fuse_matrix_ew1, self.fuse_matrix_ew2,
            k1, k2, h, w,
            overlap_h, overlap_w,
            self.ek1, self.ek2
        )
        
        self.output = preds.to(self.device)
        
        del self.outs
        del preds

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        # if self.opt['train'].get('mixup', False):
        #     self.mixup_aug()

        if self.color_augs:
            augmented_lq, augmented_gt, mask, aug_name = augment_images(self.lq, self.gt)
            self.lq = augmented_lq
            self.gt = augmented_gt

        net_g_instance = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
        
        phase_before = getattr(net_g_instance, 'training_phase', -1)
        
        if hasattr(net_g_instance, 'semantic_start_iter') and net_g_instance.semantic_start_iter > 0:
            preds = self.net_g(self.lq, current_iter=current_iter)
        else:
            preds = self.net_g(self.lq)
            
        phase_after = getattr(net_g_instance, 'training_phase', -1)
        
        if phase_before == 1 and phase_after == 2:
            self.update_optimizer_for_new_params()
            
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        if self.cri_fft:
            l_fft = 0.
            
            for pred in preds:
                l_fft += self.cri_fft(pred, self.gt)
            
            l_total += l_fft
            loss_dict['l_fft'] = l_fft


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        
        grad_clip_config = self.opt['train'].get('grad_clip')
        if grad_clip_config:
            max_norm = grad_clip_config['max_norm']
            norm_type = grad_clip_config.get('norm_type', 2)
            
            torch.nn.utils.clip_grad_norm_(
                parameters=self.net_g.parameters(), 
                max_norm=max_norm, 
                norm_type=norm_type
            )
            
        self.optimizer_g.step()

        reduced_loss_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict = reduced_loss_dict
        
        for key, value in reduced_loss_dict.items():
            self.epoch_loss_tracker[key] = self.epoch_loss_tracker.get(key, 0) + value
        self.batch_count_in_epoch += 1

    def test(self):
        """
        分批次推理，防止显存爆炸 (Minibatch Processing)
        """
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            self.outs = [] # 初始化结果列表
            
            m = self.opt['val'].get('max_minibatch', 8)
            i = 0

            apply_padding = self.opt['val'].get('padding', False)
            if apply_padding:
                padding_size = self.opt['val'].get('padding_size', 8)
                padding_mode = self.opt['val'].get('padding_mode', 'reflect')

            while i < n:
                j = i + m
                if j >= n: j = n
                
                current_batch = self.lq[i:j]
                b, c, h, w = current_batch.shape
                
                if apply_padding:
                    h_pad = (padding_size - h % padding_size) % padding_size
                    w_pad = (padding_size - w % padding_size) % padding_size
                    if h_pad > 0 or w_pad > 0:
                        input_batch = F.pad(current_batch, (0, w_pad, 0, h_pad), mode=padding_mode)
                    else:
                        input_batch = current_batch
                else:
                    input_batch = current_batch

                pred_output = self.net_g(input_batch)
                
                if isinstance(pred_output, tuple): pred = pred_output[0]
                else: pred = pred_output
                if isinstance(pred, list): pred = pred[-1]

                if apply_padding and (h_pad > 0 or w_pad > 0):
                    pred = pred[:, :, :h, :w]

                for k in range(pred.shape[0]):
                    self.outs.append(pred[k].detach().cpu())
                
                i = j

            self.outs = torch.stack(self.outs, dim=0)
            
            if not self.opt['val'].get('grids', False):
                self.output = self.outs.to(self.device)
            
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()

        use_align = self.opt['val'].get('use_align', False)
        logger = get_root_logger()
        if use_align and rank == 0:
            logger.info('Image alignment is enabled for validation.')

        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0

        all_psnr = []  # 临时收集psnr
        all_ssim = []  # 临时收集ssim
        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data, is_val=True)
            
            if self.opt['val'].get('grids', False):
                self.grids()

            self.test()

            if self.opt['val'].get('grids', False):
                self.grids_inverse()

            if use_align:
                sr_img_to_save, gt_img_to_save = None, None

                if hasattr(self, 'gt'):
                    try:
                        sr_np_float = tensor2np(self.output)
                        gt_np_float = tensor2np(self.gt)

                        aligned_sr_np, aligned_gt_np, mask_np, _ = image_align(sr_np_float, gt_np_float)

                        if with_metrics:
                            opt_metric = deepcopy(self.opt['val']['metrics'])
                            for name, opt_ in opt_metric.items():
                                metric_type = opt_.pop('type')
                                value = getattr(metric_module, metric_type)(aligned_gt_np, aligned_sr_np, mask_np, **opt_)
                                self.metric_results[name] += value
                                # 临时收集
                                if name == 'psnr':
                                    all_psnr.append(value)
                                elif name == 'ssim':
                                    all_ssim.append(value)

                        sr_img_to_save = (aligned_sr_np * 255.).round().astype(np.uint8)
                        gt_img_to_save = (aligned_gt_np * 255.).round().astype(np.uint8)

                    except Exception as e:
                        if rank == 0:
                            logger.error(f"Alignment or metric calculation failed for {img_name}: {e}. Skipping metrics for this image.")

                        sr_img_to_save = tensor2img(self.output, rgb2bgr=False, out_type=np.uint8)
                        gt_img_to_save = tensor2img(self.gt, rgb2bgr=False, out_type=np.uint8) if hasattr(self, 'gt') else None
                else:
                    sr_img_to_save = tensor2img(self.output, rgb2bgr=False, out_type=np.uint8)

                if save_img:
                    if sr_img_to_save is not None:
                        imwrite(cv2.cvtColor(sr_img_to_save, cv2.COLOR_RGB2BGR), self.get_save_path('sr', img_name, current_iter, dataset_name))
                    if gt_img_to_save is not None:
                        imwrite(cv2.cvtColor(gt_img_to_save, cv2.COLOR_RGB2BGR), self.get_save_path('gt', img_name, current_iter, dataset_name))

            else:
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
                if 'gt' in visuals:
                    gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)

                if save_img:
                    if sr_img.shape[2] == 6:
                        L_img, R_img = sr_img[:, :, :3], sr_img[:, :, 3:]
                        visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                        imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                        imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                    else:
                        save_img_path = self.get_save_path('sr', img_name, current_iter, dataset_name)
                        save_gt_img_path = self.get_save_path('gt', img_name, current_iter, dataset_name)
                        imwrite(sr_img, save_img_path)
                        if 'gt' in visuals:
                            imwrite(gt_img, save_gt_img_path)

                if with_metrics and 'gt' in visuals:
                    # This block is from your original code
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(sr_img, gt_img, **opt_)
                    else:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                                
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            # Tentative for out of GPU memory
            if hasattr(self, 'gt'): del self.gt
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # 临时保存align分支下的psnr/ssim
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.
        
    

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)

    def get_save_path(self, img_type, img_name, current_iter, dataset_name):
        """Helper function to generate save paths."""
        if self.opt['is_train']:
            if img_type == 'sr':
                return osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}.png')
            else:  # gt
                return osp.join(self.opt['path']['visualization'], img_name, f'{img_name}_{current_iter}_gt.png')
        else:
            if img_type == 'sr':
                return osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}.png')
            else:  # gt
                return osp.join(self.opt['path']['visualization'], dataset_name, f'{img_name}_gt.png')

    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
    
    def get_current_log(self):
        is_epoch_mode = 'total_epoch' in self.opt['train']
        
        if is_epoch_mode and self.batch_count_in_epoch > 0:
            avg_log = OrderedDict()
            for key, value in self.epoch_loss_tracker.items():
                avg_log[key] = value / self.batch_count_in_epoch
            return avg_log
        else:
            return self.log_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


class ImageRestorationModelWithReblur(ImageRestorationModel):
    """
    Image restoration model for ConFrENet, which incorporates a re-blur consistency loss.
    """
    def init_training_settings(self):
        # This will set up schedulers, optimizers, and the main pixel/perceptual losses.
        super().init_training_settings()

        train_opt = self.opt['train']

        # Define re-blur consistency loss
        if train_opt.get('reblur_opt'):
            reblur_type = train_opt['reblur_opt'].pop('type')
            cri_reblur_cls = getattr(loss_module, reblur_type)
            self.cri_reblur = cri_reblur_cls(**train_opt['reblur_opt']).to(self.device)
            # Define loss weights
            self.l_recon_w = train_opt.get('l_recon_w', 1.0)
            self.l_reblur_w = train_opt.get('l_reblur_w', 0.5)
            self.l_aux_w = train_opt.get('l_aux_w', 0.25)
        else:
            self.cri_reblur = None

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        # ConFrENet returns a dictionary of outputs during training
        outputs = self.net_g(self.lq)

        self.output = outputs['fused_output'] # for visualization and base metrics

        l_total = 0
        loss_dict = OrderedDict()

        # Main reconstruction loss on the final fused image
        if self.cri_pix:
            l_recon = self.l_recon_w * self.cri_pix(self.output, self.gt)
            l_total += l_recon
            loss_dict['l_recon'] = l_recon

        # Re-blur consistency loss (L1 Loss is assumed to be configured in YAML)
        if self.cri_reblur and 'reblurred_output' in outputs:
            l_reblur = self.l_reblur_w * self.cri_reblur(outputs['reblurred_output'], self.lq)
            l_total += l_reblur
            loss_dict['l_reblur'] = l_reblur

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0

            while i < n:
                j = i + m
                if j >= n:
                    j = n

                # In eval mode, the network returns (fused_output, kernel)
                pred, _ = self.net_g(self.lq[i:j])

                if isinstance(pred, list):
                    pred = pred[-1]

                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()



class ImageRestorationModelWithUncertainty(ImageRestorationModel):
    """
    Image Restoration Model with Uncertainty Estimation.
    Inherits from the standard ImageRestorationModel and modifies the training pipeline.
    """

    def __init__(self, opt):
        # Call the parent's __init__ to set up everything as before
        super(ImageRestorationModelWithUncertainty, self).__init__(opt)
        
        # The parent __init__ already handles network creation, loading, etc.
        # We just need to ensure the loss is correctly identified.

    def init_training_settings(self):
        super().init_training_settings() 
        
        self.net_g.train()
        train_opt = self.opt['train']

        # The primary loss MUST be the GaussianNLLLoss for this model.
        if train_opt.get('uncertainty_opt'):
            pixel_type_uncert = train_opt['uncertainty_opt'].pop('type')
            cri_uncert_cls = getattr(loss_module, pixel_type_uncert)
            self.cri_uncertainty = cri_uncert_cls(**train_opt['uncertainty_opt']).to(self.device)
        else:
            # 对于这个不确定性模型，这个损失是核心，所以设为必须
            raise ValueError("Uncertainty model requires 'uncertainty_opt' in the training options.")

    def optimize_parameters(self, current_iter, tb_logger):
        # This is the core part that needs to be changed.
        self.optimizer_g.zero_grad()

        if self.color_augs:
            augmented_lq, augmented_gt, mask, aug_name = augment_images(self.lq, self.gt)
            self.lq = augmented_lq
            self.gt = augmented_gt

        # MODIFICATION: The network now returns two outputs
        pred_mean, pred_log_var = self.net_g(self.lq)
        
        # For compatibility with perceptual/style losses, we define `self.output` as the mean prediction.
        self.output = pred_mean 

        l_total = 0
        loss_dict = OrderedDict()
        
        if hasattr(self, 'cri_uncertainty') and self.cri_uncertainty:
            l_uncertainty = self.cri_uncertainty(pred_mean, pred_log_var, self.gt)
            l_total += l_uncertainty
            loss_dict['l_uncertainty'] = l_uncertainty

        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt) 
            l_total += l_pix

            loss_dict['l_pix'] = l_pix 

        # Auxiliary losses are calculated on the mean prediction (self.output)
        # The logic for perceptual and FFT loss remains identical.
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        if self.cri_fft:
            l_fft = self.cri_fft(self.output, self.gt)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft

        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        
        grad_clip_config = self.opt['train'].get('grad_clip')
        if grad_clip_config:
            torch.nn.utils.clip_grad_norm_(
                parameters=self.net_g.parameters(), 
                **grad_clip_config
            )
            
        self.optimizer_g.step()

        reduced_loss_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict = reduced_loss_dict
        
        for key, value in reduced_loss_dict.items():
            self.epoch_loss_tracker[key] = self.epoch_loss_tracker.get(key, 0) + value
        self.batch_count_in_epoch += 1

    def test(self):
        # Modify the test function to handle two outputs from the network.
        self.net_g.eval()
        with torch.no_grad():
            # The rest of the logic (batching, padding) can be the same.
            # We are interested in the mean prediction for evaluation metrics.
            
            # --- The following logic is copied from your original `test` method ---
            # --- with a single change to handle the model's output ---
            n = len(self.lq)
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0
            apply_padding = self.opt['val'].get('padding', False)
            padding_size = self.opt['val'].get('padding_size', 8)
            padding_mode = self.opt['val'].get('padding_mode', 'reflect')

            while i < n:
                j = i + m
                if j >= n:
                    j = n
                current_batch = self.lq[i:j]
                b, c, h, w = current_batch.shape

                if apply_padding:
                    h_pad = (padding_size - h % padding_size) % padding_size
                    w_pad = (padding_size - w % padding_size) % padding_size
                    if h_pad > 0 or w_pad > 0:
                        padded_batch = F.pad(current_batch, (0, w_pad, 0, h_pad), mode=padding_mode)
                    else:
                        padded_batch = current_batch
                    
                    # MODIFICATION: Get the mean prediction
                    pred, _ = self.net_g(padded_batch) 
                    pred = pred[:, :, :h, :w]
                else:
                    # MODIFICATION: Get the mean prediction
                    pred, _ = self.net_g(current_batch) 

                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()
        
        
class ImageRestorationWithTextModel(ImageRestorationModel):
    """
    An extension of ImageRestorationModel that incorporates a semantic loss 
    based on CLIP features, initialized in a unified way.
    """
    
    def init_training_settings(self):
        """
        Overrides the parent method to initialize all training settings, 
        including the new semantic loss, following the project's established pattern.
        """
        # 1. 调用父类方法，完成 cri_pix, cri_perceptual, optimizers 等的初始化
        super(ImageRestorationWithTextModel, self).init_training_settings()

        train_opt = self.opt['train']
        
        # 2. 遵循与 cri_pix 完全相同的模式来初始化 cri_semantic
        if train_opt.get('semantic_opt'):
            semantic_opt_config = train_opt['semantic_opt'].copy()
            semantic_type = semantic_opt_config.pop('type')
            cri_semantic_cls = getattr(loss_module, semantic_type)
            self.cri_semantic = cri_semantic_cls(**semantic_opt_config).to(self.device)
        else:
            self.cri_semantic = None
            
    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.color_augs:
            augmented_lq, augmented_gt, _, _ = augment_images(self.lq, self.gt)
            self.lq = augmented_lq
            self.gt = augmented_gt
            
        preds_and_z = self.net_g(self.lq, current_iter=current_iter)
        
        if isinstance(preds_and_z, tuple) and len(preds_and_z) == 2:
            preds, z_final = preds_and_z
        else:
            preds, z_final = preds_and_z, None

        if not isinstance(preds, list):
            preds = [preds]
        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        
        # --- 原有损失计算 ---
        if self.cri_pix:
            l_pix = sum(self.cri_pix(pred, self.gt) for pred in preds)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
                
        if self.cri_fft:
            l_fft = sum(self.cri_fft(pred, self.gt) for pred in preds)
            l_total += l_fft
            loss_dict['l_fft'] = l_fft

        # --- 新增语义损失计算 ---
        if self.cri_semantic and z_final is not None:
            model_unwrapped = self.net_g.module if hasattr(self.net_g, 'module') else self.net_g
            with torch.no_grad():
                z_gt_raw = model_unwrapped._get_clip_features(self.gt)
                
            if hasattr(model_unwrapped, 'image_adapter'):
                with torch.no_grad():
                    z_gt_target = model_unwrapped.image_adapter(z_gt_raw)
            else:
                z_gt_target = z_gt_raw

            l_semantic = self.cri_semantic(z_final, z_gt_target)
            
            l_total += l_semantic
            loss_dict['l_semantic'] = l_semantic

        l_total.backward()
        
        if self.opt['train'].get('grad_clip'):
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), **self.opt['train']['grad_clip'])
            
        self.optimizer_g.step()

        reduced_loss_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict = reduced_loss_dict
        
        for key, value in reduced_loss_dict.items():
            self.epoch_loss_tracker[key] = self.epoch_loss_tracker.get(key, 0) + value
        self.batch_count_in_epoch += 1