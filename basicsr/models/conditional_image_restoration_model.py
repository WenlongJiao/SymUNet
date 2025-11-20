import torch
import torch.nn as nn
import torchvision.transforms as T
import kornia.augmentation as K

from collections import OrderedDict
from basicsr.utils import get_root_logger

from basicsr.models.image_restoration_model import ImageRestorationModel

class ConditionalImageRestorationModel(ImageRestorationModel):
    """Conditional Image Restoration model.

    Inherits from ImageRestorationModel and overrides methods for handling conditional inputs.
    It consistently uses the 'gt' tensor as the condition image for both training and testing.
    """

    def __init__(self, opt):
        # The parent __init__ handles all standard setup.
        super(ConditionalImageRestorationModel, self).__init__(opt)
        self.condition_augmentor = ConditionAugmentation().to(self.device)
        logger = get_root_logger()
        logger.info("Initialized ConditionalImageRestorationModel.")
        logger.info("IMPORTANT: 'gt' tensor will be used as the condition image for BOTH training and testing.")

    def feed_data(self, data, is_val=False):
        """
        Overrides feed_data to set the condition image from GT.
        """
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.cond = self.gt

    def optimize_parameters(self, current_iter, tb_logger):
        """
        Overrides optimize_parameters to call the network with two inputs.
        The loss calculation and optimization steps are inherited.
        """
        self.optimizer_g.zero_grad()
        
        with torch.no_grad():
            condition_image = self.condition_augmentor(self.gt)

        # if self.color_augs:
        #     # Assuming augment_images function is available
        #     augmented_lq, augmented_gt, _, _ = augment_images(self.lq, self.gt)
        #     self.lq = augmented_lq
        #     self.gt = augmented_gt
        #     self.cond = augmented_gt

        # --- CORE MODIFICATION: Call the network with lq and cond ---
        preds = self.net_g(self.lq, condition_image)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        # --- The rest of the logic is handled by the original implementation ---
        # Call a helper method to calculate loss to keep it DRY, or duplicate the code.
        # For simplicity, we duplicate the loss calculation here.
        l_total = 0
        loss_dict = OrderedDict()

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
        
        l_total.backward()
        
        grad_clip_config = self.opt['train'].get('grad_clip')
        if grad_clip_config:
            torch.nn.utils.clip_grad_norm_(parameters=self.net_g.parameters(), **grad_clip_config)
            
        self.optimizer_g.step()

        reduced_loss_dict = self.reduce_loss_dict(loss_dict)
        self.log_dict = reduced_loss_dict
        
        for key, value in reduced_loss_dict.items():
            self.epoch_loss_tracker[key] = self.epoch_loss_tracker.get(key, 0) + value
        self.batch_count_in_epoch += 1

    def test(self):
        """
        Overrides test to use the conditional network.
        """
        self.net_g.eval()
        with torch.no_grad():
            n = self.lq.shape[0]
            outs = []
            m = self.opt['val'].get('max_minibatch', n)
            i = 0

            while i < n:
                j = min(i + m, n)
                current_lq_batch = self.lq[i:j]
                current_cond_batch = self.cond[i:j]
                
                pred = self.net_g(current_lq_batch, current_cond_batch)

                if isinstance(pred, list):
                    pred = pred[-1]

                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
        self.net_g.train()



class ConditionAugmentation(nn.Module):
    """
    一个专门为条件图像设计的数据增强模块。
    它随机组合几何变换和外观变换，以模拟真实世界中参考图与目标图的差异。
    """
    def __init__(self,
                 affine_prob=0.9,
                 blur_prob=0.2,
                 noise_prob=0.2,
                 color_jitter_prob=0.1):
        super().__init__()
        
        # 1. 几何变换 (大概率应用)
        # RandomAffine 已经组合了平移、旋转、缩放、剪切
        self.geometric_transform = T.RandomApply(
            [
                T.RandomAffine(
                    degrees=5,           # 旋转范围: -5 到 +5 度
                    translate=(0.05, 0.05), # 平移范围: 图像尺寸的 -5% 到 +5%
                    scale=(0.98, 1.02),  # 缩放范围: 98% 到 102%
                    shear=3              # 剪切范围: -3 到 +3 度
                )
            ],
            p=affine_prob
        )
        # 水平翻转单独处理
        self.horizontal_flip = T.RandomHorizontalFlip(p=0.5)

        # 2. 外观变换 (小概率应用)
        # 我们将高斯模糊和运动模糊放入一个随机选择器中
        self.blur_transform = T.RandomApply(
            [
                T.RandomChoice([
                    T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
                    # kornia.filters.MotionBlur 需要张量输入，我们将其包装一下
                    MotionBlurTransform(kernel_size=(3, 5), angle=(-15, 15), direction=0.5)
                ])
            ],
            p=blur_prob
        )
        
        # (可选) 增加少量噪声
        self.noise_transform = T.RandomApply([AddGaussianNoise(mean=0., std=0.01)], p=noise_prob)
        
        # (可选) 颜色抖动
        self.color_jitter = T.RandomApply([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05)
        ], p=color_jitter_prob)


    @torch.no_grad()
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        对输入的图像张量应用一系列随机的增强。
        """
        # 确保输入是浮点数张量
        if not isinstance(img_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if img_tensor.dtype != torch.float32:
            img_tensor = img_tensor.float() / 255.0 if img_tensor.max() > 1 else img_tensor.float()

        # 按顺序应用变换
        out = self.geometric_transform(img_tensor)
        out = self.horizontal_flip(out)
        out = self.blur_transform(out)
        out = self.noise_transform(out)
        # out = self.color_jitter(out)
        
        return out

# --- 辅助类，用于包装 kornia 的运动模糊和添加高斯噪声 ---

class MotionBlurTransform(nn.Module):
    def __init__(self, kernel_size, angle, direction):
        super().__init__()
        self.motion_blur = K.RandomMotionBlur(kernel_size=kernel_size, angle=angle, direction=direction, p=1.0)

    def forward(self, img_tensor):
        # kornia 的输入需要是 [B, C, H, W]，所以我们临时增加一个batch维度
        was_squeezed = False
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
            was_squeezed = True
        
        blurred = self.motion_blur(img_tensor)
        
        if was_squeezed:
            blurred = blurred.squeeze(0)
            
        return blurred

class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.01):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean