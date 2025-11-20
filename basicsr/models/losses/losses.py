# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss
from transformers import CLIPModel, CLIPImageProcessor
from torchvision.transforms.functional import resize, normalize, center_crop 

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class FreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * (loss * 0.01 + self.l1_loss(pred, target))
    
class FFTL1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTL1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}. "
                             f"Supported modes are: 'none', 'mean', 'sum'")
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)

        diff = torch.abs(pred_fft - target_fft)
        
        # Apply the reduction
        if self.reduction == 'mean':
            loss = diff.mean()
        elif self.reduction == 'sum':
            loss = diff.sum()
        else:
            loss = diff
            
        return self.loss_weight * loss
    
    
class FreqNormLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqNormLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred, norm='ortho') - torch.fft.rfft2(target, norm='ortho')
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * (loss * 0.01 + self.l1_loss(pred, target))
    
class FFTLoss(nn.Module):
    """L1 loss in frequency domain with FFT.

    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W). Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise
                weights. Default: None.
        """

        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        
        return self.loss_weight * l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)
    
class StableFFTLoss(nn.Module):
    """
    A stable L1 loss in the frequency domain, combined with a spatial L1 loss.
    """
    def __init__(self, loss_weight=1.0, alpha=0.01, reduction='mean', norm='ortho', ignore_dc=True):
        super(StableFFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha  # Weight for the frequency component
        self.reduction = reduction
        self.norm = norm
        self.ignore_dc = ignore_dc

    def forward(self, pred, target):
        # Spatial L1 Loss
        spatial_loss = F.l1_loss(pred, target, reduction=self.reduction)

        # FFT of prediction and target
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm=self.norm)
        target_fft = torch.fft.fft2(target, dim=(-2, -1), norm=self.norm)

        # Separate real and imaginary parts
        pred_fft_real = pred_fft.real
        pred_fft_imag = pred_fft.imag
        target_fft_real = target_fft.real
        target_fft_imag = target_fft.imag

        # --- Stability Improvements ---
        if self.ignore_dc:
            pred_fft_real[..., 0, 0] = target_fft_real[..., 0, 0]
            pred_fft_imag[..., 0, 0] = target_fft_imag[..., 0, 0]

        # L1 loss for real and imaginary parts
        freq_loss_real = F.l1_loss(pred_fft_real, target_fft_real, reduction=self.reduction)
        freq_loss_imag = F.l1_loss(pred_fft_imag, target_fft_imag, reduction=self.reduction)
        
        freq_loss = freq_loss_real + freq_loss_imag

        # Final combined loss
        combined_loss = spatial_loss + self.alpha * freq_loss
        
        return self.loss_weight * combined_loss
    
class LogGaussianNLLLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-6):
        super(LogGaussianNLLLoss, self).__init__()
        self.loss_weight = loss_weight
        
        self.base_loss = nn.GaussianNLLLoss(full=False, eps=eps, reduction=reduction)

    def forward(self, pred_mean, pred_log_var, target):
        variance = torch.exp(pred_log_var)
        
        loss = self.base_loss(input=pred_mean, target=target, var=variance)
        
        return self.loss_weight * loss
    
class GaussianNLLLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-6):
        super(GaussianNLLLoss, self).__init__()
        self.loss_weight = loss_weight
        
        self.base_loss = nn.GaussianNLLLoss(full=False, eps=eps, reduction=reduction)

    def forward(self, pred, target):
        
        loss = self.base_loss(input=pred, target=target, var=variance)
        
        return self.loss_weight * loss
    
    
    
class CLIPLoss(nn.Module):
    """
    CLIP-based Perceptual Loss.

    This loss module computes a distance between the CLIP embeddings of
    a predicted image and a target image. It is designed to be a standalone
    component that can be easily integrated into any training pipeline.

    The module handles:
    1. Loading a pretrained CLIP model and its image processor.
    2. Preprocessing images to match CLIP's input requirements.
    3. Extracting image features using the CLIP model.
    4. Calculating the loss between the features of the prediction and target.

    Args:
        loss_weight (float): Weight for this loss component. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'. Default: 'mean'.
        loss_type (str): The type of distance metric to use between features.
                         'mse' | 'l1' | 'cosine'. 'mse' is recommended for
                         stable training. Default: 'mse'.
        clip_model_name (str): The name of the pretrained CLIP model to use from
                               Hugging Face Hub.
                               Default: 'openai/clip-vit-large-patch14'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', loss_type='mse',
                 clip_model_name='openai/clip-vit-large-patch14'):
        super(CLIPLoss, self).__init__()

        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}. "
                             f"Choose from 'none', 'mean', 'sum'.")
        if loss_type not in ['mse', 'l1', 'cosine']:
            raise ValueError(f"Unsupported loss_type: {loss_type}. "
                             f"Choose from 'mse', 'l1', 'cosine'.")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type

        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)
        
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Extract preprocessing parameters from the processor config
        self.clip_resize_size = self.processor.size["shortest_edge"]
        self.clip_crop_size = self.processor.crop_size["height"]
        self.clip_normalize_mean = self.processor.image_mean
        self.clip_normalize_std = self.processor.image_std

    def _preprocess_image(self, img):
        b, c, h, w = img.shape
        if h < w:
            new_h = self.clip_resize_size
            new_w = int(w * (self.clip_resize_size / h))
        else:
            new_h = int(h * (self.clip_resize_size / w))
            new_w = self.clip_resize_size
        
        img_resized = resize(
            img, 
            size=[new_h, new_w], 
            interpolation=InterpolationMode.BICUBIC, 
            antialias=True
        )
        
        img_cropped = center_crop(img_resized, output_size=[self.clip_crop_size, self.clip_crop_size])
        img_normalized = normalize(img_cropped, mean=self.clip_normalize_mean, std=self.clip_normalize_std)
        
        return img_normalized

    @torch.no_grad()
    def _get_clip_features(self, img):
        device = img.device
        if self.clip_model.device != device:
            self.clip_model.to(device)

        pixel_values = self._preprocess_image(img)

        projected_features = self.clip_model.get_image_features(pixel_values=pixel_values)
        return projected_features

    def forward(self, pred_img, target_img, **kwargs):
        pred_features = self._get_clip_features(pred_img)
        target_features = self._get_clip_features(target_img)

        if self.loss_type == 'mse':
            loss = F.mse_loss(pred_features, target_features, reduction=self.reduction)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(pred_features, target_features, reduction=self.reduction)
        elif self.loss_type == 'cosine':
            sim = F.cosine_similarity(pred_features, target_features, dim=1)
            loss_unreduced = 1.0 - sim
            
            if self.reduction == 'mean':
                loss = loss_unreduced.mean()
            elif self.reduction == 'sum':
                loss = loss_unreduced.sum()
            else:
                loss = loss_unreduced
        
        return self.loss_weight * loss
    
    
class CosineSimilarityLoss(nn.Module):
    """Cosine Similarity loss.

    This loss is designed to maximize the cosine similarity between two vectors.
    It is equivalent to 1 - cos(x, y). A smaller loss value means higher similarity.
    It's commonly used for feature-level supervision, like in CLIP space.

    Args:
        loss_weight (float): Loss weight for Cosine Similarity loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CosineSimilarityLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: none | mean | sum')

        self.loss_weight = loss_weight
        self.reduction = reduction
        # PyTorch's implementation of CosineSimilarity is an operator, not a loss.
        # So we implement the loss logic directly in the forward pass.
        self.similarity_func = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, D). Predicted feature tensor.
            target (Tensor): of shape (N, D). Ground truth feature tensor.
        """
        # Calculate cosine similarity along the feature dimension (dim=1)
        # The result is a tensor of shape (N,) with values in [-1, 1]
        cosine_sim = self.similarity_func(pred, target)

        # The loss is 1 minus the similarity.
        # This maps similarity from [-1, 1] to a loss in [0, 2].
        # Maximizing similarity is equivalent to minimizing this loss.
        loss = 1.0 - cosine_sim

        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return self.loss_weight * loss