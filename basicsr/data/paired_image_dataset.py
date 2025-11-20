# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torchvision
import os
import random
import numpy as np

from PIL import Image
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from torchvision import transforms
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file,
                                    Packing,
                                    Unpacking)
from basicsr.data.degradation_util import Degradation
from basicsr.data.transforms import augment, paired_random_crop, random_augmentation, paired_random_crop_hw, paired_center_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
from pathlib import Path

class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.bgr2rgb = opt.get('bgr2rgb', True)  # 新增选项，默认True

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt[
                'meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder(
                [self.lq_folder, self.gt_folder], ['lq', 'gt'],
                self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt['geometric_augs']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        # print('gt path,', gt_path)
        img_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("gt path {} not working".format(gt_path))

        lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path))


        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # padding
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            if self.geometric_augs:
                #img_gt, img_lq = random_augmentation(img_gt, img_lq)
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=self.bgr2rgb,
                                    float32=True)

        
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


class PairedALLINONEImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedALLINONEImageDataset, self).__init__()
        self.opt = opt
        self.disk_client = None
        self.lmdb_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)
        
        self.tasks = opt['tasks']
        self.degrader = None
        
        is_denoise_task = any(t.startswith('denoise') for t in self.tasks)

        if is_denoise_task:
            degradation_opt = opt.get('degradation_opt', {})
        
            if 'patch_size' not in degradation_opt:
                degradation_opt['patch_size'] = opt['gt_size']
            
            self.degrader = Degradation(degradation_opt)

        self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur': 5, 'lowlight': 6}

        self.gt_paths = []
        self.lq_paths = []
        self.task_ids = []
        
        self._init_paths()


    def _init_paths(self):
        if any(t.startswith('denoise') for t in self.tasks):
            print("Initializing Denoise paths...")
            denoise_clean_paths = []
            with open(self.opt['meta_info_denoise']) as f:
                denoise_filenames = {line.strip() for line in f} # Use a set for faster lookups
            
            all_files_in_dir = os.listdir(self.opt['dataroot_denoise'])
            for filename in all_files_in_dir:
                if filename in denoise_filenames:
                    denoise_clean_paths.append(os.path.join(self.opt['dataroot_denoise'], filename))

            # Create a temporary list for all denoising samples to shuffle them as a group
            denoise_samples = []
            for task_name in self.tasks:
                if task_name.startswith('denoise'):
                    task_id = self.de_dict[task_name]
                    # Original implementation repeats denoising data 3 times
                    for _ in range(3):
                        for path in denoise_clean_paths:
                            # (gt_path, lq_path, task_id)
                            denoise_samples.append((path, path, task_id))
            
            random.shuffle(denoise_samples) # Shuffle within the denoising task block
            
            if denoise_samples:
                gt, lq, tid = zip(*denoise_samples)
                self.gt_paths.extend(gt)
                self.lq_paths.extend(lq)
                self.task_ids.extend(tid)
            print(f"Loaded and shuffled {len(denoise_samples)} samples for denoising tasks.")

        # --- 2. Deraining Paths (LQ/GT pairs exist on disk) ---
        if 'derain' in self.tasks:
            print("Initializing Derain paths...")
            with open(self.opt['meta_info_derain']) as f:
                derain_lq_rel_paths = [line.strip() for line in f]
            
            # Create a temporary list for all deraining samples
            derain_samples = []
            # Original implementation repeats deraining data 120 times
            for _ in range(120):
                for rel_path in derain_lq_rel_paths:
                    lq_path = os.path.join(self.opt['dataroot_derain'], rel_path)
                    gt_path = lq_path.replace('rainy', 'gt').replace('rain-', 'norain-')
                    derain_samples.append((gt_path, lq_path, 3))

            random.shuffle(derain_samples) # Shuffle within the deraining task block
            
            if derain_samples:
                gt, lq, tid = zip(*derain_samples)
                self.gt_paths.extend(gt)
                self.lq_paths.extend(lq)
                self.task_ids.extend(tid)
            print(f"Loaded and shuffled {len(derain_samples)} pairs for deraining.")

        # --- 3. Dehazing Paths (LQ/GT pairs exist on disk) ---
        if 'dehaze' in self.tasks:
            print("Initializing Dehaze paths...")
            with open(self.opt['meta_info_dehaze']) as f:
                dehaze_lq_rel_paths = [line.strip() for line in f]

            # Create a temporary list for all dehazing samples
            dehaze_samples = []
            for rel_path in dehaze_lq_rel_paths:
                lq_path = os.path.join(self.opt['dataroot_dehaze'], rel_path)
                base_dir = lq_path.split("synthetic")[0]
                img_name = Path(lq_path).name.split('_')[0]
                suffix = Path(lq_path).suffix
                gt_path = os.path.join(base_dir, 'original', img_name + suffix)
                dehaze_samples.append((gt_path, lq_path, 4))
            
            random.shuffle(dehaze_samples) # Shuffle within the dehazing task block
            
            if dehaze_samples:
                gt, lq, tid = zip(*dehaze_samples)
                self.gt_paths.extend(gt)
                self.lq_paths.extend(lq)
                self.task_ids.extend(tid)
            print(f"Loaded and shuffled {len(dehaze_samples)} pairs for dehazing.")
        
        # --- 4. Deblurring Paths (Using pre-cropped images) ---
        if 'deblur' in self.tasks:
            print("Initializing Deblur paths from LMDB...")
            lmdb_paths_info = paired_paths_from_lmdb(
                [self.opt['dataroot_deblur_lq'], self.opt['dataroot_deblur_gt']],
                ['lq', 'gt']
            )
            
            deblur_samples = []
            for path_info in lmdb_paths_info:
                lq_key = path_info['lq_path']
                gt_key = path_info['gt_path']
                deblur_samples.append((gt_key, lq_key, self.de_dict['deblur']))
            
            random.shuffle(deblur_samples)

            if deblur_samples:
                gt, lq, tid = zip(*deblur_samples)
                self.gt_paths.extend(gt)
                self.lq_paths.extend(lq)
                self.task_ids.extend(tid)
            print(f"Loaded and shuffled {len(deblur_samples)} key pairs for deblurring from LMDB.")
            
        # --- 5. Low-light Paths (Scanning folders directly) ---
        if 'lowlight' in self.tasks:
            print("Initializing Lowlight paths by scanning folders...")
            
            dataroot_lowlight = self.opt['dataroot_lowlight']
            lq_folder = os.path.join(dataroot_lowlight, 'low')
            gt_folder = os.path.join(dataroot_lowlight, 'high')

            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
            lq_filenames = sorted([
                f for f in os.listdir(lq_folder) 
                if f.lower().endswith(valid_extensions)
            ])
            
            lowlight_samples = []
            
            for _ in range(200):
                for filename in lq_filenames:
                    lq_path = os.path.join(lq_folder, filename)
                    gt_path = os.path.join(gt_folder, filename)
                    
                    if os.path.exists(gt_path):
                        lowlight_samples.append((gt_path, lq_path, self.de_dict['lowlight']))
                    else:
                        print(f"Warning: GT file not found for {lq_path}, skipping.")

            random.shuffle(lowlight_samples)
            
            if lowlight_samples:
                gt, lq, tid = zip(*lowlight_samples)
                self.gt_paths.extend(gt)
                self.lq_paths.extend(lq)
                self.task_ids.extend(tid)
            print(f"Loaded and shuffled {len(lowlight_samples)} pairs for lowlight enhancement.")

    def __getitem__(self, index):
        task_id = self.task_ids[index]
        gt_path_or_key = self.gt_paths[index]
        lq_path_or_key = self.lq_paths[index]

        if task_id == self.de_dict['deblur']:
            if self.lmdb_client is None:
                lmdb_opt = {
                    'type': 'lmdb',
                    'db_paths': [
                        self.opt['dataroot_deblur_lq'], 
                        self.opt['dataroot_deblur_gt']
                    ],
                    'client_keys': ['lq', 'gt']
                }
                self.lmdb_client = FileClient(lmdb_opt.pop('type'), **lmdb_opt)
            
            img_bytes_gt = self.lmdb_client.get(gt_path_or_key, 'gt')
            img_gt = imfrombytes(img_bytes_gt, float32=True)
            
            img_bytes_lq = self.lmdb_client.get(lq_path_or_key, 'lq')
            img_lq = imfrombytes(img_bytes_lq, float32=True)

        else:
            if self.disk_client is None:
                io_backend_opt_copy = self.io_backend_opt.copy()
                self.disk_client = FileClient(io_backend_opt_copy.pop('type'), **io_backend_opt_copy)

            img_bytes_gt = self.disk_client.get(gt_path_or_key, 'gt')
            img_gt = imfrombytes(img_bytes_gt, float32=True)
            
            if task_id < 3:
                img_lq = img_gt.copy()
            else:
                img_bytes_lq = self.disk_client.get(lq_path_or_key, 'lq')
                img_lq = imfrombytes(img_bytes_lq, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            # Paired random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, self.opt['scale'], gt_path_or_key) # 使用 _or_key 变量
            
            # On-the-fly degradation for denoising AFTER cropping
            if task_id < 3:
                clean_patch_uint8 = (img_lq * 255.0).round().astype(np.uint8)
                degrad_patch_uint8 = self.degrader.single_degrade(clean_patch_uint8, task_id)
                img_lq = degrad_patch_uint8.astype(np.float32) / 255.
            
            # Geometric augmentation (flip, rotation)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # Convert to tensor. BGR to RGB, HWC to CHW.
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        
        # Normalize if specified
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path_or_key,
            'gt_path': gt_path_or_key,
            'task_id': task_id
        }

    def __len__(self):
        return len(self.gt_paths)
    
    
def crop_img(image, base=64):
    h = image.shape[0]
    w = image.shape[1]
    crop_h = h % base
    crop_w = w % base
    return image[crop_h // 2:h - crop_h + crop_h // 2, crop_w // 2:w - crop_w + crop_w // 2, :]

class DenoiseTestPairedDataset(data.Dataset):
    def __init__(self, opt):
        super(DenoiseTestPairedDataset, self).__init__()
        self.opt = opt
        
        self.gt_folder = opt['dataroot_gt']
        name_list = os.listdir(self.gt_folder)
        
        self.gt_paths = [os.path.join(self.gt_folder, name) for name in name_list]

        self.sigma = opt['sigma']
        self.toTensor = ToTensor()

    def _add_gaussian_noise(self, clean_patch):
        noise = np.random.randn(*clean_patch.shape)
        noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
        return noisy_patch, clean_patch

    def set_sigma(self, sigma):
        self.sigma = sigma

    def __getitem__(self, index):
        gt_path = self.gt_paths[index]
        
        clean_img = np.array(Image.open(gt_path).convert('RGB'))
        clean_img = crop_img(clean_img, base=16)

        clean_name = os.path.basename(gt_path).split('.')[0]

        noisy_img, _ = self._add_gaussian_noise(clean_img.copy())

        img_gt = self.toTensor(clean_img)
        img_lq = self.toTensor(noisy_img)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': gt_path,
            'gt_path': gt_path
            # 'gt_basename': clean_name 
        }

    def __len__(self):
        return len(self.gt_paths)

class PairedDerainDehazeTestDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedDerainDehazeTestDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.task = opt['task']

        self.lq_folder = opt['dataroot_lq']
        self.gt_folder = opt['dataroot_gt']
        
        self.lq_paths = sorted([os.path.join(self.lq_folder, f) for f in os.listdir(self.lq_folder)])

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path = self.lq_paths[index]
        lq_filename = Path(lq_path).name

        # 根据任务类型确定 GT 路径
        if self.task == 'derain':
            # 去雨任务: 文件名通常是对应的，只是在不同的文件夹
            gt_path = lq_path.replace('input', 'target').replace('rain-', 'norain-')
            
        elif self.task == 'dehaze':
            # 去雾任务 (SOTS): GT 文件名是 LQ 文件名的第一部分
            gt_basename = lq_filename.split('_')[0] + '.png' # SOTS 的 GT 通常是 .png
            gt_path = os.path.join(self.gt_folder, gt_basename)
        else:
            raise ValueError(f"Unknown task type: {self.task}")

        img_bytes_lq = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes_lq, float32=True)

        img_bytes_gt = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes_gt, float32=True)

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.lq_paths)
    
class PairedLowlightTestDataset(data.Dataset):
    """Paired image dataset for low-light enhancement testing.
       Loads full-resolution images without cropping.

    Args:
        opt (dict): Config for test datasets. It contains the following keys:
            dataroot_lq (str): Data root path for lq (e.g., './datasets/LOL/test/low').
            dataroot_gt (str): Data root path for gt (e.g., './datasets/LOL/test/high').
            io_backend (dict): IO backend type and other kwarg.
    """

    def __init__(self, opt):
        super(PairedLowlightTestDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.lq_folder = opt['dataroot_lq']
        self.gt_folder = opt['dataroot_gt']
        
        # 扫描低光照图像文件夹，获取所有需要测试的文件路径
        self.lq_paths = sorted([os.path.join(self.lq_folder, f) for f in os.listdir(self.lq_folder)])

    def __getitem__(self, index):
        if self.file_client is None:
            # 使用副本以保证多进程安全
            io_backend_opt_copy = self.io_backend_opt.copy()
            self.file_client = FileClient(io_backend_opt_copy.pop('type'), **io_backend_opt_copy)

        # 1. 获取低光照图像路径
        lq_path = self.lq_paths[index]
        lq_filename = os.path.basename(lq_path)

        # 2. 根据低光照图像文件名，构建对应的高光照图像路径
        gt_path = os.path.join(self.gt_folder, lq_filename)

        # 3. 加载 LQ 和 GT 图像 (完整尺寸)
        img_bytes_lq = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes_lq, float32=True)

        img_bytes_gt = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes_gt, float32=True)
        
        # 4. 转换为 Tensor (完整尺寸，无裁剪)
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.lq_paths)


class UnpairedImageTestDataset(data.Dataset):
    """BasicSR 风格的无配对图像测试/推理数据集。

    用于对任意单张图片或一个文件夹中的所有图片进行推理。

    Args:
        opt (dict): 数据集配置。包含以下键：
            dataroot_lq (str): 指向单个图像文件或一个包含图像的文件夹。
            io_backend (dict): IO 后端配置。
    """

    def __init__(self, opt):
        super(UnpairedImageTestDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.lq_folder = opt['dataroot_lq']
        if os.path.isfile(self.lq_folder):
            self.lq_paths = [self.lq_folder]
        else:
            extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
            self.lq_paths = sorted([
                os.path.join(self.lq_folder, f) 
                for f in os.listdir(self.lq_folder) 
                if any(f.endswith(ext) for ext in extensions)
            ])
        
        if not self.lq_paths:
            raise ValueError(f"No images found in dataroot_lq: {self.lq_folder}")

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_path = self.lq_paths[index]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # img_lq = padding(img_lq, 16) # 可选

        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)

        return {'lq': img_lq, 'lq_path': lq_path} # 注意：没有 'gt'

    def __len__(self):
        return len(self.lq_paths)
    
    
class PairedDemosaicDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedDemosaicDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', True)

    def _read_raw_file(self, file_bytes, is_gt=False):
        img = np.frombuffer(file_bytes, dtype=np.uint16)
        
        if is_gt:
            dim = int(np.sqrt(img.size / 3))
            img = img.reshape((dim, dim, 3))
        else:
            dim = int(np.sqrt(img.size))
            img = img.reshape((dim, dim, 1))
            
        return img.astype(np.float32) / 65535.0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)
        
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        
        img_gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = self._read_raw_file(img_gt_bytes, is_gt=True)
        except Exception as e:
            raise Exception(f"GT path {gt_path} broken: {e}")

        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = self._read_raw_file(img_lq_bytes, is_gt=False)
        except Exception as e:
            raise Exception(f"LQ path {lq_path} broken: {e}")

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            if self.geometric_augs:
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])


        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)