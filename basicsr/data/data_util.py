# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir


def read_img_seq(path, require_mod_crop=False, scale=1):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]
    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)
    return imgs


def generate_frame_indices(crt_idx,
                           max_frame_num,
                           num_frames,
                           padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle',
                       'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(
            f'{input_key} folder and {gt_key} folder should both in lmdb '
            f'formats. But received {input_key}: {input_folder}; '
            f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(
            f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(
                dict([(f'{input_key}_path', lmdb_key),
                      (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file,
                                     filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, (
        'The len of folders should be 2 with [input_folder, gt_folder]. '
        f'But got {len(folders)}')
    assert len(keys) == 2, (
        'The len of keys should be 2 with [input_key, gt_key]. '
        f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = list(scandir(input_folder))
    gt_paths = list(scandir(gt_folder))
    assert len(input_paths) == len(gt_paths), (
        f'{input_key} and {gt_key} datasets have different number of images: '
        f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []
    for idx in range(len(gt_paths)):
        gt_path = gt_paths[idx]
        basename, ext = osp.splitext(osp.basename(gt_path))
        input_path = input_paths[idx]
        basename_input, ext_input = osp.splitext(osp.basename(input_path))
        input_name = f'{filename_tmpl.format(basename)}{ext_input}'
        input_path = osp.join(input_folder, input_name)
        assert input_name in input_paths, (f'{input_name} is not in '
                                           f'{input_key}_paths.')
        gt_path = osp.join(gt_folder, gt_path)
        paths.append(
            dict([(f'{input_key}_path', input_path),
                  (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3,
                     4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(
        0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x

def Packing(images):
    """
    Packs a 1-channel Bayer pattern image (torch.Tensor or np.ndarray)
    into a 4-channel image where each channel corresponds to R, G1, G2, B pixels.
    Automatically detects (H, W, 1) / (B, H, W, 1) (HWC/NHWC) or
    (1, H, W) / (B, 1, H, W) (CHW/NCHW) input format.

    Args:
        images (torch.Tensor or np.ndarray): Input image(s).
            Expected shape patterns: (H, W, 1), (B, H, W, 1), (1, H, W), (B, 1, H, W).

    Returns:
        torch.Tensor or np.ndarray: Packed image(s) in the same format as input.
            Output shape patterns: (H/2, W/2, 4), (B, H/2, W/2, 4), (4, H/2, W/2), (B, 4, H/2, W/2).

    Raises:
        TypeError: If input is not a torch.Tensor or np.ndarray.
        ValueError: If input dimension is not 3 or 4, or shape is ambiguous/incorrect.
        AssertionError: If input doesn't have 1 channel, or spatial dimensions are odd.
    """
    is_torch = isinstance(images, torch.Tensor)
    if not (is_torch or isinstance(images, np.ndarray)):
        raise TypeError("Input must be a PyTorch tensor or NumPy array.")

    original_device = images.device if is_torch else None
    original_dtype = images.dtype
    input_shape = images.shape
    input_ndim = images.ndim
    was_3d = False
    is_hwc = False # Flag to remember original format (HWC or CHW)

    # 1. Determine input format (HWC/CHW) and convert to 4D NumPy (NCHW or NHWC)
    # Input C is assumed to be 1 for packing

    if input_ndim == 3:
        was_3d = True
        # For C=1, 3D shapes are typically (H, W, 1) or (1, H, W).
        # The channel dimension (1) is small compared to H and W.
        # We can check if the 1 is the first or last dimension.
        if input_shape[0] == 1 and input_shape[-1] != 1: # Likely (1, H, W) CHW
            images_np = images.cpu().numpy() if is_torch else images # (1, H, W)
            images_np = np.expand_dims(images_np, axis=0) # Add batch dim: (1, 1, H, W) -> NCHW
            is_hwc = False
        elif input_shape[-1] == 1 and input_shape[0] != 1: # Likely (H, W, 1) HWC
            images_np = images.cpu().numpy() if is_torch else images # (H, W, 1)
            images_np = np.expand_dims(images_np, axis=0) # Add batch dim: (1, H, W, 1) -> NHWC
            is_hwc = True
        else:
            # Ambiguous shape like (1, 1, H) or (H, 1, 1) or H=W=1
            raise ValueError(f"Ambiguous or incorrect 3D shape for C=1: {input_shape}. Expected (1, H, W) or (H, W, 1) with H, W > 1.")

    elif input_ndim == 4:
        # For C=1, 4D shapes are typically (B, 1, H, W) NCHW or (B, H, W, 1) NHWC.
        # The channel dimension (1) is either index 1 or index 3.
        if input_shape[1] == 1 and input_shape[-1] != 1: # Likely (B, 1, H, W) NCHW
            images_np = images.cpu().numpy() if is_torch else images # (B, 1, H, W)
            is_hwc = False
        elif input_shape[-1] == 1 and input_shape[1] != 1: # Likely (B, H, W, 1) NHWC
            images_np = images.cpu().numpy() if is_torch else images # (B, H, W, 1)
            is_hwc = True
        else:
            # Ambiguous shape like (B, 1, 1, H) or (B, H, 1, 1) or B=H=W=1
            raise ValueError(f"Ambiguous or incorrect 4D shape for C=1: {input_shape}. Expected (B, 1, H, W) or (B, H, W, 1) with H, W > 1.")

    else:
        raise ValueError(f"Input tensor/array must be 3D or 4D, but got {input_ndim}D with shape {input_shape}")

    # At this point, images_np is a 4D NumPy array, either NCHW or NHWC, float32 (if torch input) or original dtype (if numpy input)

    # 2. Convert internal representation to NCHW (B, C, H, W) if it was NHWC
    if is_hwc:
         # Transpose from NHWC (B, H, W, C) to NCHW (B, C, H, W)
         images_np = images_np.transpose(0, 3, 1, 2) # (B, 1, H, W)

    # Now images_np is guaranteed to be (B, 1, H, W)

    # 3. Perform core packing logic using NCHW format
    b, c, H, W = images_np.shape

    assert c == 1, f"Input image must have 1 channel (Bayer pattern) after format detection, but got {c} channels. This indicates an issue with format detection or input shape."
    assert H % 2 == 0 and W % 2 == 0, f"Height ({H}) and width ({W}) of the image must be even for packing."

    # Slicing is done on NCHW (B, C, H, W)
    R = images_np[:, 0, 0:H:2, 0:W:2]   # R pixels (B, H/2, W/2)
    G1 = images_np[:, 0, 0:H:2, 1:W:2]  # G pixels (top row, right column) (B, H/2, W/2)
    G2 = images_np[:, 0, 1:H:2, 0:W:2]  # G pixels (bottom row, left column) (B, H/2, W/2)
    B = images_np[:, 0, 1:H:2, 1:W:2]   # B pixels (B, H/2, W/2)

    # Concatenate along channel dimension (axis 1) to get (B, 4, H/2, W/2) NCHW
    packed_images_np_nchw = np.concatenate((
        np.expand_dims(R, axis=1),  # (B, 1, H/2, W/2)
        np.expand_dims(G1, axis=1), # (B, 1, H/2, W/2)
        np.expand_dims(G2, axis=1), # (B, 1, H/2, W/2)
        np.expand_dims(B, axis=1)   # (B, 1, H/2, W/2)
    ), axis=1) # Result is (B, 4, H/2, W/2)

    # 4. Convert packed NumPy array back to original format and dimension (3D/4D)
    final_images_np = packed_images_np_nchw # Start with NCHW result

    if is_hwc:
         # If original was HWC/NHWC, convert NCHW (B, 4, H/2, W/2) back to NHWC (B, H/2, W/2, 4)
         final_images_np = final_images_np.transpose(0, 2, 3, 1)

    if was_3d:
        # If original was 3D, remove the batch dimension (axis 0)
        final_images_np = np.squeeze(final_images_np, axis=0)
        # Now shape is (4, H/2, W/2) if original was CHW, or (H/2, W/2, 4) if original was HWC

    # 5. Convert back to original type (torch.Tensor or np.ndarray)
    if is_torch:
        # Convert back to torch tensor, move to original device, and set original dtype
        packed_images = torch.from_numpy(final_images_np).to(original_device).type(original_dtype)
    else:
        # Return NumPy array directly
        packed_images = final_images_np

    # 6. Return result
    return packed_images


def Unpacking(packed_images):
    """
    Unpacks a 4-channel packed image (torch.Tensor or np.ndarray)
    back into a 1-channel Bayer pattern image.
    Automatically detects (H/2, W/2, 4) / (B, H/2, W/2, 4) (HWC/NHWC) or
    (4, H/2, W/2) / (B, 4, H/2, W/2) (CHW/NCHW) input format.

    Args:
        packed_images (torch.Tensor or np.ndarray): Packed image(s).
            Expected shape patterns: (H/2, W/2, 4), (B, H/2, W/2, 4), (4, H/2, W/2), (B, 4, H/2, W/2).

    Returns:
        torch.Tensor or np.ndarray: Unpacked image(s) in the same format as input.
            Output shape patterns: (H, W, 1), (B, H, W, 1), (1, H, W), (B, 1, H, W).

    Raises:
        TypeError: If input is not a torch.Tensor or np.ndarray.
        ValueError: If input dimension is not 3 or 4, or shape is ambiguous/incorrect.
        AssertionError: If input doesn't have 4 channels.
    """
    is_torch = isinstance(packed_images, torch.Tensor)
    if not (is_torch or isinstance(packed_images, np.ndarray)):
         raise TypeError("Input must be a PyTorch tensor or NumPy array.")

    original_device = packed_images.device if is_torch else None
    original_dtype = packed_images.dtype
    input_shape = packed_images.shape
    input_ndim = packed_images.ndim
    was_3d = False
    is_hwc = False # Flag to remember original format (HWC or CHW)

    # 1. Determine input format (HWC/CHW) and convert to 4D NumPy (NCHW or NHWC)
    # Input C is assumed to be 4 for unpacking

    if input_ndim == 3:
        was_3d = True
        # For C=4, 3D shapes are typically (H/2, W/2, 4) or (4, H/2, W/2).
        # The channel dimension (4) is small compared to H/2 and W/2.
        # We can check if the 4 is the first or last dimension.
        if input_shape[0] == 4 and input_shape[-1] != 4: # Likely (4, H/2, W/2) CHW
            packed_images_np = packed_images.cpu().numpy() if is_torch else packed_images # (4, H/2, W/2)
            packed_images_np = np.expand_dims(packed_images_np, axis=0) # Add batch dim: (1, 4, H/2, W/2) -> NCHW
            is_hwc = False
        elif input_shape[-1] == 4 and input_shape[0] != 4: # Likely (H/2, W/2, 4) HWC
            packed_images_np = packed_images.cpu().numpy() if is_torch else packed_images # (H/2, W/2, 4)
            packed_images_np = np.expand_dims(packed_images_np, axis=0) # Add batch dim: (1, H/2, W/2, 4) -> NHWC
            is_hwc = True
        else:
            # Ambiguous shape like (4, 4, H/2) or (H/2, 4, 4) or H/2=W/2=4
            raise ValueError(f"Ambiguous or incorrect 3D shape for C=4: {input_shape}. Expected (4, H/2, W/2) or (H/2, W/2, 4) with H/2, W/2 > 4.")

    elif input_ndim == 4:
        # For C=4, 4D shapes are typically (B, 4, H/2, W/2) NCHW or (B, H/2, W/2, 4) NHWC.
        # The channel dimension (4) is either index 1 or index 3.
        if input_shape[1] == 4 and input_shape[-1] != 4: # Likely (B, 4, H/2, W/2) NCHW
            packed_images_np = packed_images.cpu().numpy() if is_torch else packed_images # (B, 4, H/2, W/2)
            is_hwc = False
        elif input_shape[-1] == 4 and input_shape[1] != 4: # Likely (B, H/2, W/2, 4) NHWC
            packed_images_np = packed_images.cpu().numpy() if is_torch else packed_images # (B, H/2, W/2, 4)
            is_hwc = True
        else:
             # Ambiguous shape like (B, 4, 4, H/2) or (B, H/2, 4, 4) or B=H/2=W/2=4
             raise ValueError(f"Ambiguous or incorrect 4D shape for C=4: {input_shape}. Expected (B, 4, H/2, W/2) or (B, H/2, W/2, 4) with H/2, W/2 > 4.")

    else:
        raise ValueError(f"Input tensor/array must be 3D or 4D, but got {input_ndim}D with shape {input_shape}")

    # At this point, packed_images_np is a 4D NumPy array, either NCHW or NHWC

    # 2. Convert internal representation to NCHW (B, C, H, W) if it was NHWC
    if is_hwc:
         # Transpose from NHWC (B, H/2, W/2, 4) to NCHW (B, 4, H/2, W/2)
         packed_images_np = packed_images_np.transpose(0, 3, 1, 2) # (B, 4, H/2, W/2)

    # Now packed_images_np is guaranteed to be (B, 4, h, w)

    # 3. Perform core unpacking logic using NCHW format
    b, c, h, w = packed_images_np.shape
    assert c == 4, f"The number of channels in packed images must be 4 after format detection, but got {c} channels. This indicates an issue with format detection or input shape."

    R = packed_images_np[:, 0, :, :]  # (B, h, w)
    G1 = packed_images_np[:, 1, :, :] # (B, h, w)
    G2 = packed_images_np[:, 2, :, :] # (B, h, w)
    B = packed_images_np[:, 3, :, :]  # (B, h, w)

    H_out, W_out = h * 2, w * 2
    # Create output array in NCHW format (B, 1, H_out, W_out)
    unpacked_images_np_nchw = np.zeros((b, 1, H_out, W_out), dtype=original_dtype) # Use original dtype

    # Assign pixels to the NCHW output array
    unpacked_images_np_nchw[:, 0, 0:H_out:2, 0:W_out:2] = R
    unpacked_images_np_nchw[:, 0, 0:H_out:2, 1:W_out:2] = G1
    unpacked_images_np_nchw[:, 0, 1:H_out:2, 0:W_out:2] = G2
    unpacked_images_np_nchw[:, 0, 1:H_out:2, 1:W_out:2] = B

    # 4. Convert unpacked NumPy array back to original format and dimension (3D/4D)
    final_images_np = unpacked_images_np_nchw # Start with NCHW result (B, 1, H_out, W_out)

    if is_hwc:
         # If original was HWC/NHWC, convert NCHW (B, 1, H_out, W_out) back to NHWC (B, H_out, W_out, 1)
         final_images_np = final_images_np.transpose(0, 2, 3, 1)

    if was_3d:
        # If original was 3D, remove the batch dimension (axis 0)
        final_images_np = np.squeeze(final_images_np, axis=0)
        # Now shape is (1, H_out, W_out) if original was CHW, or (H_out, W_out, 1) if original was HWC

    # 5. Convert back to original type (torch.Tensor or np.ndarray)
    if is_torch:
        unpacked_images = torch.from_numpy(final_images_np).to(original_device).type(original_dtype)
    else:
        unpacked_images = final_images_np

    # 6. Return result
    return unpacked_images