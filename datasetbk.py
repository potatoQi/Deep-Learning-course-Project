import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
from utils import get_patch_size
import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from PIL import Image



from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform


# 调窗操作
def window_image(image, window_width=400, window_level=50):
    """
    调窗函数，适应指定的窗宽和窗位
    """
    min_intensity = window_level - window_width // 2
    max_intensity = window_level + window_width // 2

    # 限制像素值在 min_intensity 和 max_intensity 之间
    image = np.clip(image, min_intensity, max_intensity)

    # 将像素值归一化到 [0, 255]
    image = ((image - min_intensity) / (max_intensity - min_intensity) * 255).astype(np.uint8)

    return image


# 读取 NIfTI 文件 (.nii.gz) 使用 SimpleITK
def read_nifti_with_simpleitk(file_path):
    # 使用 SimpleITK 读取 NIfTI 文件
    image = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image)  # 将 SimpleITK 图像转换为 NumPy 数组
    return image_array

# 使用 nibabel 读取 NIfTI 文件 (.nii.gz)
def read_nifti_with_nibabel(file_path):
    # 使用 nibabel 读取 NIfTI 文件
    img = nib.load(file_path)
    image_array = img.get_fdata()  # 获取图像数据
    return image_array

# 显示医学图像数据的某个切片
def display_slice(image_data, slice_index):
    plt.imshow(image_data[slice_index, :, :], cmap='gray')
    plt.title(f"Slice {slice_index}")
    plt.show()

def display_image(image_data, slice_index, xx):
    assert 0 <= slice_index < image_data.shape[0], "Slice index out of range"
    slice_index = len(image_data) - 2
    display_slice(image_data, slice_index)

    slice_image = image_data[slice_index, :, :]
    plt.imshow(slice_image, cmap='gray')
    plt.title(f"Original Slice {slice_index}")
    plt.savefig(f"slice_{slice_index}_{xx}.png")

# 处理 NIfTI 文件并显示
def get_data(file_path, use_simpleitk=True):
    # 选择读取方式
    if use_simpleitk:
        image_data = read_nifti_with_simpleitk(file_path)
    else:
        image_data = read_nifti_with_nibabel(file_path)
        image_data = np.transpose(image_data, (2, 1, 0))  # 转置为 (depth, height, width) 格式
    # print("max: ", np.max(image_data), "min: ", np.min(image_data))
    return image_data

rootx = "data/train"
rootval = "data/val"
rooty = "data/test"

def save_imagefolder(dataset, target_path, maps=None):
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path, exist_ok=True)
    
    for idx in range(len(dataset)):
        img, label = dataset[idx]  
        if hasattr(dataset, 'classes'):
            class_name = dataset.classes[label]
        else:
            if maps is None:
                raise ValueError("Please provide a mapping of label to class name")
            class_name = maps[label]
        class_dir = os.path.join(target_path, class_name)
        os.makedirs(class_dir, exist_ok=True)  
        
        img = to_pil_image(img) if not isinstance(img, Image.Image) else img
        
        target_file = os.path.join(class_dir, f"{idx}.png")
        img.save(target_file)


class MyDataset(Dataset):
    def __init__(
        self,
        data_dir='data',
        mode='train',
        length=16,
        augment=False,
        size=[32,32],
        reprocess = False
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.length = length
        self.augment = augment
        self.size = size
        if reprocess:
            self._load_metadata()

    def _load_metadata(self):
        self.features_dir = os.path.join(self.data_dir, 'imagesTr')
        self.labels_dir = os.path.join(self.data_dir, 'labelsTr')
        # 读取 self.features_dir 和 self.labels_dir 下的文件名
        self.x_list = [os.path.join(self.features_dir, f) for f in os.listdir(self.features_dir)]
        self.y_list = [os.path.join(self.labels_dir, f) for f in os.listdir(self.labels_dir)]

        # 拿取前 70% 数据
        num_train = int(len(self.x_list) * 0.7)
        self.x_list_train = self.x_list[:num_train]
        self.y_list_train = self.y_list[:num_train]
        count = 0
        os.makedirs(f"{rootx}/image", exist_ok=True)
        for idx in range(len(self.x_list_train)):
            x_path = self.x_list_train[idx]
            y_path = self.y_list_train[idx]
            x_data = get_data(x_path, use_simpleitk=True)
            y_data = get_data(y_path, use_simpleitk=True)
            t, h, w = x_data.shape
            t1, h1, w1 = y_data.shape
            assert t!=t1 or h!=h1 or w!=w1, f"Image shape {x_data.shape} and label shape {y_data.shape} do not match."
            
            for i in range(t):
                x = x_data[i, :, :]
                y = y_data[i, :, :]
                if np.any(y) > 0:
                    img = to_pil_image(img) if not isinstance(img, Image.Image) else img
                    img = window_image(x, window_width=400, window_level=50)
                    target_file = os.path.join(f"{rootx}/image", f"{count}.png")



        # 拿取中 20% 数据
        num_val = int(len(self.x_list) * 0.2)
        self.x_list_val = self.x_list[num_train:num_train + num_val]
        self.y_list_val = self.y_list[num_train:num_train + num_val]

        # 拿取最后 10% 数据
        self.x_list_test = self.x_list[num_train + num_val:]
        self.y_list_test = self.y_list[num_train + num_val:]

        assert len(self.x_list) == len(self.y_list), "Feature and label lists must have the same length."

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        # 读取数据
        x_path = self.x_list[idx]
        y_path = self.y_list[idx]
        x_data = get_data(x_path, use_simpleitk=True)
        y_data = get_data(y_path, use_simpleitk=True)

        t, h, w = x_data.shape
        if h != 512 or w != 512:
            raise ValueError(f"Image shape is not (x, 512, 512), but {x_data.shape}")
        t, h, w = y_data.shape
        if h != 512 or w != 512:
            raise ValueError(f"Label shape is not (x, 512, 512), but {y_data.shape}")
        
        # 把 y_data 里所有值为 2 的像素值改为 1 (肝脏肿瘤同视为肝脏)
        # 此时: 0: 背景, 1: 肝脏
        y_data[y_data == 2] = 1

        # 长度裁剪到 self.length
        assert self.length <= x_data.shape[0], f"Length {self.length} is larger than image depth {x_data.shape[0]}."
        if self.mode == 'train':
            # random 裁剪
            start_idx = np.random.randint(0, x_data.shape[0] - self.length + 1)
            end_idx = start_idx + self.length
            x_data = x_data[start_idx:end_idx, :, :]
            y_data = y_data[start_idx:end_idx, :, :]
        else:
            # 验证集和测试集其实不会用到 __getitem__ 的逻辑
            x_data = x_data[:self.length, :, :]
            y_data = y_data[:self.length, :, :]

        # 尺寸裁剪到 self.size
        x_data, y_data = self.resize(x_data, y_data, self.size)

        # 对 features 进行归一化
        # TODO: 目前这里只是实现了一个简单的 [-1, 1] 归一化配合固定值截断, 是否还有更好的方式
        x_data = np.clip(x_data, -1024, 3100)
        x_data = (x_data + 1024) / 4124.0
        x_data = x_data * 2 - 1
        
        patch_size = x_data.shape
        print("patch_size: ", patch_size)

        # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
        # order of the axes is determined by spacing, not image size
        do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > 3
        if do_dummy_2d_data_aug:
            # why do we rotate 180 deg here all the time? We should also restrict it
            rotation_for_DA = (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi)
        else:
            rotation_for_DA = (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)
        mirror_axes = (-3, -2, -1) #(0, 1, 2)
        initial_patch_size = get_patch_size(patch_size[-3:],
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            rotation_for_DA,
                                            (0.85, 1.25))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]
        
        x_data = torch.tensor(x_data).float()
        y_data = torch.tensor(y_data).float()
        
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size[1:]
            ignore_axes = None

        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False, mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True,
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        transforms = ComposeTransforms(transforms)
        if(self.mode == 'train'):
            transforms = transforms
        else:
            transforms = ComposeTransforms([
                RemoveLabelTansform(-1, 0)
            ])
    
        transformed = transforms(**{'image': x_data, 'segmentation': y_data})
        x_data, y_data = transformed['image'], transformed['segmentation']

        # 确认标签数值未变化（仅位置可能变）
        assert set(np.unique(y_data)) <= {0, 1}, f"Label values changed: {np.unique(y_data)}"


        res = {
            'feature': x_data,
            'label': y_data,
            'feature_path': x_path,
            'label_path': y_path,
            'length': self.length,
            'size': list(self.size),
        }

        return res
    
    def resize(self, x_data, y_data, target_size):
        # 使用简单的裁剪或缩放（如果数据尺寸大于目标尺寸）
        t, h, w = x_data.shape
        target_h, target_w = target_size[0], target_size[1]

        if h > target_h or w > target_w:
            # 随机裁剪
            top = np.random.randint(0, h - target_h + 1)
            left = np.random.randint(0, w - target_w + 1)
            x_data = x_data[:, top:top + target_h, left:left + target_w]
            y_data = y_data[:, top:top + target_h, left:left + target_w]
        elif h < target_h or w < target_w:
            # 填充到目标尺寸
            pad_h = target_h - h
            pad_w = target_w - w
            x_data = np.pad(x_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            y_data = np.pad(y_data, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        return x_data, y_data
    
    def augment_data(self, x_data):
        # 定义数据增强的变换
        transform = T.Compose([
            T.RandomHorizontalFlip(p=0.5),  # 50% 概率进行水平翻转
            T.RandomVerticalFlip(p=0.5),    # 50% 概率进行垂直翻转
            T.RandomRotation(30),           # 随机旋转，最大旋转角度为 30°
            T.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 随机平移
        ])

        x_data = torch.tensor(x_data).float()
        x_data = transform(x_data)
        return x_data

if __name__ == '__main__':
    dataset = MyDataset(
        data_dir='data',
        mode='train',
    )

    base_features_path = 'data/imagesTr'
    base_labels_path = 'data/labelsTr'

    t1 = get_data(os.path.join(base_features_path, 'liver_0.nii.gz'), use_simpleitk=True)   # [75 512 512]
    display_image(t1, 0, 0)
    t2 = get_data(os.path.join(base_labels_path, 'liver_0.nii.gz'), use_simpleitk=True)     # [75 512 512]
    display_image(t2, 0, 1)
