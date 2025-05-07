import torch
import random
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
import cv2
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import shutil
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image
from PIL import Image
from torchvision import transforms


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
def get_data(file_path, use_simpleitk=False):
    # 选择读取方式
    if use_simpleitk:
        image_data = read_nifti_with_simpleitk(file_path)
    else:
        image_data = read_nifti_with_nibabel(file_path)
    return image_data

def process_data(root, x_list, y_list):
    count = 0
    os.makedirs(f"{root}/image", exist_ok=True)
    os.makedirs(f"{root}/label", exist_ok=True)
    for idx in range(len(x_list)):
        x_path = x_list[idx]
        y_path = y_list[idx]
        x_data = get_data(x_path)
        y_data = get_data(y_path)
        h, w, t = x_data.shape
        h1, w1, t1 = y_data.shape
        assert t==t1 or h==h1 or w==w1, f"Image shape {x_data.shape} and label shape {y_data.shape} do not match."
        
        for i in range(t):
            x = x_data[:, :, i]
            y = y_data[:, :, i]
            if np.any(y) > 0:
                img = window_image(x, window_width=400, window_level=50)
                cv2.imwrite(f"{root}/image/{count}.png", img)
                y[y==1] = 255
                y[y==2] = 255
                y = y.astype(np.uint8)
                cv2.imwrite(f"{root}/label/{count}.png", y)
                count += 1
    print(f"{root}数据已保存到 {root}，共 {count} 张图像。")


rootx = "data/train"
rootval = "data/val"
rooty = "data/test"

class MyDataset(Dataset):
    def __init__(
        self,
        data_dir='data',
        mode='train',
        length=16,
        augment=False,
        size=[512,512],
        img_size=512,
        reprocess = False,  ################## 这个变量是用来处理数据变成 png 格式的 第一次需要开启为True，之后就不需要了
        transform=None,
        seed=666,
    ):
        super(MyDataset, self).__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.length = length
        self.augment = augment
        self.size = size
        self.img_size = img_size
        self.seed = seed
        self.transform = transform
        if reprocess:
            if os.path.exists(rootx):
                shutil.rmtree(rootx)
            if os.path.exists(rootval):
                shutil.rmtree(rootval)
            if os.path.exists(rooty):
                shutil.rmtree(rooty)
            self.features_dir = os.path.join(self.data_dir, 'imagesTr')
            self.labels_dir = os.path.join(self.data_dir, 'labelsTr')
            # 读取 self.features_dir 和 self.labels_dir 下的文件名
            x_list = [os.path.join(self.features_dir, f) for f in os.listdir(self.features_dir)]
            y_list = [os.path.join(self.labels_dir, f) for f in os.listdir(self.labels_dir)]

            # 拿取前 80% 数据
            num_train = int(len(x_list) * 0.8)
            x_list_train = x_list[:num_train]
            y_list_train = y_list[:num_train]

            process_data(rootx, x_list_train, y_list_train)

            # 拿取中 10% 数据
            num_val = int(len(x_list) * 0.1)
            x_list_val = x_list[num_train:num_train + num_val]
            y_list_val = y_list[num_train:num_train + num_val]

            process_data(rootval, x_list_val, y_list_val)

            # 拿取最后 10% 数据
            x_list_test = x_list[num_train + num_val:]
            y_list_test = y_list[num_train + num_val:]

            process_data(rooty, x_list_test, y_list_test)

            assert len(x_list) == len(y_list), "Feature and label lists must have the same length."
        
        assert os.listdir(f"{rootx}/image") != [] and os.listdir(f"{rootx}/label") != [], "数据集为空，重新处理数据集"
        
        if self.mode == "train":
            self.root = rootx
            self.x_list = os.listdir(f"{rootx}/image")
            self.y_list = os.listdir(f"{rootx}/label")
        elif self.mode == "val":
            self.root = rootval
            self.x_list = os.listdir(f"{rootval}/image")
            self.y_list = os.listdir(f"{rootval}/label")
        elif self.mode == "test":
            self.root = rooty
            self.x_list = os.listdir(f"{rooty}/image")
            self.y_list = os.listdir(f"{rooty}/label")



    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        # 读取数据
        image_pth = os.path.join(f"{self.root}/image", self.x_list[idx])
        label_pth = os.path.join(f"{self.root}/label", self.y_list[idx])
        x_data = Image.open(image_pth).convert("RGB")
        y_data = Image.open(label_pth).convert("L")
        
        self.iadvanced_transforms =[]
        # 更可控的空间变换配置（同时作用于图像和分割标签，需保持几何一致性）
        self.iadvanced_transforms.append(
            SpatialTransform(
                # 基础参数
                (self.img_size, self.img_size),  # 输出图像尺寸（高度，宽度）
                patch_center_dist_from_border=0,  # 采样中心点距离边缘的偏移（0表示从图像中心采样）
                random_crop=False,  # 禁用随机裁剪（避免意外裁剪关键解剖结构）

                # 弹性形变配置（医学图像建议关闭）
                p_elastic_deform=0,  # 设置为0完全禁用，因为弹性形变可能扭曲器官形状

                # 旋转增强配置
                p_rotation=0.2,  # 20%概率应用旋转增强
                rotation=(
                    -30. / 360 * 2. * np.pi,  # 最小旋转角度（-30度，转换为弧度）
                    30. / 360 * 2. * np.pi    # 最大旋转角度（+30度）
                ),  # 限制旋转范围避免极端角度导致解剖结构不合理

                # 缩放增强配置
                p_scaling=0.2,  # 20%概率应用缩放
                scaling=(0.9, 1.1),  # 缩放范围（0.9-1.1倍），轻微缩放保持形状有效性
                p_synchronize_scaling_across_axes=1,  # 100%概率保持x/y轴同步缩放（避免各向异性形变）

                # 分割标签专用参数
                bg_style_seg_sampling=False,  # 禁用背景样式采样（保持标签二值性）
                mode_seg='nearest'  # 分割标签插值方式（必须用最近邻NEAREST避免边缘模糊）
        ))

        # 强度变换组合（仅应用于图像，不影响分割标签）

        self.iadvanced_transforms.append( 
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.05),  # 噪声方差范围（0-0.05，低强度噪声）
                    p_per_channel=1  # 100%概率对所有通道添加噪声（单通道医学图像仍适用）
                ),
                apply_probability=0.1  # 整体10%概率启用该变换
            )
        )
        self.iadvanced_transforms.append( 
            RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),  # 模糊核sigma范围（轻度模糊）
                    # 以下参数保持默认（单通道医学图像不受影响）
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5
                ),
                apply_probability=0.1  # 10%概率启用
            )
        )
            # 高斯噪声增强
        self.iadvanced_transforms.append( 
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=(0.9, 1.1),  # 亮度乘数范围（±10%调整）
                    synchronize_channels=True,  # 多通道时同步调整（医学图像通常单通道，此参数防御性保留）
                    p_per_channel=1
                ),
                apply_probability=0.1  # 10%概率启用
            )
        )
        self.iadvanced_transforms.append( 
            RandomTransform(
                ContrastTransform(
                    contrast_range=(0.9, 1.1),  # 对比度乘数范围（±10%调整）
                    preserve_range=True,  # 关键参数！保持CT/MRI原始数值范围（如HU单位）
                    synchronize_channels=True,  # 多通道同步调整
                    p_per_channel=1
                ),
                apply_probability=0.1  # 10%概率启用
            )
        )
        self.iadvanced_transforms = ComposeTransforms(self.iadvanced_transforms)
        # 定义图像变换
        if self.transform is None:
            if self.mode == "train":
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(90, interpolation=Image.NEAREST),  # 标签使用最近邻插值
                    transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
                ])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.img_size, self.img_size), interpolation=Image.NEAREST),
                ])
                
        random.seed(self.seed + idx)
        np.random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)
        x_data = self.transform(x_data)

        # 对标签应用相同变换（禁用插值，使用最近邻，防止标签变化）
        random.seed(self.seed + idx)
        np.random.seed(self.seed + idx)
        torch.manual_seed(self.seed + idx)
        y_data = self.transform(y_data)

        label_mapping = {0: 0, 255: 1}  # 映射标签值到连续索引
        # 转为Tensor
        x_data = transforms.ToTensor()(x_data) # (C, H, W)
        # 转换为Numpy数组
        y_data = np.array(y_data)
        # 映射标签值到连续索引
        y_data = np.vectorize(label_mapping.get)(y_data)
        y_data = torch.from_numpy(y_data).long()  # (H, W)
        x_data = x_data[0].unsqueeze(0)  # (1, H, W)

        if self.mode == "train": # 这里的增强需要保证 图像和分割都是  (1, H, W)
            # 应用空间变换
            y_data = y_data.unsqueeze(0) # (1, H, W)
            tmp = self.iadvanced_transforms(**{"image": x_data, "segmentation": y_data})
            x_data, y_data = tmp['image'], tmp['segmentation']
            y_data = y_data.squeeze(0)  # (H, W)

        res = {
            'feature': x_data,
            'label': y_data,
            'feature_path': image_pth,
            'label_path': label_pth,
            'length': self.length,
            'size': list(self.size),
        }

        return res


if __name__ == '__main__':
    dataset = MyDataset(
        data_dir='data',
        mode='train',
    )
    print(f"训练集大小: {len(dataset)}")
    print(f"打印一个数据: {dataset[1]['label']}")
    print("type of label: ", type(dataset[1]['label']))
    print("shape of label: ", dataset[1]['label'].shape)
    base_features_path = 'data/imagesTr'
    base_labels_path = 'data/labelsTr'

    t1 = get_data(os.path.join(base_features_path, 'liver_0.nii.gz'))   # [75 512 512]
    display_image(t1, 0, 0)
    t2 = get_data(os.path.join(base_labels_path, 'liver_0.nii.gz'))     # [75 512 512]
    display_image(t2, 0, 1)
