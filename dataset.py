import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os, pickle, random
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.random import RandomTransform

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

def display_image(image_data, slice_index=None):
    if image_data.ndim == 3:
        assert 0 <= slice_index < image_data.shape[0], "Slice index out of range"
        display_slice(image_data, slice_index)
    else:
        plt.imshow(image_data, cmap='gray')
        plt.show()

# 处理 NIfTI 文件并显示
def get_data(file_path, use_simpleitk=True):
    # 选择读取方式
    if use_simpleitk:
        image_data = read_nifti_with_simpleitk(file_path)
    else:
        image_data = read_nifti_with_nibabel(file_path)
        image_data = np.transpose(image_data, (2, 1, 0))  # 转置为 (depth, height, width) 格式
    return image_data

class MyDataset(Dataset):
    def __init__(
        self,
        data_dir='D:\Downloads\medical',
        mode='train',
        length=16,
        augment=False,
        size=[32,32],
        original=False,
        use_metadata=True,
        debug=False,
        accelerate=False,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.length = length
        self.augment = augment
        self.size = size
        self.original = original
        self.use_metadata = use_metadata
        self.debug = debug
        self.accelerate = accelerate

        assert not (accelerate and use_metadata), 'accelerate 和 use_metadata 不能同时为 True, 选择其中一种加速方式即可'
        self._load_metadata()

    def _load_metadata(self):
        if self.accelerate:
            os.makedirs('data', exist_ok=True)
            metadata_name = 'data/metadata_' + self.mode + '.pkl'
            try:
                with open(metadata_name, 'rb') as f:
                    self.x_list, self.y_list = pickle.load(f)
                if self.debug:
                    self.x_list = self.x_list[:10]
                    self.y_list = self.y_list[:10]
                print(f'从{metadata_name}中加载了{len(self.x_list)}条数据')
                return
            except:
                print(f'{metadata_name}没找到, 开始重新计算')

        if self.use_metadata and not self.accelerate:
            os.makedirs('metadata', exist_ok=True)
            metadata_name = 'metadata/metadata_' + self.mode + '.pkl'
            try:
                with open(metadata_name, 'rb') as f:
                    self.x_list, self.y_list = pickle.load(f)
                if self.debug:
                    self.x_list = self.x_list[:10]
                    self.y_list = self.y_list[:10]
                print(f'从{metadata_name}中加载了{len(self.x_list)}条数据')
                return
            except:
                print(f'{metadata_name}没找到, 开始重新计算')

        self.features_dir = os.path.join(self.data_dir, 'imagesTr')
        self.labels_dir = os.path.join(self.data_dir, 'labelsTr')
        # 读取 self.features_dir 和 self.labels_dir 下的文件名
        self.x_list = [os.path.join(self.features_dir, f) for f in os.listdir(self.features_dir)]
        self.y_list = [os.path.join(self.labels_dir, f) for f in os.listdir(self.labels_dir)]

        if self.mode == 'train':
            # 拿取前 70% 数据
            num_train = int(len(self.x_list) * 0.7)
            self.x_list = self.x_list[:num_train]
            self.y_list = self.y_list[:num_train]
        elif self.mode == 'val':
            # 拿取中 20% 数据
            num_train = int(len(self.x_list) * 0.7)
            num_val = int(len(self.x_list) * 0.2)
            self.x_list = self.x_list[num_train:num_train + num_val]
            self.y_list = self.y_list[num_train:num_train + num_val]
        elif self.mode == 'test':
            # 拿取最后 10% 数据
            num_train = int(len(self.x_list) * 0.7)
            num_val = int(len(self.x_list) * 0.2)
            self.x_list = self.x_list[num_train + num_val:]
            self.y_list = self.y_list[num_train + num_val:]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        assert len(self.x_list) == len(self.y_list), "Feature and label lists must have the same length."

        # 把 3D 图像展开为若干 2D 图像, 并只保存 label 非全零的 pairs
        x_list_t = []
        y_list_t = []
        for i in range(len(self.x_list)):
            x_path = self.x_list[i]
            y_path = self.y_list[i]
            y_data = get_data(y_path, use_simpleitk=True)
            y_data[y_data == 2] = 1
            for j in range(y_data.shape[0]):
                y_label = y_data[j, :, :]       # [H W]
                if y_label.sum() > 0:
                    x_list_t.append(x_path + '@' + str(j))
                    y_list_t.append(y_path + '@' + str(j))
        self.x_list = x_list_t
        self.y_list = y_list_t

        if self.debug:
            self.x_list = self.x_list[:10]
            self.y_list = self.y_list[:10]

        if self.accelerate:
            cache_root = os.path.join('data', self.mode)
            img_cache_dir = os.path.join(cache_root, 'images')
            label_cache_dir = os.path.join(cache_root, 'labels')
            os.makedirs(img_cache_dir, exist_ok=True)
            os.makedirs(label_cache_dir, exist_ok=True)
            new_x_list = []
            new_y_list = []
            for x_meta, y_meta in tqdm(zip(self.x_list, self.y_list), total=len(self.x_list), desc='数据预处理进度'):
                x_path, slice_idx = x_meta.split('@')
                y_path, _  = y_meta.split('@')
                si = int(slice_idx)
                x_data = get_data(x_path, use_simpleitk=True)[si, :, :]   # [H W]
                y_data = get_data(y_path, use_simpleitk=True)[si, :, :]   # [H W]
  
                # 保存为 .npy 文件
                base_x = os.path.splitext(os.path.basename(x_path))[0] + '_image'  # liver_0_image
                base_y = os.path.splitext(os.path.basename(y_path))[0] + '_label'  # liver_0_label
                img_file = os.path.join(img_cache_dir, f"{base_x}_{si}.npy")
                label_file = os.path.join(label_cache_dir, f"{base_y}_{si}.npy")
                np.save(img_file, x_data)
                np.save(label_file, y_data)

                new_x_list.append(img_file)
                new_y_list.append(label_file)
            self.x_list = new_x_list
            self.y_list = new_y_list
            
            with open(metadata_name, 'wb') as f:
                pickle.dump((self.x_list, self.y_list), f)
            print(f'共加载了数据{len(self.x_list)}条到{metadata_name}里')
            return

        if self.use_metadata and not self.accelerate:
            with open(metadata_name, 'wb') as f:
                pickle.dump((self.x_list, self.y_list), f)
            print(f'共加载了数据{len(self.x_list)}条到{metadata_name}里')

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        # 读取数据
        if self.accelerate:
            x_path = self.x_list[idx]  # e.g. data\train\images\liver_0.nii_image_45.npy
            y_path = self.y_list[idx]
            x_data = np.load(x_path)   # [H, W]
            y_data = np.load(y_path)   # [H, W]
        else:
            x_path, x_slice_str = self.x_list[idx].split('@')
            y_path, y_slice_str = self.y_list[idx].split('@')
            x_data = get_data(x_path, use_simpleitk=True)[int(x_slice_str), :, :]   # [H W]
            y_data = get_data(y_path, use_simpleitk=True)[int(y_slice_str), :, :]   # [H W]

        h, w = x_data.shape
        if h != 512 or w != 512:
            raise ValueError(f"Image shape is not (x, 512, 512), but {x_data.shape}")
        h, w = y_data.shape
        if h != 512 or w != 512:
            raise ValueError(f"Label shape is not (x, 512, 512), but {y_data.shape}")
        
        # 把 y_data 里所有值为 2 的像素值改为 0 (肝脏肿瘤同视为肝脏)
        # 此时: 0: 背景, 1: 肝脏
        y_data[y_data == 2] = 1

        if self.original:
            res = {
                'feature': x_data,
                'label': y_data,
                'feature_path': x_path,
                'label_path': y_path,
                'length': self.length,
                'size': list(self.size),
            }
            return res

        if x_data.ndim == 3:
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
        x_data = np.clip(x_data, -300, 300)
        x_data = (x_data + 300) / 600.0
        x_data = x_data * 2 - 1

        # 对 x_data 进行数据增强
        # TODO: 这里目前简单实现了一下, 但是我们需要参考: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py 的 643 ~ 673 行的实现方式
        if self.augment and self.mode == 'train':
        if self.augment and self.mode == 'train':
            x_data, y_data = self.augment_data(x_data, y_data)
        else:
            x_data = torch.tensor(x_data).float()
            y_data = torch.tensor(y_data).float()

        res = {
            'feature': x_data.unsqueeze(0),
            'label': y_data.unsqueeze(0),
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
    
    def augment_data(self, x_data, y_data):
        # x_data, y_data 都是 ndarray, x_data 在 [-1, 1] 之间, y_data 是 0/1, shape 都是 [H W]
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90, interpolation=Image.NEAREST),  # 标签使用最近邻插值
        ])

        advanced_transforms =[]
        # 更可控的空间变换配置（同时作用于图像和分割标签，需保持几何一致性）
        advanced_transforms.append(
            SpatialTransform(
                # 基础参数
                (512, 512),  # 输出图像尺寸（高度，宽度）
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

        advanced_transforms.append( 
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.05),  # 噪声方差范围（0-0.05，低强度噪声）
                    p_per_channel=1  # 100%概率对所有通道添加噪声（单通道医学图像仍适用）
                ),
                apply_probability=0.1  # 整体10%概率启用该变换
            )
        )
        advanced_transforms.append( 
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
        advanced_transforms.append( 
            RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=(0.9, 1.1),  # 亮度乘数范围（±10%调整）
                    synchronize_channels=True,  # 多通道时同步调整（医学图像通常单通道，此参数防御性保留）
                    p_per_channel=1
                ),
                apply_probability=0.1  # 10%概率启用
            )
        )
        advanced_transforms.append( 
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
        advanced_transforms = ComposeTransforms(advanced_transforms)

        # x_data, y_data 都是 ndarray, x_data 在 [-1, 1] 之间, y_data 是 0/1, shape 都是 [H W]
        x = torch.tensor(x_data).float()
        y = torch.tensor(y_data).float()

        # 应用空间变换
        x = x.unsqueeze(0) # (1, H, W)
        y = y.unsqueeze(0) # (1, H, W)
        tmp = advanced_transforms(**{"image": x, "segmentation": y})
        x, y = tmp['image'], tmp['segmentation']
        x = x.squeeze(0) # (H, W)
        y = y.squeeze(0) # (H, W)
        return x, y

if __name__ == '__main__':
    dataset = MyDataset(
        data_dir='D:\Downloads\medical',
        mode='train',
        length=16,
        augment=True,
        size=[32, 32],
        original=False,
        use_metadata=False,
        debug=True,
        accelerate=True,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for step, batch in enumerate(dataloader):
        x_data = batch['feature']
        y_data = batch['label']
        display_image(x_data.squeeze())
        display_image(y_data.squeeze())
        raise