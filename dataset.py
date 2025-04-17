import cv2
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

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

def display_image(image_data, slice_index):
    assert 0 <= slice_index < image_data.shape[0], "Slice index out of range"
    display_slice(image_data, slice_index)

    slice_image = image_data[slice_index, :, :]
    plt.imshow(slice_image, cmap='gray')
    plt.title(f"Original Slice {slice_index}")

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
        mode='trian',
        length=16,
        augment=False,
    ):
        self.data_dir = data_dir
        self.mode = mode
        self.length = length
        self.augment = augment
        self._load_metadata()

    def _load_metadata(self):
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
        
        # 把 y_data 里所有值为 2 的像素值改为 0 (肝脏肿瘤同视为肝脏)
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
            x_data = x_data[:self.length, :, :]
            y_data = y_data[:self.length, :, :]

        # 对 features 进行归一化
        '''
        TODO: 目前这里只是实现了一个简单的 [0, 1] 归一化
        是否要进行固定值截断, 还是根据分布进行截断, 去调研清楚
        '''
        x_data = x_data / np.max(x_data)  # 归一化到 [0, 1]

        # 对 x_data 进行数据增强
        # TODO: 这里暂未实现, 请参考: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py 的 643 ~ 673 行的实现方式
        if self.augment:
            pass

        res = {
            'feature': x_data,
            'label': y_data,
        }

        return res

if __name__ == '__main__':
    dataset = MyDataset(
        data_dir='D:\Downloads\medical',
        mode='train',
    )

    # base_features_path = 'D:\Downloads\medical\imagesTr'
    # base_labels_path = 'D:\Downloads\medical\labelsTr'

    # t1 = get_data(os.path.join(base_features_path, 'liver_0.nii.gz'), use_simpleitk=True)   # [75 512 512]
    # display_image(t1, 0)
    # t2 = get_data(os.path.join(base_labels_path, 'liver_0.nii.gz'), use_simpleitk=True)     # [75 512 512]
    # display_image(t2, 0)