import torch
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import os, pickle
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T

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
            for x_meta, y_meta in zip(self.x_list, self.y_list):
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
        if self.augment:
            x_data = self.augment_data(x_data)

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
        data_dir='D:\Downloads\medical',
        mode='train',
        length=16,
        augment=False,
        size=[32, 32],
        original=True,
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