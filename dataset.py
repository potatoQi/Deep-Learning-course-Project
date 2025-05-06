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
        img_size=32,
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

        # 对图像应用变换
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
