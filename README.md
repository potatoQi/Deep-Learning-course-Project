# 🏥 重庆大学深度学习课程期末项目：基于深度卷积网络的肝脏分割

## 🚀 项目简介

本项目利用 PyTorch Lightning 构建，采用深度卷积网络（UNet）实现肝脏医学图像分割。通过 YAML 配置、模块化代码，以及 WandB/TensorBoard 日志记录，提供灵活、可复现、易扩展的训练管线。

## ✨ 主要功能

1. 🔍 **可视化预测**

   * 在验证集上实时展示模型预测结果与真实分割标签对比
2. ⚡ **双重数据加速**

   * 原始路径缓存：首次解析 `.nii.gz` 生成切片路径，后续无需重复解析
   * `.npy` 切片缓存：预先将非空 2D 切片转换并保存为 `.npy`，显著加速 I/O
3. 🧬 **3D UNet 支持（开发中）**
4. 📊 **多种 Logger**

   * WandB、TensorBoard、或关闭日志
5. 🛠️ **清爽配置管理**

   * 所有功能开关、超参数、路径统一在 `config.yaml` 中管理
6. 🔄 **断点续训**

   * 自动比较配置，加载最新 checkpoint 或指定 checkpoint 或清空重训
7. 🖼️ 数据增强

   * 本项目在 DataModule 中集成了可控的数据增强策略，包含基础翻转、旋转和高级强度及空间变换

## 📦 安装与依赖

```bash
git clone https://github.com/potatoQi/Deep-Learning-course-Project LiverSegmentation
cd LiverSegmentation

conda create -n LiverSeg python=3.10
conda activate LiverSeg

然后去 pytorch 官网安装适合自己电脑版本的 torch

pip install -r requirements.txt
```

## ⚙️ 配置说明

在 `config.yaml` 中统一管理所有参数：

```yaml
Trainer:
  seed: 123
  exp_dir: './results/0'            # 训练结果保存路径 (ckpt, log (wandb 会保存在根目录), etc.)
  logger: tensorboard                     # wandb / tensorboard / null
  use_ckpt: true
  ckpt_save_num: 1                  # 保留最新的 x 个 ckpt
  ckpt_save_interval: 5             # x 个 epoch 保存一次 ckpt
  max_epochs: 10
  log_every_n_steps: 1              # 每 1 步 step 打印一次训练指标
  check_val_every_n_epoch: 1        # 每 x 个 epoch 验证一次

Dataset:
  target: dataloader.DataModuleFromConfig
  batch_size: 1
  num_workers: 0
  train:
    _target_: dataset.MyDataset
    data_dir: 'D:\Downloads\medical'
    mode: train
    length: 1             # 2D 卷积就把这里设为 1 (目前 3D 卷积还在 dev 阶段)
    augment: false         # 是否开启数据增强
    size: [32, 32]        # 这个参数只支持 3D 卷积, 2D 卷积会无视这个参数
    use_metadata: false    # 是否将数据集路径缓存到本地加速读取
    accelerate: true     # 是否将数据集转换为 npy 到本地加速读取
    debug: false          # 是否使用 debug 模式, debug 模式下只会读取 10 个数据

UNet:
  _target_: UNet.UNet
  im_channels: ${Dataset.train.length}
  down_channels: [32, 64, 128, 256]
  down_sample: [true, true, true, false]
  num_heads: 4
  num_down_layers: 1
  num_mid_layers: 1
  num_up_layers: 1
  lr: 1e-4
```

> **根据需求** 修改 `data_dir`、`logger`、`accelerate`、`down_channels` 等字段。

## 🏃‍♀️ 快速开始

1. 编辑并保存 `config.yaml`
2. 运行训练脚本（已包括验证和测试逻辑）

   ```bash
   python train.py
   ```
3. 通过 WandB 或 TensorBoard 查看实时日志与可视化结果

## 📈 日志与可视化

* **WandB**

  ```bash
  wandb login
  ```

  查看实时训练曲线与预测对比图
* **TensorBoard**

  ```bash
  tensorboard --logdir <exp_dir>
  ```

## 📜 许可证

本项目遵循 **MIT License**。