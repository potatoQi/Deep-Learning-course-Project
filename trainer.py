import torch
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import instantiate
from dataloader import DataModuleFromConfig
import datetime, argparse, os, glob
import lightning as L
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint # 用来实例化 checkpoint_callback 的
from UNet import UNet
from utils import get_latest_checkpoint, CheckpointCleanupCallback
from metric import CustomMetricCallback

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    L.seed_everything(config.Trainer.seed, workers=True)
    if not os.path.exists(config.Trainer.exp_dir):
        os.makedirs(config.Trainer.exp_dir)

    # 数据集
    data_module = DataModuleFromConfig(config.Dataset)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # 准备模型
    unet = instantiate(config.UNet)

    logger = config.Trainer.logger
    if logger == 'wandb':
        # 配置 wandb
        logger = WandbLogger(
            # Team 名称
            entity='Error_666',
            # 项目名称
            project='medical',
            # 运行名称 (便于在运行列表中识别)
            name=now,
            # 运行的 ID (便于在域名中识别, id 不能重复, 重复会有奇怪的 bug)
            id=now,
            # 要记录的 metadata     可以通过 run.config.update() 追加
            config=OmegaConf.to_container(config, resolve=True),
        )
    elif logger == 'tensorboard':
        logger = TensorBoardLogger(
            save_dir=os.path.join(config.Trainer.exp_dir, 'tensorboard'),  # 日志根目录
            name='0',             # 子文件夹名，也可以用 now
            version=now           # run 的版本号，方便区分
        )
    else:
        logger = False

    # 定期保存 ckpt
    ckpt_dir = os.path.join(config.Trainer.exp_dir, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='{epoch}_{step}',
        every_n_epochs=config.Trainer.ckpt_save_interval,  # 10 个 epoch 保存一次
        save_top_k=-1,  # 保存所有检查点, 自定义的 CheckpointCleanupCallback 会删除多余的检查点
    )

    last_config = None
    if os.path.exists('last_config.yaml'):
        with open('last_config.yaml', 'r') as f:
            last_config = OmegaConf.load(f)
    latest_checkpoint = None
    if config.Trainer.use_ckpt:
        if last_config is not None and OmegaConf.to_container(last_config) == OmegaConf.to_container(config):
            latest_checkpoint = get_latest_checkpoint(checkpoint_dir=ckpt_dir)
            print('配置与上次一样, 加载的检查点:', latest_checkpoint)
        else:
            print('配置与上次不同, 正在重新训练...')
            OmegaConf.save(config, 'last_config.yaml')
            ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
            for ckpt_file in ckpt_files:
                os.remove(ckpt_file)
    else:
        ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
        for ckpt_file in ckpt_files:
            os.remove(ckpt_file)

    trainer = L.Trainer(
        logger=logger,
        max_epochs=config.Trainer.max_epochs,
        log_every_n_steps=config.Trainer.log_every_n_steps,    # 每 x 个 step 打一次 训练log
        check_val_every_n_epoch=config.Trainer.check_val_every_n_epoch,  # 每 x 个 epoch 验证一次
        precision=16,   # 开 fp16 精度训练
        callbacks=[
            checkpoint_callback,
            CheckpointCleanupCallback(ckpt_dir, config.Trainer.ckpt_save_num),
            CustomMetricCallback(),
        ],
    )

    trainer.fit(
        model=unet,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=latest_checkpoint,
    )

    trainer.test(model=unet, dataloaders=test_loader)