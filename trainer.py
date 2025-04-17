import torch
import numpy as np
from omegaconf import OmegaConf
from dataloader import DataModuleFromConfig
import datetime, argparse, os
import lightning as L
from lightning.pytorch.loggers import WandbLogger   # 用来实例化 wandb_logger 的
from lightning.pytorch.callbacks import ModelCheckpoint # 用来实例化 checkpoint_callback 的
from UNet import UNet
from utils import get_latest_checkpoint, CheckpointCleanupCallback

if __name__ == '__main__':
    config = OmegaConf.load('config.yaml')
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    L.seed_everything(config.Other.seed, workers=True)

    # 数据集
    data_module = DataModuleFromConfig(config.Dataset)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    # 准备模型
    unet = UNet()

    # 配置 wandb
    wandb_logger = WandbLogger(
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

    # 定期保存 ckpt
    ckpt_dir = config.Other.ckpt_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='epoch-{epoch}_step-{step}',
        every_n_epochs=config.Other.ckpt_save_interval,  # 10 个 epoch 保存一次
    )
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir=ckpt_dir)

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.Other.max_epochs,
        log_every_n_steps=config.Other.log_every_n_steps,    # 每 1 个 step 打一次 log
        callbacks=[checkpoint_callback, CheckpointCleanupCallback(ckpt_dir, config.Other.ckpt_save_num)],
    )

    trainer.fit(
        model=unet,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=latest_checkpoint,
    )

    trainer.test(model=unet, dataloaders=test_loader)