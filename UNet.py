import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
import torch.optim as optim

class UNet(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(16 * 512 * 512, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 16 * 512 * 512),
        )
    
    def training_step(self, batch, batch_idx):
        data = batch    # x: [b c h w]
        x = data['feature']
        x = self.flatten(x)    # x: [b, c*h*w]
        z = self.encoder(x)
        x_hat = self.decoder(z) # x_hat: [b, c*h*w]
        loss = F.mse_loss(x_hat, x)
        # 下面这句话是直接 log 到 tensorboard
        self.log("train_loss", loss)
        return loss
    
    # configure_optimizers 是 lightning model 必写的一个函数
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer
    
    # 如果传入了 val_dataloader, 那么就要实现一下 validation_step
    # lightning 默认每个 epoch 训练完就会跑一遍验证集
    def validation_step(self, batch, batch_idx):
        data = batch    # x: [b c h w]
        x = data['feature']
        x = self.flatten(x)    # x: [b, c*h*w]
        z = self.encoder(x)
        x_hat = self.decoder(z) # x_hat: [b, c*h*w]
        loss = F.mse_loss(x_hat, x)
        # prog_bar=True: 在进度条上显示 val_loss
        # on_epoch=True: tensorboard 里只记录验证集执行完的指标
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        data = batch    # x: [b c h w]
        x = data['feature']
        x = self.flatten(x)    # x: [b, c*h*w]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("test_loss", loss, prog_bar=True)
        return loss