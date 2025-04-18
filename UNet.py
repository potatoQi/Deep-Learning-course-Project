import torch
from torch import nn
import lightning as L
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample, num_heads, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
    
        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers)
        ])
        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers)
        ])
        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])

        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])
        self.down_sample_cov = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if down_sample else nn.Identity()

    def forward(self, x):
        # x: [b c h w]
        for i in range(self.num_layers):
            res = x
            x = self.resnet_conv_1[i](x)
            x = self.resnet_conv_2[i](x)
            x = self.residual_conv[i](res) + x

            b, c, h, w = x.shape
            res = x
            x = self.norm_layers[i](x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.attention_layers[i](x, x, x)[0]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = res + x
        x = self.down_sample_cov(x)
        return x
    
class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers + 1)
        ])
        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers + 1)
        ])
        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers + 1)
        ])

        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [b c h w]
        res = x
        x = self.resnet_conv_1[0](x)
        x = self.resnet_conv_2[0](x)
        x = self.residual_conv[0](res) + x
    
        for i in range(self.num_layers):
            res = x
            x = self.resnet_conv_1[i + 1](x)
            x = self.resnet_conv_2[i + 1](x)
            x = self.residual_conv[i + 1](res) + x

            b, c, h, w = x.shape
            res = x
            x = self.norm_layers[i](x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.attention_layers[i](x, x, x)[0]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = res + x
        
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample, num_heads, num_layers=1):
        super().__init__()
        self.num_layers = num_layers

        self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1) if up_sample else nn.Identity()

        self.resnet_conv_1 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for i in range(num_layers)
        ])
        self.resnet_conv_2 = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_layers)
        ])

        self.residual_conv = nn.ModuleList([
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
            for i in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(8, out_channels) for _ in range(num_layers)
        ])
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(out_channels, num_heads, batch_first=True) for _ in range(num_layers)
        ])

    def forward(self, x, out_down):
        # x: [b in_channels//2 h//2 w//2]
        # out_down: [b in_channels//2 h w]
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)

        for i in range(self.num_layers):
            res = x
            x = self.resnet_conv_1[i](x)
            x = self.resnet_conv_2[i](x)
            x = self.residual_conv[i](res) + x

            b, c, h, w = x.shape
            res = x
            x = self.norm_layers[i](x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.attention_layers[i](x, x, x)[0]
            x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
            x = res + x

        return x

class UNet(L.LightningModule):
    def __init__(
        self,

        im_channels=3,

        down_channels=[32, 64, 128], # 输入通道数
        down_sample=[True, True, False],

        num_heads=4,
        num_down_layers=1,
        num_mid_layers=1,
        num_up_layers=1,

        lr=1e-4,
    ):
        super().__init__()
        self.lr = lr

        self.im_channels = im_channels
        self.conv_in = nn.Conv2d(im_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        for i in range(len(down_channels)-1):
            assert down_channels[i] * 2 == down_channels[i+1], f"第{i}层的输出通道数不等于第{i+1}层的输入通道数"

        self.downs = nn.ModuleList([
            DownBlock(
                in_channels=down_channels[i],
                out_channels=down_channels[i] * 2,
                down_sample=down_sample[i],
                num_heads=num_heads,
                num_layers=num_down_layers
            ) for i in range(len(down_channels))
        ])
        self.mids = nn.ModuleList([
            MidBlock(
                in_channels=down_channels[-1] * 2,
                out_channels=down_channels[-1] * 2 if i != len(down_channels) - 1 else down_channels[-1],
                num_heads=num_heads,
                num_layers=num_mid_layers
            ) for i in range(len(down_channels))
        ])
        self.ups = nn.ModuleList([
            UpBlock(
                in_channels=down_channels[i] * 2,
                out_channels=down_channels[i-1] if i != 0 else down_channels[0],
                up_sample=down_sample[i],
                num_heads=num_heads,
                num_layers=num_up_layers
            ) for i in reversed(range(len(down_channels)))
        ])
        self.norm = nn.GroupNorm(8, down_channels[0])
        self.silu = nn.SiLU()
        self.conv_out = nn.Conv2d(down_channels[0], im_channels, kernel_size=3, stride=1, padding=1)
        self.loss_fn = nn.BCEWithLogitsLoss()   # 二元交叉熵损失
    
    def training_step(self, batch, batch_idx):
        x = batch['feature']    # [b c h w]
        y = batch['label']      # [b c h w]
        
        x = self.conv_in(x)
        down_outs = []
        for block in self.downs:
            down_outs.append(x)
            x = block(x)
        for block in self.mids:
            x = block(x)
        for block in self.ups:
            x = block(x, down_outs.pop())
        x = self.norm(x)
        x = self.silu(x)
        x = self.conv_out(x)    # [b c h w], logits

        loss = self.loss_fn(x, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss
    
    # configure_optimizers 是 lightning model 必写的一个函数
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    # 必须实现 validation_step, 回调钩子里的 on_validation_batch_end, on_validation_epoch_end 才会被调用    
    def validation_step(self, batch, batch_idx):
        return None
    
    # 同理, 占位, 具体逻辑实现见钩子
    def test_step(self, batch, batch_idx):
        return None