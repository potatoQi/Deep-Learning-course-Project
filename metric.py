from lightning.pytorch.callbacks import Callback
import numpy as np
import torch
from dataset import get_data
import torch.nn.functional as F
from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb

class CustomMetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.dice_list = []
        self.fake_label_list = []
        self.real_label_list = []
        self.img_tim = 0

    @torch.no_grad()
    def cal_3D(self, pl_module, batch):
        x_path = batch['feature_path']  # [b]
        y_path = batch['label_path']    # [b]
        chunk_size = batch['length'][0]         # 要切的长度: 16
        size = batch['size'][0]                 # list: [h, w]
        target_h, target_w = size[0], size[1]

        with tqdm(total=len(x_path), desc=f"Processing batches", unit="batch") as pbar1:
            for i in range(len(x_path)):
                x_data = get_data(x_path[i], use_simpleitk=True)    # [c h w]
                y_data = get_data(y_path[i], use_simpleitk=True)    # [c h w]

                y_data[y_data == 2] = 1

                x_data = np.clip(x_data, -300, 300)
                x_data = (x_data + 300) / 600.0
                x_data = x_data * 2 - 1

                x_data = torch.tensor(x_data).float().unsqueeze(0)  # [1 c h w]
                y_data = torch.tensor(y_data).float().unsqueeze(0)  # [1 c h w]

                gt_w = y_data.sum()  # ground_truth 的白色像素个数
                pred_w = 0  # 预测的白色像素个数
                coi_w = 0  # 重合的白色像素个数

                # 其实可以看成一张有 c * h * w 个像素的大图片, 去计算它的 DICE
                # 我每次取出的小块是 chunk_size * target_h * target_w 大小的图片
                # 如果要验证正确性的话, 可以每张图片都是 9 个像素, 然后每张图片黑白任意组合, 但是要保证每张图都有 4 个白像素且重合白像素只有一个
                # 按理说, 这样算出来的结果是 1/4

                # 验证代码
                # x_data = torch.tensor([
                #     [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                #     [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                # ]).float().unsqueeze(0)   # [1, 2, 3, 3]
                # y_data = torch.tensor([
                #     [[1, 1, 0], [1, 1, 0], [0, 0, 0]],
                #     [[0, 0, 0], [0, 1, 1], [0, 1, 1]],
                # ]).float().unsqueeze(0)   # [1, 2, 3, 3]
                # chunk_size = 1
                # target_h = 2
                # target_w = 2
                # gt_w = y_data.sum()

                c = x_data.shape[1]
                num_chunks = (c + chunk_size - 1) // chunk_size  # 计算需要的块数
                total_tim = (num_chunks * ((512 + target_h - 1) // target_h) * ((512 + target_w - 1) // target_w)).item()

                with tqdm(total=total_tim, desc=f"Processing chunk", unit="chunk") as pbar2:
                    for j in range(num_chunks):
                        start_c = j * chunk_size
                        end_c = min(start_c + chunk_size, c+1)

                        x_chunk = x_data[:, start_c:end_c, :, :]  # [1, chunk_size, h, w]
                        y_chunk = y_data[:, start_c:end_c, :, :]  # [1, chunk_size, h, w]

                        if x_chunk.shape[1] != chunk_size:
                            padding = chunk_size - x_chunk.shape[1]
                            x_chunk = F.pad(x_chunk, (0, 0, 0, 0, 0, padding))  # 填充 c 维度
                            y_chunk = F.pad(y_chunk, (0, 0, 0, 0, 0, padding))  # 填充 c 维度

                        # 对 h, w 维度进行切割
                        h, w = x_chunk.shape[2], x_chunk.shape[3]  # 当前切片的高度和宽度
                        num_h_chunks = (h + target_h - 1) // target_h  # 计算需要的块数
                        num_w_chunks = (w + target_w - 1) // target_w  # 计算需要的块数

                        # 遍历切割的 h 和 w
                        for h_idx in range(num_h_chunks):
                            for w_idx in range(num_w_chunks):
                                start_h = h_idx * target_h
                                end_h = min(start_h + target_h, h+1)
                                start_w = w_idx * target_w
                                end_w = min(start_w + target_w, w+1)

                                x_subchunk = x_chunk[:, :, start_h:end_h, start_w:end_w]
                                y_subchunk = y_chunk[:, :, start_h:end_h, start_w:end_w]

                                # 填充不足部分
                                if x_subchunk.shape[2] < target_h or x_subchunk.shape[3] < target_w:
                                    x_subchunk = F.pad(x_subchunk, (0, target_w - x_subchunk.shape[3], 0, target_h - x_subchunk.shape[2]))
                                    y_subchunk = F.pad(y_subchunk, (0, target_w - y_subchunk.shape[3], 0, target_h - y_subchunk.shape[2]))

                                b, c, h, w = x_subchunk.shape
                                if b != 1 or c != chunk_size or h != target_h or w != target_w:
                                    raise ValueError('指标计算切片这里出现了维度问题')
                                logits = pl_module(x_subchunk.to('cuda')).to('cpu')  # [1, c, target_h, target_w]
                                pred = torch.sigmoid(logits)
                                coi_w += ((pred > 0.5) & (y_subchunk == 1)).float().sum()
                                pred_w += (pred > 0.5).float().sum()
                                pbar2.update(1)
                    
                    pbar1.update(1)
                
                # 现在有了 pred_w 和 gt_w, 可以计算 DICE 系数了
                xx = 2 * coi_w
                yy = pred_w + gt_w + 1e-6
                self.dice_list.append(xx / yy)

    @torch.no_grad()
    def cal_2D(self, pl_module, batch):
        x_data = batch['feature'].to(pl_module.device)  # [b c h w]
        y_data = batch['label'].to(pl_module.device)    # [b c h w]
        logits = pl_module(x_data)  # [b c h w]
        pred = torch.sigmoid(logits)

        coi_w = ((pred > 0.5) & (y_data == 1)).float().sum()
        pred_w = (pred > 0.5).float().sum()
        gt_w = y_data.sum()
        dice = 2 * coi_w / (pred_w + gt_w + 1e-6)
        self.dice_list.append(dice.item())

        self.img_tim += 1
        if self.img_tim <= 10:
            self.fake_label_list.append((pred > 0.5)[0].cpu().numpy())     # [c h w]
            self.real_label_list.append(y_data[0].cpu().numpy())           # [c h w]

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # self.cal_3D(pl_module, batch)
        self.cal_2D(pl_module, batch)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        mean_dice = sum(self.dice_list) / len(self.dice_list)
        self.log('val_dice', mean_dice, prog_bar=True, on_step=False, on_epoch=True)

        fake_np = np.stack(self.fake_label_list, axis=0)  # [N, C, H, W]
        real_np = np.stack(self.real_label_list, axis=0)
        logger = trainer.logger
        if isinstance(logger, WandbLogger):
            wb = logger.experiment
            wb.log({
                "val_fake_images": [wandb.Image(img) for img in fake_np],
                "val_real_images": [wandb.Image(img) for img in real_np],
            }, step=trainer.current_epoch)
        elif isinstance(logger, TensorBoardLogger):
            tb = logger.experiment
            tb.add_images("val/fake", fake_np, trainer.current_epoch, dataformats="NCHW")
            tb.add_images("val/real", real_np, trainer.current_epoch, dataformats="NCHW")

        self.dice_list = []
        self.fake_label_list = []
        self.real_label_list = []
        self.img_tim = 0
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # self.cal_3D(pl_module, batch)
        self.cal_2D(pl_module, batch)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        mean_dice = sum(self.dice_list) / len(self.dice_list)
        self.dice_list = []
        self.log('test_dice', mean_dice, prog_bar=True, on_step=False, on_epoch=True)