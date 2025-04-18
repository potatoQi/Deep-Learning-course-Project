from lightning.pytorch.callbacks import Callback
import numpy as np
import torch
from dataset import get_data
import torch.nn.functional as F

class CustomMetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.dice_list = []

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # x_path = batch['feature_path']  # [b]
        # y_path = batch['label_path']    # [b]
        # chunk_size = batch['length'][0]
        # target_h, target_w = batch['size'][0]

        # for i in range(len(x_path)):
        #     x_data = get_data(x_path[i], use_simpleitk=True)
        #     y_data = get_data(y_path[i], use_simpleitk=True)

        #     y_data[y_data == 2] = 1

        #     x_data = np.clip(x_data, -300, 300)
        #     x_data = (x_data + 300) / 600.0
        #     x_data = x_data * 2 - 1

        #     x_data = torch.tensor(x_data).float().unsqueeze(0).to('cuda')  # [1 c h w]
        #     y_data = torch.tensor(y_data).float().unsqueeze(0).to('cuda')  # [1 c h w]

        #     dice = 0
        #     c = x_data.shape[1]
        #     num_chunks = (c + chunk_size - 1) // chunk_size  # 计算需要的块数

        #     for j in range(num_chunks):
        #         start_c = j * chunk_size
        #         end_c = min(start_c + chunk_size, c)

        #         x_chunk = x_data[:, start_c:end_c, :, :]  # [1, chunk_size, h, w]
        #         y_chunk = y_data[:, start_c:end_c, :, :]  # [1, chunk_size, h, w]

        #         if x_chunk.shape[1] != chunk_size:
        #             padding = chunk_size - x_chunk.shape[1]
        #             x_chunk = F.pad(x_chunk, (0, 0, 0, 0, 0, padding))  # 填充 c 维度
        #             y_chunk = F.pad(y_chunk, (0, 0, 0, 0, 0, padding))  # 填充 c 维度

        #         # 对 h, w 维度进行切割
        #         h, w = x_chunk.shape[2], x_chunk.shape[3]  # 当前切片的高度和宽度
        #         num_h_chunks = (h + target_h - 1) // target_h  # 计算需要的块数
        #         num_w_chunks = (w + target_w - 1) // target_w  # 计算需要的块数

        #         # 遍历切割的 h 和 w
        #         for h_idx in range(num_h_chunks):
        #             for w_idx in range(num_w_chunks):
        #                 start_h = h_idx * target_h
        #                 end_h = min(start_h + target_h, h)
        #                 start_w = w_idx * target_w
        #                 end_w = min(start_w + target_w, w)

        #                 x_subchunk = x_chunk[:, :, start_h:end_h, start_w:end_w]
        #                 y_subchunk = y_chunk[:, :, start_h:end_h, start_w:end_w]

        #                 # 填充不足部分
        #                 if x_subchunk.shape[2] < target_h or x_subchunk.shape[3] < target_w:
        #                     x_subchunk = F.pad(x_subchunk, (0, target_w - x_subchunk.shape[3], 0, target_h - x_subchunk.shape[2]))
        #                     y_subchunk = F.pad(y_subchunk, (0, target_w - y_subchunk.shape[3], 0, target_h - y_subchunk.shape[2]))

        #                 # 计算 Dice 系数
        #                 b, c, h, w = x_subchunk.shape
        #                 if b != 1 or c != chunk_size or h != target_h or w != target_w:
        #                     print('Fuck')
        #                     raise
        #                 logits = pl_module(x_subchunk)  # [1, c, target_h, target_w]
        #                 pred = torch.sigmoid(logits)
        #                 pred = (pred > 0.5).float()

        #                 dice += self.dice_compute(pred, y_subchunk).item()
            
        #     self.dice_list.append(dice)
        pass

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        mean_dice = sum(self.dice_list) / len(self.dice_list)
        self.dice_list = []
        self.log('val_dice', mean_dice, prog_bar=True, on_step=False, on_epoch=True)

    @torch.no_grad()
    def dice_compute(self, pred, target):
        intersection = (pred * target).sum((1, 2, 3))
        total = (pred + target).sum((1, 2, 3))
        dice = (2. * intersection + 1e-6) / (total + 1e-6)
        return dice.mean()
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        # TODO
        pass

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        mean_dice = sum(self.dice_list) / len(self.dice_list)
        self.dice_list = []
        self.log('test_dice', mean_dice, prog_bar=True, on_step=False, on_epoch=True)