from lightning.pytorch.callbacks import Callback
import numpy as np
import torch
from dataset import get_data
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

class CustomMetricCallback(Callback):
    def __init__(self):
        super().__init__()
        self.dice_list = []

    @torch.no_grad()
    def cal(self, pl_module, batch):
        x_data = batch['feature']  # [b,]
        y_data = batch['label']    # [b,]
        x_data = x_data.to(pl_module.device)  # [b, c, h, w]
        y_data = y_data.to(pl_module.device)  # [b, h, w]
        y_preds = pl_module(x_data)  # [b, cls, h, w]
        y_preds = torch.argmax(y_preds, dim=1)
        with tqdm(total=x_data.shape[0], desc=f"Processing batches", unit="batch") as pbar1:
            dice_sum = torch.tensor([0.0]).to(pl_module.device)
            for i in range(x_data.shape[0]):
                y = y_data[i] # [h w]
                y_pred = y_preds[i] # [h w]
                # print(y.shape, y_pred.shape)
                # 保证原始的信息
                transform = transforms.Compose([
                    transforms.Lambda(lambda x: Image.fromarray((x.cpu().numpy()*255).astype(np.uint8))),
                    transforms.Resize((512, 512), interpolation=Image.NEAREST),
                    transforms.ToTensor()
                ])
                y = transform(y)
                y_pred = transform(y_pred)

                gt_w = y.sum().float()
                pred_w = y_pred.sum().float()
                coi_w = ((y_pred == y) & (y_pred == 1)).sum().float()
                dice_sum += 2 * coi_w / (pred_w + gt_w + 1e-6)
            dice_sum /= x_data.shape[0]
            pbar1.update(1)

        self.dice_list.append(dice_sum.item())

    @torch.no_grad()
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.cal(pl_module, batch)

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        mean_dice = sum(self.dice_list) / len(self.dice_list)
        self.dice_list = []
        self.log('val_dice', mean_dice, prog_bar=True, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.cal(pl_module, batch)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        mean_dice = sum(self.dice_list) / len(self.dice_list)
        self.dice_list = []
        self.log('test_dice', mean_dice, prog_bar=True, on_step=False, on_epoch=True)