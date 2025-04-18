from lightning.pytorch.callbacks import Callback
import wandb

class CustomMetricCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        pass