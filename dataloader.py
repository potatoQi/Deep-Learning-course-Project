import lightning as L
from torch.utils.data import DataLoader
from dataset import MyDataset
from hydra.utils import instantiate

class DataModuleFromConfig(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config['batch_size']
        self.num_workers = config['num_workers']
        self.train_config = config['train']
        self.val_config = config['val']
        self.test_config = config['test']
    
    def setup(self, stage=None):
        self.train_dataset = instantiate(self.train_config)
        self.val_dataset = instantiate(self.val_config)
        self.test_dataset = instantiate(self.test_config)

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
        print(f'Train set size: {len(self.train_dataset)} episode files. Have been loaded.')
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(f'Validation set size: {len(self.val_dataset)} episode files. Have been loaded.')
        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        print(f'Test set size: {len(self.test_dataset)} episode files. Have been loaded.')
        return dataloader