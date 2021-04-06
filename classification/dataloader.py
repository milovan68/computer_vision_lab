from torch.utils.data import Dataset
import numpy as np
from torchvision import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from albumentations.core.serialization import from_dict
from helpers.parsing_cfg import object_from_dict
from albumentations.pytorch.transforms import ToTensorV2


class ClassificationDataset(Dataset):
    def __init__(self, params, imgfolder: str, train: bool = True, transform = None):
        self.imgfolder = imgfolder
        self.train = train
        self.aug = transform
        self.params = params
        self.FolderDataset = datasets.ImageFolder(self.imgfolder,
                                                  loader=object_from_dict(self.params['img_loader']['description'],
                                                                          reference=True),
                                                  )

    def __getitem__(self, idx):
        x, y = self.FolderDataset[idx]
        x = np.asarray(x)
        if (self.aug):
            x = self.aug(image=x)['image']
        if self.train:
            return {'x': x, 'y': y}
        return {'x': x}

    def __len__(self):
        return len(self.FolderDataset)



class ClassificationDataloader(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_transform = from_dict(self.params['train_aug'])
        self.valid_transform = from_dict(self.params['val_aug'])


    def setup(self, stage=None):
        self.train_dataset = ClassificationDataset(self.params,
                                                   self.params['train_folder'],
                                                   train=True,
                                                   transform=self.train_transform)
        self.val_dataset = ClassificationDataset(self.params,
                                                 self.params['val_folder'],
                                                 train=True,
                                                 transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['batch_size'], num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params['batch_size'], num_workers=4)


class InferenceDataloader(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.transform = self.params['test']

    def setup(self, stage=None):
        self.dataset = ClassificationDataset(self.params,
                                             self.params['train_folder'],
                                             train=False,
                                             transform=self.transform)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.params['batch_size'], num_workers=4)

