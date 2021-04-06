from helpers.parsing_cfg import object_from_dict
import pytorch_lightning as pl
from torch import nn
import torch
import numpy as np
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image


class LightingModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.net = object_from_dict(self.params['model']['description'])

        if self.params['model']['description']['type'] == 'torchvision.models.inception_v3':
            self.inception = True
            in_features = self.net.AuxLogits.fc.in_features
            self.net.AuxLogits.fc = nn.Linear(in_features, self.params['model']['classes'])
            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, self.params['model']['classes'])
        else:
            self.inception = False
            in_features = self.net._fc.in_features
            self.net._fc = nn.Linear(in_features, self.params['model']['classes'])


    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = object_from_dict(self.params['optimizer'],
                                     params=[i for i in self.net.parameters() if i.requires_grad])
        scheduler = object_from_dict(self.params['scheduler'], optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        if self.inception:
            y_hat_1, y_hat_2 = self(x)
            loss_1 = F.cross_entropy(y_hat_1, y)
            loss_2 = F.cross_entropy(y_hat_2, y)
            loss = loss_1 + 0.4 * loss_2
            acc = accuracy(y_hat_1, y)
        else:
            y_hat = self(x)
            loss = F.cross_entropy(y_hat, y)
            acc = accuracy(y_hat, y)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch["x"], batch["y"]

        x = x.squeeze(0).permute(1,2,0)
        tiles_size = self.params['img_size']
        valid_batch = self.params['inference_batch']
        self.slicer = ImageSlicer(x.shape,
                                  tile_size=(tiles_size, tiles_size),
                                  tile_step=(tiles_size, tiles_size),
                                  weight='mean')
        tiles = [tensor_from_rgb_image(tile) for tile in (self.slicer.split(x.cpu().numpy()))]

        loss = []
        acc = []
        for tile in range(0, len(tiles), valid_batch):
            x = tiles[tile: tile+valid_batch]
            if len(x) < valid_batch:
                continue
            x = torch.stack(x).to('cuda')

            y_repeat = y.repeat(valid_batch)

            y_hat = (self(x))
            l = F.cross_entropy(y_hat, y_repeat)
            loss.append(l)
            acc.append(accuracy(y_hat, y_repeat))
        if len(loss) > 0:
            loss = torch.mean(torch.stack(loss).to('cuda'))
        else:
            loss = torch.ones([1]).to('cuda')
        if len(acc) > 0:
            acc = torch.mean(torch.stack(acc).to('cuda'))
        else:
            acc = torch.zeros([1]).to('cuda')


        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)