from helpers.parsing_cfg import object_from_dict
import pytorch_lightning as pl
from torch import nn
import torch
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy


class RegressionModel(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.net = object_from_dict(self.params['model']['description'])
        if self.params['model']['description']['type'] == 'torchvision.models.inception_v3':
            self.inception = True
            in_features = self.net.AuxLogits.fc.in_features
            self.net.AuxLogits.fc = nn.Linear(in_features, 1)
            num_ftrs = self.net.fc.in_features
            self.net.fc = nn.Linear(num_ftrs, 1)
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
        loss_ = nn.MSELoss()
        if self.inception:
            y_hat_1, y_hat_2 = self(x)
            y_hat_1 = y_hat_1.squeeze(dim=1)
            y_hat_2 = y_hat_2.squeeze(dim=1)
            loss_1 = loss_(y_hat_1, y.float())
            loss_2 = loss_(y_hat_2, y.float())
            loss = loss_1 + 0.4 * loss_2
        else:
            y_hat = self(x)
            y_hat = y_hat.squeeze(dim=1)
            loss = loss_(y_hat, y.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y = batch["x"], batch["y"]
        y_hat = self(x)
        loss_ = nn.MSELoss()
        y_hat = y_hat.squeeze(dim=1)
        loss = loss_(y_hat, y.float())
        self.log("val_loss", loss, prog_bar=True, logger=True)
