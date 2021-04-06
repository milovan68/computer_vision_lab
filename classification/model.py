from helpers.parsing_cfg import object_from_dict
import pytorch_lightning as pl
from torch import nn
from torch.nn import functional as F
from pytorch_lightning.metrics.functional import accuracy


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
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log("val_acc", acc, prog_bar=True, logger=True),
        self.log("val_loss", loss, prog_bar=True, logger=True)