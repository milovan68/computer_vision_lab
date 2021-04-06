import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
from object_detection.utils import get_samples, _evaluate_iou
from albumentations.core.serialization import from_dict
from torch.utils.data import DataLoader
from object_detection.dataloader import SuperviselyDetectionDataset

class FasterRCNN(pl.LightningModule):

    def __init__(
        self,
        class_dict = None,
        params = None,
        learning_rate: float = 0.0001,
        num_classes: int = 91,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        trainable_backbone_layers: int = 3,
        replace_head: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.params = params
        self.class_dict = class_dict

        model = fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained_backbone,
            trainable_backbone_layers=trainable_backbone_layers,
        )

        if replace_head:
            in_features = model.roi_heads.box_predictor.cls_score.in_features

            head = faster_rcnn.FastRCNNPredictor(in_features, self.params['num_classes'])
            model.roi_heads.box_predictor = head
        else:
            assert num_classes == 91, "replace_head must be true to change num_classes"

        self.model = model
        self.learning_rate = learning_rate

    def setup(self, stage):
        samples = get_samples(self.params['image_path'], self.params['label_path'])

        num_train = int((1 - self.params["val_split"]) * len(samples))

        self.train_samples = samples[:num_train]
        self.val_samples = samples[num_train:]

        print("Len train samples = ", len(self.train_samples))
        print("Len val samples = ", len(self.val_samples))

    def train_dataloader(self):
        train_aug = from_dict(self.params["train_aug"])

        if "epoch_length" not in self.params["train_parameters"]:
            epoch_length = None
        else:
            epoch_length = self.params["train_parameters"]["epoch_length"]

        result = DataLoader(
            SuperviselyDetectionDataset(samples=self.train_samples,
                                        transforms=train_aug,
                                        class_dict=self.class_dict),
            batch_size=self.params["train_parameters"]["batch_size"],
            num_workers=self.params["num_workers"],
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        print("Train dataloader = ", len(result))
        return result

    def val_dataloader(self):
        val_aug = from_dict(self.params["val_aug"])

        result = DataLoader(
            SuperviselyDetectionDataset(samples=self.val_samples,
                                        transforms=val_aug,
                                        class_dict=self.class_dict),
            batch_size=self.params["val_parameters"]["batch_size"],
            num_workers=self.params["num_workers"],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        print("Val dataloader = ", len(result))

        return result

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):

        images = batch['features']
        targets = []

        for idx in range(images.shape[0]):
            d = {}
            d['boxes'] = torch.squeeze(batch['boxes'], 0)
            d['labels'] = torch.squeeze(batch['labels'].to('cuda'), 0)

            targets.append(d)
        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):

        images = batch['features']

        targets = [batch]
        outs = self.model(images)

        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

