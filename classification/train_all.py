import yaml
import sys
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
sys.path.append(".")

from model import LightingModel
from dataloader import ClassificationDataloader
from dataloader_big_images import ClassificationDataloaderBigImage
from model_regression import RegressionModel
from helpers.parsing_cfg import get_args

if __name__ == '__main__':
    args = get_args()
    with open(args.config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    comet_logger = pl_loggers.CometLogger(save_dir=params['logs_folder'])
    if args.model_type == 'classification':
        dm = ClassificationDataloader(params)
        model = LightingModel(params)
    elif args.model_type == 'classification_big':
        dm = ClassificationDataloaderBigImage(params)
        model = LightingModel(params)
    elif args.model_type == 'regression':
        dm = ClassificationDataloader(params)
        model = RegressionModel(params)
    model_checkpoint = ModelCheckpoint(monitor=params['monitor_metric'],
                                       save_top_k=1,
                                       verbose=True,
                                       filename="{epoch}_{val_loss:.4f}")
    early_stopping = EarlyStopping(params['monitor_metric'], patience=10)

    trainer = pl.Trainer(gpus=-1,max_epochs=30,callbacks=[model_checkpoint, early_stopping], logger=comet_logger)
    trainer.fit(model, dm)


