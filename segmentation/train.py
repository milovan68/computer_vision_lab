import yaml
import sys
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
sys.path.append(".")

from segmentation.dataloader import SegmentationDataset
from segmentation.model import SegmentationModel
from helpers.parsing_cfg import get_args
from helpers.parsing_cfg import object_from_dict

if __name__ == '__main__':
    args = get_args()

    with open(args.config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    model = SegmentationModel(params)

    comet_logger = pl_loggers.CometLogger(save_dir=params['logs_folder'])
    trainer = object_from_dict(
        params["trainer"],
        logger=comet_logger,
        checkpoint_callback=object_from_dict(params["checkpoint_callback"]),
    )
    trainer.fit(model)
