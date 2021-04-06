import yaml
import sys
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint,EarlyStopping
sys.path.append(".")

from object_detection.dataloader import SuperviselyDetectionDataset
from object_detection.utils import parse_annot_for_class_dict
from object_detection.model import FasterRCNN
from helpers.parsing_cfg import get_args
from helpers.parsing_cfg import object_from_dict

if __name__ == '__main__':
    args = get_args()

    with open(args.config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    class_dict = parse_annot_for_class_dict(params)
    model = FasterRCNN(params=params, class_dict=class_dict, replace_head=True)

    comet_logger = pl_loggers.CometLogger(save_dir=params['logs_folder'])

    trainer = object_from_dict(
        params["trainer"],
        logger=comet_logger,
        checkpoint_callback=object_from_dict(params["checkpoint_callback"]),
    )
    trainer.fit(model)