import pydoc
import sys
from importlib import import_module
from pathlib import Path
from typing import Union
import argparse
from pathlib import Path
from addict import Dict

def get_args(inference=False, classification=True):
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    arg("-m", "--model_type", type=str, help="Model type(regression, classification, classification_big)", required=True)
    if not inference:
        return parser.parse_args()
    arg("-w", "--weights", type=Path, help="Path to the weights.", required=True)
    arg("-i", "--folder", type=Path, help="Path to the folder with source images.", required=True)
    arg("-t", "--threshold", type=float, help="Threshold for scores in object detection", required=False)
    if classification:
        arg("-p", "--predict_csv", type=Path, help="Path to the file with prediction results.", required=True)


    else:
        arg("-f", "--output_folder", type=Path, help="Path to the folder with models outputs.", required=True)
    return parser.parse_args()


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        else:
            return value
        raise ex


def py2dict(file_path: Union[str, Path]) -> dict:

    file_path = Path(file_path).absolute()

    if file_path.suffix != ".py":
        raise TypeError(f"Only Py file can be parsed, but got {file_path.name} instead.")

    if not file_path.exists():
        raise FileExistsError(f"There is no file at the path {file_path}")

    module_name = file_path.stem

    if "." in module_name:
        raise ValueError("Dots are not allowed in config file path.")

    config_dir = str(file_path.parent)

    sys.path.insert(0, config_dir)

    mod = import_module(module_name)
    sys.path.pop(0)
    cfg_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}

    return cfg_dict


def py2cfg(file_path: Union[str, Path]) -> ConfigDict:
    cfg_dict = py2dict(file_path)

    return ConfigDict(cfg_dict)


def object_from_dict(d, parent=None, reference=False, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)
    if reference:
        return pydoc.locate(object_type)
    return pydoc.locate(object_type)(**kwargs)
