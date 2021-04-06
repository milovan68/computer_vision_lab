import yaml
import os
import torch
import sys
sys.path.append(".")
from model_big_images import LightingModel
import pandas as pd
from helpers.parsing_cfg import get_args
from helpers.img_loader import ImageLoader
from albumentations.core.serialization import from_dict
from albumentations.pytorch.transforms import ToTensorV2
from pytorch_toolbelt.inference.tiles import ImageSlicer
from pytorch_toolbelt.utils.torch_utils import tensor_from_rgb_image
import numpy as np

def get_class_idx(path):
    classes_names = os.listdir(path)
    classes_names.sort()
    return {idx: class_ for idx, class_ in enumerate(classes_names)}

if __name__ == '__main__':
    args = get_args(inference=True)
    with open(args.config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    inference_folder = args.folder
    result_file = args.predict_csv
    weights_file = args.weights
    model = LightingModel(params)
    training_result = torch.load(weights_file)
    if 'state_dict' not in training_result:
        print(f"Weights file {weights_file} load failure")
        raise ValueError
    else:
        model.load_state_dict(training_result['state_dict'])
        model.to('cuda')
    class_to_idx = get_class_idx(params['train_folder'])
    augs = from_dict(params['test_aug'])
    model.eval()
    result_arr = []
    with torch.no_grad():
        for file in os.listdir(inference_folder):
            img = ImageLoader.cv2_loader(os.path.join(inference_folder, file))
            slicer = ImageSlicer(img.shape, tile_size=(299, 299), tile_step=(299, 299), weight='mean')
            tiles = [augs(image=tile)['image'] for tile in (slicer.split(img))]
            img = torch.stack(tiles).to('cuda')
            out = model.forward(img)
            _, predicted = torch.max(out.data, 1)
            predicted_idx = np.bincount(predicted.cpu().numpy()).argmax()
            print(class_to_idx[predicted_idx])
            result_arr.append({'file': file, 'prediction': class_to_idx[predicted_idx]})
    pd.DataFrame(result_arr).to_csv(result_file, index=False)

