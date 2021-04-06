import yaml
import os
import torch
import sys
import numpy as np
import cv2
sys.path.append(".")
from model import SegmentationModel
from segmentation.utils import pad_to_size, unpad_from_size
from helpers.parsing_cfg import get_args
from helpers.img_loader import ImageLoader
from albumentations.core.serialization import from_dict

def predict(inference_folder, file, model):
    img = ImageLoader.cv2_loader(os.path.join(inference_folder, file))
    height, width = img.shape[:2]
    img = augs(image=img)['image']
    pad_dict = pad_to_size((max(img.shape[:2]), max(img.shape[:2])), img)
    img = ImageLoader.torch_tensor_from_image(pad_dict['image']).to('cuda')
    img = img.unsqueeze(0)
    out = model.forward(img)
    mask = (out[0][0].cpu().numpy() > 0).astype(np.uint8) * 255
    mask = unpad_from_size(pad_dict['pads'], image=mask)["image"]
    mask = cv2.resize(
        mask, (width, height), interpolation=cv2.INTER_NEAREST
    )
    return mask


if __name__ == '__main__':
    args = get_args(inference=True, classification=False)
    with open(args.config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    inference_folder = args.folder
    output_folder = args.output_folder
    weights_file = args.weights
    model = SegmentationModel(params)
    training_result = torch.load(weights_file)
    if 'state_dict' not in training_result:
        print(f"Weights file {weights_file} load failure")
        raise ValueError
    else:
        model.load_state_dict(training_result['state_dict'])
        model.to('cuda')
    augs = from_dict(params['test_aug'])
    model.eval()
    result_arr = []
    with torch.no_grad():
        for file in os.listdir(inference_folder):
            mask = predict(inference_folder, file, model)
            cv2.imwrite(os.path.join(output_folder, f"{file}.png"), mask)

