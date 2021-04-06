from pathlib import Path
from torchvision.ops import box_iou
import torch
import os
import pickle
import json
from collections import defaultdict

def get_id2_file_paths(path, labels=False):
    if labels:
        return {x.stem: x for x in Path(path).glob("*.*")}
    return {x.name: x for x in Path(path).glob("*.*")}


def get_samples(image_path, labels_path):
    image2path = get_id2_file_paths(image_path)
    label2path = get_id2_file_paths(labels_path, labels=True)
    return [(image_file_path, label2path[file_id]) for file_id, image_file_path in image2path.items()]


def _evaluate_iou(target, pred):
    """
    Evaluate intersection over union (IOU) for target from dataset and output prediction
    from model
    """
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)

    return box_iou(torch.squeeze(target["boxes"], 0), pred["boxes"]).diag().mean()

def parse_annot_for_class_dict(params):
    path = params['label_path']
    pickle_file = params['class_idx_dict']
    class_dict = defaultdict(int)
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            json_file = json.load(f)
            for j_dict in (json_file['objects']):
                if j_dict['classTitle'] not in class_dict:
                    class_dict[j_dict['classTitle']] += len(class_dict)

    with open(pickle_file, 'wb') as f:
        pickle.dump(class_dict, f)
    return class_dict