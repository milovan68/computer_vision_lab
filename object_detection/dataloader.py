from torch.utils.data import DataLoader, Dataset
import torch
from helpers.img_loader import ImageLoader
import json
from torchvision.datasets import VisionDataset
from collections import defaultdict
import numpy as np
import pickle
from albumentations.pytorch.transforms import ToTensorV2



class SuperviselyDetectionDataset(VisionDataset):
    def __init__(self,
                 samples,
                 class_dict,
                 length=None,
                 transforms=None):

        self.class_dict = class_dict
        self.samples = samples
        self.transforms = transforms
        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        index = index % len(self.samples)
        image_path, label_path = self.samples[index]
        img = ImageLoader.cv2_loader(image_path)
        target = self.parse_json(label_path)

        labels = [bbox[0] for bbox in target]
        bboxes = [bbox[1:] for bbox in target]
        image_dict = {
            'image': img,
            'bboxes': bboxes,
            'labels': labels
        }
        if self.transforms is not None:
            sample = self.transforms(**image_dict)
            img, bboxes = sample["image"], sample["bboxes"]

        if isinstance(img, np.ndarray):
            img = torch.tensor(img).permute(2, 0, 1)



        return {
            "image_id": image_path.stem,
            "features": img.float(),
            "boxes": torch.FloatTensor(bboxes),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }


    def __len__(self):
        return len(self.samples)


    def parse_json(self, json_file):
        with open(json_file, 'r') as f:
            json_file = json.load(f)
            size = json_file['size']
            bbox_list = []
            for j_dict in (json_file['objects']):
                if (j_dict['geometryType']) == 'rectangle':
                    if 'points' in j_dict:
                        bbox_ = j_dict['points']['exterior']
                        bbox_list.append([self.class_dict[j_dict['classTitle']],
                                          bbox_[0][0],
                                          bbox_[0][1],
                                          bbox_[1][0],
                                          bbox_[1][1]])
        return bbox_list

