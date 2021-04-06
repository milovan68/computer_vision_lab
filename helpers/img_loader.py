import cv2
from PIL import Image
import numpy as np
import torch


class ImageLoader:

    @staticmethod
    def cv2_loader(path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def pil_loader(path):
        with open(str(path), 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    @staticmethod
    def cv2_grayscale_loader(path):
        return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    @staticmethod
    def torch_tensor_from_image(nparray, extra_dimension=True):
        if len(nparray.shape) not in {2, 3}:
            raise ValueError(f"Image must have shape [H,W] or [H,W,C]. Got image with shape {nparray.shape}")

        if len(nparray.shape) == 2:
            if extra_dimension:
                image = np.expand_dims(nparray, 0)
        else:
            # HWC -> CHW
            image = np.moveaxis(nparray, -1, 0)
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image)
        return image