from helpers.img_loader import ImageLoader
from torch.utils.data import Dataset
import numpy as np
import torch


class SegmentationDataset(Dataset):
    def __init__(self, samples, transform, length=None):
        self.samples = samples
        self.transform = transform
        if length is None:
            self.length = len(self.samples)
        else:
            self.length = length


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples)

        image_path, mask_path = self.samples[idx]
        image = ImageLoader.cv2_loader(image_path)
        mask = ImageLoader.cv2_grayscale_loader(mask_path)

        # apply augmentations
        sample = self.transform(image=image, mask=mask)
        image, mask = sample["image"], sample["mask"]

        mask = (mask > 0).astype(np.uint8)

        mask = torch.from_numpy(mask)

        return {
            "image_id": image_path.stem,
            "features": ImageLoader.torch_tensor_from_image(image),
            "masks": torch.unsqueeze(mask, 0).float(),
        }
