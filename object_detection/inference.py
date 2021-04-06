import yaml
import os
import torch
import numpy as np
import cv2
from object_detection.model import FasterRCNN
from helpers.parsing_cfg import get_args
from helpers.img_loader import ImageLoader
from albumentations.core.serialization import from_dict
import pickle
from albumentations.pytorch.transforms import ToTensorV2


def predict(inference_folder, file, model, augs):
    img = ImageLoader.cv2_loader(os.path.join(inference_folder, file))
    #because albumentations library wants labels and boxes for augmentation
    test_dict = {
        'image': img,
        'bboxes': torch.as_tensor([[0]], dtype=torch.float32),
        'labels': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32)
    }
    if augs:
        img = augs(**test_dict)['image']
    img = img.unsqueeze(0)
    out = model.forward(img.to('cuda').float())
    return out

def draw_box(inference_folder, file, bboxes, threshold, class_dict, colors):
    bbox = bboxes['boxes']
    scores = bboxes['scores']
    labels = bboxes['labels']
    ids_to_cats = {v: k for k, v in class_dict.items()}
    bbox_after_scoring = bbox[scores > threshold].cpu().numpy()
    labels_after_scoring = labels[scores > threshold].cpu().numpy()
    image = ImageLoader.cv2_loader(os.path.join(inference_folder, file))
    for idx in range(bbox_after_scoring.shape[0]):
        x1 = int(bbox_after_scoring[idx][0])
        y1 = int(bbox_after_scoring[idx][1])
        x2 = int(bbox_after_scoring[idx][2])
        y2 = int(bbox_after_scoring[idx][3])
        label = int(labels_after_scoring[idx])
        color = colors[label - 1]
        color = (int(color[0]), int(color[1]), int(color[2]))
        imgHeight, imgWidth, _ = image.shape
        thick = int((imgHeight + imgWidth) // 900)
        cv2.rectangle(image, (x1, y1), (x2, y2), tuple(color), thick)
        cv2.putText(image, ids_to_cats[label], (x1, y1 - 12), 0, 1e-3 * imgHeight, tuple(color), thick // 3)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

if __name__ == '__main__':
    args = get_args(inference=True, classification=False)
    with open(args.config_path) as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    with open(params['class_idx_dict'], 'rb') as f:
        class_dict = pickle.load(f)
    colors = [np.random.randint(0, 255, size=(3,)) for _ in range(len(class_dict))]
    inference_folder = args.folder
    output_folder = args.output_folder
    weights_file = args.weights
    thr = args.threshold
    model = FasterRCNN(params=params)
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
            outs = predict(inference_folder, file, model, augs)[0]
            cv2.imwrite(os.path.join(output_folder, f"{file}_boxes.png"), draw_box(inference_folder, file, outs, thr, class_dict, colors))