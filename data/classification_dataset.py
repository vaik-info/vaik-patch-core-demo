import glob
import os
import re
import random

import torchvision
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class ClassificationDataset(Dataset):
    def __init__(self, input_dir_path, steps=1000, transform=None):
        self.steps = steps
        self.transform = transform
        self.image_dict, self.classes = self.prepare_image_dict(input_dir_path)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        class_label = random.choice(self.classes)
        image_path = random.choice(self.image_dict[class_label])
        image = torchvision.transforms.functional.to_pil_image(read_image(image_path, ImageReadMode.RGB))
        if self.transform:
            image = self.transform(image)
        return image, self.classes.index(class_label)

    @staticmethod
    def prepare_image_dict(input_dir_path):
        image_dict = {}
        dir_path_list = sorted(glob.glob(os.path.join(input_dir_path, '*/')))
        classes = []
        for dir_path in tqdm(dir_path_list, desc=f'_prepare_image_dict'):
            class_label = os.path.basename(os.path.dirname(dir_path))
            classes.append(class_label)
            image_dict[class_label] = []
            image_path_list = [file_path for file_path in glob.glob(os.path.join(dir_path, '**/*.*'), recursive=True) if
                               re.search('\.(png|jpg|bmp)$', file_path)]
            for image_path in image_path_list:
                image_dict[class_label].append(image_path)
        return image_dict, classes
