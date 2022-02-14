import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DigitsDataset(Dataset):
    def __init__(self, data, transform=None):
        self.images = data['images']
        self.labels = data['labels']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = Image.fromarray(image.astype(np.uint8))
            image = self.transform(image)
            image = np.array(image).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1) / 255
        label = torch.tensor(label).long()

        return image, label
    