import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, char2idx):
        self.image_paths = image_paths
        self.labels = labels
        self.char2idx = char2idx

    def __len__(self):
        return len(self.image_paths)

    def encode_label(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 32))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        img = self.preprocess(self.image_paths[idx])
        label = self.encode_label(self.labels[idx])

        return torch.tensor(img), torch.tensor(label)