import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, char2idx, target_size=(128, 32), crop_top_ratio=0.3):
        self.image_paths = image_paths
        self.labels = labels
        self.char2idx = char2idx
        self.target_size = target_size
        self.crop_top_ratio = crop_top_ratio

    def __len__(self):
        return len(self.image_paths)

    def encode_label(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        height, _ = img.shape
        crop_end = max(1, int(height * self.crop_top_ratio))
        img = img[:crop_end, :]
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        return img

    def __getitem__(self, idx):
        img = self.preprocess(self.image_paths[idx])
        label = self.encode_label(self.labels[idx])

        return torch.from_numpy(img), torch.tensor(label, dtype=torch.long)
