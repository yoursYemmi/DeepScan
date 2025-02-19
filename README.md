# DeepScan
Smart Document Scanner &amp; OCR

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

// Load Dataset & Preprocess
class OCRDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.char_map = {ch: idx for idx, ch in enumerate("abcdefghijklmnopqrstuvwxyz0123456789")}  # Character to index
        self.idx_map = {idx: ch for ch, idx in self.char_map.items()}  # Index to character

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        image = cv2.resize(image, (128, 32))  # Resize to (128,32)
        image = np.expand_dims(image, axis=0) / 255.0  # Normalize and add channel

        label = self.labels[idx]
        label_encoded = [self.char_map[ch] for ch in label if ch in self.char_map]

        if self.transform:
            image = self.transform(image)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label_encoded, dtype=torch.long)

# Example Dataset (Replace with real dataset paths)
image_paths = ["img1.png", "img2.png", "img3.png"]
labels = ["hello", "world", "ocr"]

transform = transforms.Compose([transforms.ToTensor()])
dataset = OCRDataset(image_paths, labels, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

