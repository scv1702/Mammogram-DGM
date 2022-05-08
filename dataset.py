import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from config import config
from utils import open_dcm

class MammogramDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx, 0]
        image = open_dcm(img_path)
        if self.transform:
            # reshaping
            image = self.transform(image)
            # convert to [-1, 1]
            image = (image / 127.5) - 1
        if self.metadata_path == config.normal_metadata_path:
            return image
        elif self.metadata_path == config.train_metadata_path:
            normal = self.metadata.iloc[idx, 1]
            return (image, normal)

class CXRDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx, 0]
        image = open_dcm(f"CXR_data/stage_2_train_images/{img_path}.dcm")
        if self.transform:
            # reshaping
            image = self.transform(image)
            # convert to [-1, 1]
            image = (image / 127.5) - 1
        if self.metadata.iloc[idx, 5] == 1:
            return image
        elif self.metadata.iloc[idx, 5] == 0:
            normal = self.metadata.iloc[idx, 1]
            return (image, normal)