import os

from torch.utils.data import Dataset
import torch
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd
from PIL import Image


class PainDatasets(Dataset):

    def __init__(self, img_dir, label_path, transform=None, target_transform=None):
        super(PainDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()


    def __getitem__(self, i):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
        image = self.convtensor(Image.open(img_path))

        label = self.img_labels.iloc[i, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.img_labels)
    
#################################################################################################
#################################################################################################
class PainImgBioDatasets(Dataset):

    def __init__(self, img_dir,label_path, transform=None, target_transform=None):
        super(PainImgBioDatasets, self).__init__()

        self.img_labels = pd.read_csv(label_path, sep=" ")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.convtensor = transforms.ToTensor()


    def __getitem__(self, i):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[i, 0])
        image = self.convtensor(Image.open(img_path))

        bio_path = os.path.join(self.img_dir,self.img_labels.iloc[i, 2])
        phy_signals = pd.read_csv(bio_path, sep='\t')
        signal = phy_signals["gsr"] #Only use the EDA modality
        signal = torch.tensor(signal.values)

        label = self.img_labels.iloc[i, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, signal

    def __len__(self):
        return len(self.img_labels)
