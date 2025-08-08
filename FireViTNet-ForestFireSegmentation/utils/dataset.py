import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FireDataset(Dataset):
    def __init__(self, data_dir, input_size=(224, 224), augment=False):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        self.augment = augment
        
        # Get image and mask paths
        self.image_paths = sorted(list((self.data_dir / 'images').glob('*.jpg')))
        self.mask_paths = sorted(list((self.data_dir / 'masks').glob('*.png')))
        
        assert len(self.image_paths) == len(self.mask_paths), "Number of images and masks must match"
        
        # Define transforms
        if augment:
            self.transform = A.Compose([
                A.Resize(*input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
                A.GaussNoise(p=0.2),
                A.Blur(p=0.2),
                A.RandomGamma(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*input_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert mask to float and normalize to [0, 1]
        mask = mask.float() / 255.0
        mask = mask.unsqueeze(0)  # Add channel dimension
        
        return image, mask
