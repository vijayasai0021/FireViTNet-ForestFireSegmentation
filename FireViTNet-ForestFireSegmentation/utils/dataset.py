# In utils/dataset.py

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
        # ... (the rest of the __init__ method is the same)
        
        # --- THIS IS THE UPDATED AUGMENTATION PART ---
        if augment:
            self.transform = A.Compose([
                A.Resize(*input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
                
                # Use A.Cutout instead of A.CoarseDropout with updated arguments
                A.Cutout(num_holes=1, max_h_size=50, max_w_size=50, fill_value=255, p=0.2),
                
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
        
        # Ensure mask is binary (0 or 1) and has a channel dimension
        mask = (mask > 0).float() # This makes it more robust by thresholding
        mask = mask.unsqueeze(0)
        
        return image, mask
