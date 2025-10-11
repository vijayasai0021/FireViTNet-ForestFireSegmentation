# In utils/segformer_dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegFormerFireDataset(Dataset):
    """
    A specialized dataset class that formats data for the Hugging Face
    SegFormer model and Trainer.
    """
    def __init__(self, data_dir, input_size=(224, 224), augment=False):
        self.data_dir = Path(data_dir)
        self.input_size = input_size
        
        # Search for all common image types
        image_extensions = ['*.jpg', '*.jpeg', '*.png']
        self.image_paths = []
        for ext in image_extensions:
            self.image_paths.extend(list((self.data_dir / 'images').glob(ext)))
        self.image_paths = sorted(self.image_paths)
        
        self.mask_paths = sorted(list((self.data_dir / 'masks').glob('*.png')))
        
        assert len(self.image_paths) == len(self.mask_paths), \
            f"Mismatch: Found {len(self.image_paths)} images and {len(self.mask_paths)} masks"
        
        # Define transforms
# In utils/segformer_dataset.py

# ... (keep the imports and the class definition the same) ...

        # Define transforms
        if augment:
            # THIS IS THE NEW, MORE POWERFUL AUGMENTATION PIPELINE
            self.transform = A.Compose([
                A.Resize(*input_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3, brightness_limit=0.2, contrast_limit=0.2),
                A.CoarseDropout(max_holes=1, max_height=50, max_width=50, min_holes=1, min_height=50, min_width=50, fill_value=255, p=0.2),
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

# ... (the rest of the file stays the same) ...
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_name = Path(image_path).stem + '.png'
        mask_path = self.data_dir / 'masks' / mask_name

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Convert mask to binary (0 or 1)
        mask = (mask > 0).float()
        
        # --- THIS IS THE KEY CHANGE ---
        # The Trainer expects a dictionary with specific keys
        return {'pixel_values': image, 'labels': mask}
