# In training/find_firevit_threshold.py

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import sys
import os

# --- 1. Add project path ---
# --- IMPORTANT: UPDATE THIS PATH ---
sys.path.append('/content/drive/MyDrive/path/to/your/FireViTNet-ForestFireSegmentation') 
from models.firevitnet import FireViTNet
from utils.dataset import FireDataset 

# --- 2. Configuration ---
DATA_DIR = "/content/Processed_Dataset"
# --- IMPORTANT: UPDATE THIS PATH ---
MODEL_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels/best_firevitnet_model.pth"
BATCH_SIZE = 8
INPUT_SIZE = (224, 224)

def find_best_threshold():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Test Dataset ---
    full_dataset = FireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set size: {len(test_dataset)}")

    # --- Load Model ---
    model = FireViTNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("FireViTNet model loaded.")

    # --- Get all predictions and masks ---
    all_probs = []
    all_masks = []
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs) # Get probabilities
            
            all_probs.append(probs.cpu())
            all_masks.append(masks.cpu())

    # Concatenate all batches into single tensors
    all_probs = torch.cat(all_probs).view(-1)
    all_masks = torch.cat(all_masks).view(-1)

    # --- Loop through thresholds to find the best one ---
    best_f1 = 0.0
    best_threshold = 0.0
    
    # Test thresholds from 0.1 to 0.9
    for threshold in np.arange(0.1, 0.9, 0.05):
        preds = (all_probs > threshold).float()
        
        tp = ((preds == 1) & (all_masks == 1)).sum().item()
        fp = ((preds == 1) & (all_masks == 0)).sum().item()
        fn = ((preds == 0) & (all_masks == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Testing Threshold: {threshold:.2f} -> F1-Score: {f1_score:.4f}")
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_threshold = threshold

    print("\n" + "="*30)
    print(f"Optimal Threshold found: {best_threshold:.2f}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print("="*30)

if __name__ == '__main__':
    find_best_threshold()
