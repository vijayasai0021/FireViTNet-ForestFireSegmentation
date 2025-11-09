# In training/fuse_and_evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
import segmentation_models_pytorch as smp
import os
import sys

# --- 1. Add project path ---
# --- IMPORTANT: UPDATE THIS PATH ---
sys.path.append('/content/drive/MyDrive/path/to/your/FireViTNet-ForestFireSegmentation') 
from models.firevitnet import FireViTNet
from utils.dataset import FireDataset 

# --- 2. Configuration ---
DATA_DIR = "/content/Processed_Dataset"
# --- IMPORTANT: UPDATE THESE PATHS ---
FIREVITNET_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels/best_firevitnet_model.pth"
EFFICIENTNET_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels/best_efficientnet_model.pth"

BATCH_SIZE = 8
INPUT_SIZE = (224, 224)

# --- 3. Optimal Thresholds (from our tests) ---
VIT_THRESH = 0.30  # The best threshold for FireViTNet
EFF_THRESH = 0.50  # The best threshold for EfficientNet (default)

# --- 4. Fusion Weights ---
# We give more weight to EfficientNet since it scored higher
W_VIT = 0.3  # 30% weight for FireViTNet
W_EFF = 0.7  # 70% weight for EfficientNet
FUSED_THRESH = 0.5 # Threshold for the combined prediction

def calculate_metrics(preds, masks):
    """Helper function to calculate metrics for a batch."""
    tp = ((preds == 1) & (masks == 1)).sum().item()
    fp = ((preds == 1) & (masks == 0)).sum().item()
    fn = ((preds == 0) & (masks == 1)).sum().item()
    return tp, fp, fn

def print_final_scores(model_name, metrics_dict):
    """Helper function to calculate and print final scores."""
    tp, fp, fn = metrics_dict['tp'], metrics_dict['fp'], metrics_dict['fn']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    print(f"\n--- Results for: {model_name.upper()} ---")
    print(f"  IoU: {iou:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

def run_fusion_evaluation():
    os.system("pip install segmentation-models-pytorch -q")
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

    # --- Load Models ---
    firevit_model = FireViTNet(num_classes=1).to(device)
    firevit_model.load_state_dict(torch.load(FIREVITNET_PATH, map_location=device))
    firevit_model.eval()
    print("FireViTNet model loaded.")

    efficientnet_model = smp.Unet(
        encoder_name="efficientnet-b4", encoder_weights=None, in_channels=3, classes=1
    ).to(device)
    efficientnet_model.load_state_dict(torch.load(EFFICIENTNET_PATH, map_location=device))
    efficientnet_model.eval()
    print("EfficientNet-Unet model loaded.")

    metrics = {
        'firevit': {'tp': 0, 'fp': 0, 'fn': 0},
        'efficientnet': {'tp': 0, 'fp': 0, 'fn': 0},
        'fused_ensemble': {'tp': 0, 'fp': 0, 'fn': 0}
    }

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # 1. FireViTNet Prediction
            firevit_logits = firevit_model(images)
            firevit_probs = torch.sigmoid(firevit_logits)
            # --- USE OPTIMAL THRESHOLD ---
            firevit_preds = (firevit_probs > VIT_THRESH).float() 
            tp, fp, fn = calculate_metrics(firevit_preds, masks)
            metrics['firevit']['tp'] += tp
            metrics['firevit']['fp'] += fp
            metrics['firevit']['fn'] += fn

            # 2. EfficientNet Prediction
            efficientnet_logits = efficientnet_model(images)
            efficientnet_probs = torch.sigmoid(efficientnet_logits)
            # --- USE OPTIMAL THRESHOLD ---
            efficientnet_preds = (efficientnet_probs > EFF_THRESH).float() 
            tp, fp, fn = calculate_metrics(efficientnet_preds, masks)
            metrics['efficientnet']['tp'] += tp
            metrics['efficientnet']['fp'] += fp
            metrics['efficientnet']['fn'] += fn

            # 3. Fused Prediction
            fused_probs = (W_VIT * firevit_probs) + (W_EFF * efficientnet_probs)
            fused_preds = (fused_probs > FUSED_THRESH).float() 
            tp, fp, fn = calculate_metrics(fused_preds, masks)
            metrics['fused_ensemble']['tp'] += tp
            metrics['fused_ensemble']['fp'] += fp
            metrics['fused_ensemble']['fn'] += fn

    # --- Print Final Results Table ---
    print("\n" + "="*40)
    print("  FINAL PROJECT PERFORMANCE COMPARISON")
    print("="*40)
    
    for model_name, values in metrics.items():
        print_final_scores(model_name, values)
        
    print("="*40)
    print("Project 100% Complete.")

if __name__ == '__main__':
    run_fusion_evaluation()
