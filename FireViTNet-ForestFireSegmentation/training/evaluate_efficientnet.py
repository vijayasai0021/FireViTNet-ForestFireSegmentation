# In training/evaluate_efficientnet.py

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import segmentation_models_pytorch as smp # Import the smp library

# This is crucial for importing your custom dataset from the utils folder
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
# Path to your main project folder
sys.path.append('/content/drive/MyDrive/path/to/your/FireViTNet-ForestFireSegmentation')
from utils.dataset import FireDataset # Use the original dataset

# --- Configuration ---
DATA_DIR = "/content/Processed_Dataset"
# --- IMPORTANT: UPDATE THIS PATH ---
# Path to your trained EfficientNet model file
MODEL_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels/best_efficientnet_model.pth"
BATCH_SIZE = 8 # Can use a slightly larger batch size for evaluation
INPUT_SIZE = (224, 224)

def evaluate_efficientnet():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Test Dataset ---
    # Use the original FireDataset
    full_dataset = FireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=False)
    # Split to get the same test set as before
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset)) # Keep 10% for validation split consistency
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set size: {len(test_dataset)}")

    # --- Load the Trained EfficientNet Model ---
    # Load the U-Net model with the EfficientNet backbone
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None, # Don't need pretrained weights, we are loading our trained ones
        in_channels=3,
        classes=1,
    ).to(device)
    # Load the saved weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("EfficientNet model loaded.")

    # --- Calculate Metrics ---
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)

            # Get model predictions
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # Calculate metrics
            total_tp += ((preds == 1) & (masks == 1)).sum().item()
            total_fp += ((preds == 1) & (masks == 0)).sum().item()
            total_fn += ((preds == 0) & (masks == 1)).sum().item()

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    print("\n--- EfficientNet Evaluation Metrics ---")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == '__main__':
    evaluate_efficientnet()
