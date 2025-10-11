# In training/evaluate_segformer.py

import torch
from torch.utils.data import DataLoader, random_split
from transformers import SegformerForSemanticSegmentation
import numpy as np

# This is crucial for importing your custom dataset from the utils folder
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
sys.path.append('/content/FireViTNet-ForestFireSegmentation/FireViTNet-ForestFireSegmentation') 
from utils.segformer_dataset import SegFormerFireDataset

# --- Configuration ---
DATA_DIR = "/content/Processed_Dataset"
# --- IMPORTANT: UPDATE THIS PATH ---
MODEL_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels" 
BATCH_SIZE = 4
INPUT_SIZE = (224, 224)

def evaluate_segformer():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Test Dataset ---
    full_dataset = SegFormerFireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=False)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    _, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set size: {len(test_dataset)}")

    # --- Load the Trained SegFormer Model ---
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    # --- Calculate Metrics ---
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch['pixel_values'].to(device)
            masks = batch['labels'].to(device)
            
            # Get model predictions
            outputs = model(pixel_values=pixel_values).logits
            
            # Upsample logits to match mask size and apply sigmoid
            upsampled_logits = torch.nn.functional.interpolate(
                outputs,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            preds = (torch.sigmoid(upsampled_logits) > 0.5).float()
            
            # Calculate metrics
            total_tp += ((preds == 1) & (masks == 1)).sum().item()
            total_fp += ((preds == 1) & (masks == 0)).sum().item()
            total_fn += ((preds == 0) & (masks == 1)).sum().item()

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    print("\n--- SegFormer Evaluation Metrics ---")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

if __name__ == '__main__':
    evaluate_segformer()
