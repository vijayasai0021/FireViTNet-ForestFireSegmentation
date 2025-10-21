# In training/train_efficientnet.py

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import timm # We use timm to get EfficientNet

# This is crucial for importing your custom dataset from the utils folder
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
# This must be the path to your main project folder in Google Drive
sys.path.append('/content/FireViTNet-ForestFireSegmentation/FireViTNet-ForestFireSegmentation') 
from utils.dataset import FireDataset

# --- Configuration ---
DATA_DIR = "/content/Processed_Dataset"
# --- IMPORTANT: UPDATE THIS PATH ---
# This is where your new EfficientNet model will be saved
MODEL_SAVE_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels"
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 100 
INPUT_SIZE = (224, 224)

# --- Dice Loss Definition ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice_score

# --- Main Training Function ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load Dataset ---
    full_dataset = FireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # --- Load EfficientNet Model ---
    # This is the key change: we load 'efficientnet_b4'
    # num_classes=1 for binary segmentation (fire/no-fire)
    model = timm.create_model(
        'efficientnet_b4', 
        pretrained=True, 
        in_chans=3, 
        num_classes=1
    ).to(device)
    
    print("EfficientNet-B4 model loaded successfully.")

    # --- Setup Loss and Optimizer ---
    loss_fn = DiceLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()
                
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_file = os.path.join(MODEL_SAVE_PATH, 'best_efficientnet_model.pth')
            torch.save(model.state_dict(), model_save_file)
            print(f"Model saved to {model_save_file}")

    print("--- EfficientNet Training Complete ---")

if __name__ == '__main__':
    train_model()
