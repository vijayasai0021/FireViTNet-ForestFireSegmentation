import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import os

# --- 1. Import your custom classes ---
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
sys.path.append('/content/FireViTNet-ForestFireSegmentation/FireViTNet-ForestFireSegmentation')
from models.firevitnet import FireViTNet
from utils.dataset import FireDataset

# --- 2. FOCAL LOSS IMPLEMENTATION ---
# This loss is designed for extreme class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', smooth=1.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Calculate pt (probability of the correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Calculate focal loss modulating factor
        focal_weight = (1 - p_t).pow(self.gamma)
        
        # Calculate alpha-weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Final focal loss
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --- 3. Hyperparameters and Setup ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
EPOCHS = 100
DATA_DIR = "/content/Processed_Dataset" # IMPORTANT: Change this path
# --- IMPORTANT: UPDATE THIS PATH ---
MODEL_SAVE_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels"
INPUT_SIZE = (224, 224)

def train_model():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Load and Split Dataset ---
    full_dataset = FireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=True)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- 5. Initialize Model, Loss, and Optimizer ---
    model = FireViTNet(num_classes=1).to(device)
    
    # --- THIS IS THE KEY CHANGE ---
    criterion = FocalLoss().to(device) 
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 6. Training and Validation Loop ---
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(MODEL_SAVE_PATH, 'best_firevitnet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training finished!")

if __name__ == '__main__':
    train_model()
