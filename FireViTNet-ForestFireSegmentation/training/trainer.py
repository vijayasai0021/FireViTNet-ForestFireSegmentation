import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import os

# --- 1. Import your custom classes ---
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
sys.path.append('/content/FireViTNet-ForestFireSegmentation/FireViTNet-ForestFireSegmentation')
from models.firevitnet import FireViTNet
from utils.dataset import FireDataset

# --- 2. Loss Implementations ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

# --- THIS IS THE NEW COMBINED LOSS ---
class CombinedLoss(nn.Module):
    def __init__(self, smooth=1.0, bce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss() # This loss is more stable
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        # Calculate BCE loss
        bce = self.bce_loss(logits, targets)
        # Calculate Dice loss
        dice = self.dice_loss(logits, targets)
        # Combine them
        return (self.bce_weight * bce) + ((1 - self.bce_weight) * dice)
# -------------------------------------

# --- 3. Hyperparameters and Setup ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 2
EPOCHS = 100
DATA_DIR = "/content/Processed_Dataset" # IMPORTANT: Change this path
# --- IMPORTANT: UPDATE THIS PATH ---
MODEL_SAVE_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels"
INPUT_SIZE = (224, 224)

def train_model():
    # Create directory to save models
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Load and Split Dataset ---
    full_dataset = FireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=True)
    
    # Split dataset into 80% train, 10% val, 10% test
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- 5. Initialize Model, Loss, and Optimizer ---
    model = FireViTNet(num_classes=1).to(device)
    
    # --- THIS IS THE KEY CHANGE ---
    criterion = CombinedLoss().to(device) 
    
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 6. Training and Validation Loop ---
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # -- Training Phase --
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

        # -- Validation Phase --
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

        # Save the model if validation loss has decreased
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Make sure to save to the correct full path
            save_path = os.path.join(MODEL_SAVE_PATH, 'best_firevitnet_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    print("Training finished!")

if __name__ == '__main__':
    train_model()
