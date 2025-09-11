import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. Import your custom classes ---
from models.firevitnet import FireViTNet
from utils.dataset import FireDataset

# --- 2. Configuration ---
DATA_DIR = "/content/Processed_Dataset"
MODEL_PATH = "./trained_models/best_firevitnet_model.pth"
BATCH_SIZE = 4 # We can use a slightly larger batch size for evaluation
INPUT_SIZE = (224, 224)

def evaluate_model():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 3. Load the Test Dataset ---
    # We need to create the same train/val/test split as in training to get the test set
    full_dataset = FireDataset(data_dir=DATA_DIR, input_size=INPUT_SIZE, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    _, _, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test set size: {len(test_dataset)}")

    # --- 4. Load the Trained Model ---
    model = FireViTNet(num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set the model to evaluation mode (very important!)

    # --- 5. Calculate Metrics ---
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad(): # Disable gradient calculations
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get model predictions
            outputs = model(images)
            
            # Convert predictions to a binary mask (0 or 1)
            preds = (outputs > 0.5).float()
            
            # Calculate metrics for this batch
            total_tp += ((preds == 1) & (masks == 1)).sum().item()
            total_fp += ((preds == 1) & (masks == 0)).sum().item()
            total_fn += ((preds == 0) & (masks == 1)).sum().item()

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    print("\n--- Evaluation Metrics ---")
    print(f"Intersection over Union (IoU): {iou:.4f}")
    print(f"F1-Score: {f1_score:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # --- 6. Visualize Some Predictions ---
    visualize_predictions(model, test_dataset, device)

def visualize_predictions(model, dataset, device, num_samples=5):
    plt.figure(figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image, mask = dataset[idx]
        
        # Get model prediction for a single image
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            pred_mask = (output > 0.5).squeeze(0).cpu().numpy()

        # Prepare images for display
        image_display = image.permute(1, 2, 0).numpy() # Convert from (C, H, W) to (H, W, C)
        true_mask_display = mask.squeeze(0).numpy()
        
        # Display Original Image
        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(image_display)
        plt.title("Original Image")
        plt.axis('off')
        
        # Display Ground Truth Mask
        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(true_mask_display, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.axis('off')
        
        # Display Predicted Mask
        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    evaluate_model()
