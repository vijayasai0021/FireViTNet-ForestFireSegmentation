# predict_efficientnet.py

import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp # Import smp
import os

# --- 1. Add project path ---
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
sys.path.append('/content/FireViTNet-ForestFireSegmentation/FireViTNet-ForestFireSegmentation') 

# --- 2. Configuration ---
# --- IMPORTANT: UPDATE THESE PATHS ---
MODEL_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels"
IMAGE_PATH = "/content/path/to/your/test_image.jpg"  # <-- Change this to your image
INPUT_SIZE = (224, 224)

def predict_single_image(model, image_path, device):
    """Loads an image, preprocesses it, and returns the model's prediction."""
    # Define transformations (same as validation/test set)
    transform = A.Compose([
        A.Resize(*INPUT_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Keep original for display
    
    processed_image_dict = transform(image=original_image)
    processed_image = processed_image_dict['image']
    
    # Add batch dimension and send to device
    image_tensor = processed_image.unsqueeze(0).to(device)

    # --- Make Prediction ---
    model.eval() 
    with torch.no_grad():
        output_logits = model(image_tensor)
        # Apply sigmoid and threshold
        pred_mask = (torch.sigmoid(output_logits) > 0.5).squeeze().cpu().numpy()

    # Return original image (resized for consistency) and the predicted mask
    display_image = cv2.resize(original_image, INPUT_SIZE) 
    return display_image, pred_mask.squeeze()


if __name__ == '__main__':
    # Install smp if not already done in the session
    os.system("pip install segmentation-models-pytorch -q")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load EfficientNet-Unet Model ---
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None, 
        in_channels=3,
        classes=1,
    ).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("EfficientNet model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please check the path.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    # --- Predict on a single image ---
    try:
        display_image, predicted_mask = predict_single_image(model, IMAGE_PATH, device)
    except FileNotFoundError as e:
        print(e)
        exit()

    # --- Visualize the Results ---
    # Create overlay
    overlay_image = display_image.copy()
    # Convert mask to uint8 for findContours
    binary_mask_uint8 = (predicted_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours (green, thickness 2)
    cv2.drawContours(overlay_image, contours, -1, (0, 255, 0), 2) 

    # Display 
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1); plt.imshow(display_image); plt.title("Original Image (Resized)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(predicted_mask, cmap='gray'); plt.title("Predicted Mask"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(overlay_image); plt.title("Prediction Overlay"); plt.axis('off')
    plt.tight_layout()
    plt.show()
