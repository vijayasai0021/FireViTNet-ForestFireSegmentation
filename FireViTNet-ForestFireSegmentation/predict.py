import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# --- 1. Import your custom model ---
from models.firevitnet import FireViTNet

# --- 2. Configuration ---
# Update these paths
MODEL_PATH = "./trained_models/best_firevitnet_model.pth"
IMAGE_PATH = "./content/forest-fire-fire-smoke-conservation-51951_example.jpeg"  # <-- IMPORTANT: Change this to your image
INPUT_SIZE = (224, 224)

def predict_single_image(model, image_path, device):
    """
    Loads an image, preprocesses it, and returns the model's prediction.
    """
    # Define the same transformations as your validation/test set
    transform = A.Compose([
        A.Resize(*INPUT_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply transformations
    processed_image = transform(image=image)['image']
    
    # Add a batch dimension and send to device
    image_tensor = processed_image.unsqueeze(0).to(device)

    # --- Make Prediction ---
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        output = model(image_tensor)
        # Convert output to a binary mask
        pred_mask = (output > 0.5).squeeze(0).cpu().numpy()

    return image, pred_mask.squeeze()

def de_normalize(tensor):
    """
    De-normalizes a tensor image for display.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.clone().permute(1, 2, 0).cpu().numpy() # C, H, W -> H, W, C
    tensor = tensor * std + mean
    tensor = np.clip(tensor, 0, 1)
    return tensor

if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    model = FireViTNet(num_classes=1).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{MODEL_PATH}'. Please check the path.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        exit()

    # --- Predict on a single image ---
    try:
        original_image, predicted_mask = predict_single_image(model, IMAGE_PATH, device)
    except FileNotFoundError:
        print(f"Error: Input image not found at '{IMAGE_PATH}'. Please check the path.")
        exit()

    # --- Visualize the Results ---
    # Create an overlay of the mask on the original image
    overlay_image = original_image.copy()
    # Convert mask to the right format to find contours
    binary_mask = (predicted_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw contours on the overlay image in a bright color (e.g., green)
    cv2.drawContours(overlay_image, contours, -1, (0, 255, 0), 2) # Draw in green with thickness 2

    # Display the three images
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_image)
    plt.title("Prediction Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
