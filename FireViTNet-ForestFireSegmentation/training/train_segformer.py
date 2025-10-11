# In training/train_segformer.py

import matplotlib.pyplot as plt
import os
os.system("pip install transformers accelerate evaluate -q")

import torch
from torch.utils.data import DataLoader, random_split
from transformers import SegformerForSemanticSegmentation, Trainer, TrainingArguments
import numpy as np

# This is crucial for importing your custom dataset from the utils folder
import sys
# --- IMPORTANT: UPDATE THIS PATH ---
# This must be the path to your main project folder in Google Drive
sys.path.append('/content/FireViTNet-ForestFireSegmentation/FireViTNet-ForestFireSegmentation') 
from utils.segformer_dataset import SegFormerFireDataset

# --- Configuration ---
DATA_DIR = "/content/Processed_Dataset"
MODEL_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512" 
# --- IMPORTANT: UPDATE THIS PATH ---
# This is where your new, trained SegFormer model will be saved permanently
MODEL_SAVE_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels"

# --- Custom Dice Loss and Trainer ---
class DiceLoss(torch.nn.Module):
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

class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fct = DiceLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = self.loss_fct(upsampled_logits, labels)
        return (loss, outputs) if return_outputs else loss

def train_segformer():
    # --- Load Dataset ---
    full_dataset = SegFormerFireDataset(data_dir=DATA_DIR, input_size=(224, 224), augment=True)
    
    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # --- Data Sanity Check ---
    print("--- Performing a quick data sanity check ---")
    sample = train_dataset[0]
    image = sample['pixel_values'].permute(1, 2, 0).numpy()
    mask = sample['labels'].numpy()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1); plt.imshow(image); plt.title("Sample Image (Augmented)"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(mask, cmap='gray'); plt.title("Sample Mask"); plt.axis('off')
    plt.show()
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- Load Pre-trained SegFormer Model ---
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT, num_labels=1, ignore_mismatched_sizes=True
    )

    # --- Training Configuration ---
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=1e-5,  # Lower learning rate for stability
        num_train_epochs=50,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Use our CustomTrainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # --- Start Training ---
    print("\n--- Starting NEW SegFormer Fine-Tuning with Dice Loss ---")
    trainer.train() # <-- Corrected: No resume_from_checkpoint
    trainer.save_model(MODEL_SAVE_PATH)
    print("--- SegFormer Training Complete ---")

if __name__ == '__main__':
    train_segformer()
