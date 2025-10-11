# In training/train_segformer.py

# Step 1: Install necessary libraries (this line will be run by the script)
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
from utils.segformer_dataset import FireDataset

# --- Configuration ---
DATA_DIR = "/content/Processed_Dataset"
# We use a SegFormer model pre-trained on a large segmentation dataset (ADE20K)
MODEL_CHECKPOINT = "nvidia/segformer-b0-finetuned-ade-512-512" 
# --- IMPORTANT: UPDATE THIS PATH ---
# This is where your new, trained SegFormer model will be saved permanently
MODEL_SAVE_PATH = "/content/drive/MyDrive/ForestFire-TrainedModels"

def train_segformer():
    # --- Load Dataset ---
    # We use augment=True to make the training more robust
   full_dataset = SegFormerFireDataset(data_dir=DATA_DIR, input_size=(224, 224), augment=True)
    
    # Split into 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- Load Pre-trained SegFormer Model ---
    model = SegformerForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=1, # We only have one class: "fire"
        ignore_mismatched_sizes=True # This is required to fine-tune on a new dataset
    )

    # --- Training Configuration ---
    # These settings control the training process
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=6e-5, # A good starting learning rate for SegFormer
        num_train_epochs=25, # 25 epochs is enough to get a strong result for your review
        per_device_train_batch_size=4, # Increase if you have enough GPU memory
        per_device_eval_batch_size=4,
        save_total_limit=2,
        eval_strategy="steps", # Evaluate during training
        eval_steps=200, # Evaluate every 200 steps
        save_steps=200,
        logging_steps=50,
        load_best_model_at_end=True, # The Trainer will keep the best model
        metric_for_best_model="eval_loss",
    )

    # The Hugging Face Trainer handles the entire training loop for us
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # --- Start Training ---
    print("\n--- Starting SegFormer Fine-Tuning ---")
    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH) # Save the final best model
    print("--- SegFormer Training Complete ---")

if __name__ == '__main__':
    train_segformer()
