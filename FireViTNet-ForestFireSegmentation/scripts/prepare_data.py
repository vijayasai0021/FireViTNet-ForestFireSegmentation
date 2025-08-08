import argparse
import cv2
import numpy as np
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split

def convert_labelme_to_mask(json_path, output_path, image_shape):
    """Convert LabelMe JSON annotation to binary mask"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    for shape in data['shapes']:
        if shape['label'].lower() in ['fire', 'flame']:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    
    cv2.imwrite(str(output_path), mask)
    return mask

def prepare_dataset(data_dir, output_dir, test_size=0.2, val_size=0.1):
    """Prepare dataset by splitting and organizing files"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.png'))
    
    # Split dataset
    train_files, test_files = train_test_split(image_files, test_size=test_size, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=val_size/(1-test_size), random_state=42)
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    for split_name, files in splits.items():
        print(f"Processing {split_name} split: {len(files)} files")
        
        for img_path in files:
            # Copy image
            dst_img = output_dir / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Process annotation if exists
            json_path = img_path.with_suffix('.json')
            if json_path.exists():
                # Load image to get shape
                image = cv2.imread(str(img_path))
                
                # Convert to mask
                mask_path = output_dir / split_name / 'masks' / f"{img_path.stem}.png"
                convert_labelme_to_mask(json_path, mask_path, image.shape)
            else:
                print(f"Warning: No annotation found for {img_path}")
    
    print("Dataset preparation completed!")

def apply_augmentations(input_dir, output_dir, augment_factor=2):
    """Apply data augmentation to increase dataset size"""
    import albumentations as A
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Augmentation pipeline
    augment = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.4, brightness_limit=0.2, contrast_limit=0.2),
        A.GaussNoise(p=0.3, var_limit=(10, 50)),
        A.Blur(p=0.3, blur_limit=3),
    ])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    (output_dir / 'masks').mkdir(exist_ok=True)
    
    image_files = list((input_dir / 'images').glob('*.jpg')) + list((input_dir / 'images').glob('*.png'))
    
    for img_path in image_files:
        # Load image and mask
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = input_dir / 'masks' / f"{img_path.stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Apply augmentations
        for i in range(augment_factor):
            augmented = augment(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
            
            # Save augmented data
            aug_img_path = output_dir / 'images' / f"{img_path.stem}_aug{i}.jpg"
            aug_mask_path = output_dir / 'masks' / f"{img_path.stem}_aug{i}.png"
            
            aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(aug_img_path), aug_image_bgr)
            cv2.imwrite(str(aug_mask_path), aug_mask)
    
    print(f"Augmentation completed! Generated {len(image_files) * augment_factor} additional samples.")

def main():
    parser = argparse.ArgumentParser(description='Prepare FireViTNet dataset')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with images and annotations')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for organized dataset')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.1, help='Validation set size ratio')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--augment_factor', type=int, default=2, help='Augmentation factor')
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_dataset(args.input_dir, args.output_dir, args.test_size, args.val_size)
    
    # Apply augmentation if requested
    if args.augment:
        train_dir = Path(args.output_dir) / 'train'
        aug_dir = Path(args.output_dir) / 'train_augmented'
        apply_augmentations(train_dir, aug_dir, args.augment_factor)
    
    print("Data preparation completed successfully!")

if __name__ == '__main__':
    main()
