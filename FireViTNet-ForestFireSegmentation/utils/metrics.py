import torch
import torch.nn.functional as F
import numpy as np

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    Calculate IoU, Precision, Recall, and F1 score for binary segmentation
    
    Args:
        predictions: Model predictions (B, 1, H, W)
        targets: Ground truth masks (B, 1, H, W)
        threshold: Threshold for binarizing predictions
    
    Returns:
        dict: Dictionary containing all metrics
    """
    with torch.no_grad():
        # Apply sigmoid and threshold
        predictions = torch.sigmoid(predictions)
        pred_binary = (predictions > threshold).float()
        
        # Flatten tensors
        pred_flat = pred_binary.view(-1)
        target_flat = targets.view(-1)
        
        # Calculate TP, FP, FN, TN
        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()
        tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()
        
        # Calculate metrics
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # IoU calculation
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + 1e-8)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'iou': iou,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }

def dice_coefficient(predictions, targets, smooth=1e-8):
    """Calculate Dice coefficient"""
    predictions = torch.sigmoid(predictions)
    
    # Flatten tensors
    pred_flat = predictions.view(-1)
    target_flat = targets.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice.item()
