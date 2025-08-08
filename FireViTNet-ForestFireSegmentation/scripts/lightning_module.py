import torch
import pytorch_lightning as pl
from models.firevitnet import FireViTNet
from utils.loss import CombinedLoss
from utils.metrics import calculate_metrics

class FireViTNetLightning(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Model
        self.model = FireViTNet(num_classes=num_classes)
        
        # Loss function
        self.criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
        
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        
        # For binary segmentation, use only the fire class
        if outputs.shape[1] == 2:
            outputs = outputs[:, 1:2]  # Fire class only
        
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        with torch.no_grad():
            metrics = calculate_metrics(outputs, masks)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_iou', metrics['iou'], on_step=False, on_epoch=True)
        self.log('train_precision', metrics['precision'], on_step=False, on_epoch=True)
        self.log('train_recall', metrics['recall'], on_step=False, on_epoch=True)
        self.log('train_f1', metrics['f1'], on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self.model(images)
        
        # For binary segmentation, use only the fire class
        if outputs.shape[1] == 2:
            outputs = outputs[:, 1:2]  # Fire class only
        
        loss = self.criterion(outputs, masks)
        
        # Calculate metrics
        metrics = calculate_metrics(outputs, masks)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_iou', metrics['iou'], on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', metrics['precision'], on_step=False, on_epoch=True)
        self.log('val_recall', metrics['recall'], on_step=False, on_epoch=True)
        self.log('val_f1', metrics['f1'], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_f1'
            }
        }
