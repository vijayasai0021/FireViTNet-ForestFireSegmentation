import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from models.firevitnet import FireViTNet

class FireViTNetLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = FireViTNet()
        self.learning_rate = config['training']['learning_rate']
        # You can load additional hyperparams from config

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, masks)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = F.binary_cross_entropy_with_logits(outputs, masks)
        self.log('val_loss', loss, prog_bar=True)
        # You can add metrics here (IoU, Dice, etc.)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # Optional: Add scheduler if needed
        return optimizer
