import pytorch_lightning as pl

def train_model(model, train_loader, val_loader, config):
    trainer = pl.Trainer(
        max_epochs=config['training']['epochs'],
        gpus=1 if torch.cuda.is_available() else 0,
        progress_bar_refresh_rate=20,
    )
    trainer.fit(model, train_loader, val_loader)
