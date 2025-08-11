import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            images, masks = batch
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(test_loader)
