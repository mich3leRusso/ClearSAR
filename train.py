import torch 

def evaluate(model, data_loader, device):
    """Runs a validation pass. FasterRCNN in train mode returns losses, eval mode returns predictions."""
    # Temporarily set to train mode to get losses, but freeze BN/Dropout
    model.train()
    model.encoder.model.eval()

    total_loss = 0.0
    with torch.no_grad():
        for i, (images, targets, _, og_img_size) in enumerate(data_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model.detector(images, targets)
            total_loss += sum(loss for loss in loss_dict.values()).item()

    return total_loss / len(data_loader)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    model.encoder.model.eval()
    total_loss = 0.0
    for i, (images, targets, _, og_img_size) in enumerate(data_loader):
        # Move images and targets to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # FasterRCNN in train mode returns a dict of losses
        loss_dict = model.detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping to stabilize ViT fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses.item()

        print(
            f"Epoch [{epoch}] Iter [{i+1}/{len(data_loader)}] "
            f"Loss: {losses.item():.4f} | "
            + " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
        )

    return total_loss / len(data_loader)
