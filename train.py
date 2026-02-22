import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

from dataset import get_data_loaders
from model import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # ===== WandB (with resume enabled) =====
    wandb.init(
        project="rice-convit-final",
        resume="allow",
        config={
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "model": "convit_tiny"
        }
    )

    config = wandb.config

    # ===== Data =====
    train_loader, val_loader, num_classes = get_data_loaders(
        "data",
        batch_size=config.batch_size
    )

    # ===== Model =====
    model = get_model(num_classes).to(device)

    # ===== Faster Class Weight Computation (NO IMAGE LOADING) =====
    print("Computing class weights (fast mode)...")
    targets = train_loader.dataset.targets
    class_counts = torch.bincount(torch.tensor(targets))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum()
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    # ===== Checkpoint Setup =====
    checkpoint_path = "/workspace/checkpoint.pth"
    start_epoch = 0
    best_acc = 0.0
    patience = 8
    trigger_times = 0

    # ===== Resume If Checkpoint Exists =====
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if use_amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        trigger_times = checkpoint['trigger_times']

        print(f"Resuming from epoch {start_epoch}")

    # ===== Training Loop =====
    for epoch in range(start_epoch, config.epochs):

        print(f"\nEpoch {epoch+1}/{config.epochs}")

        model.train()
        total_loss = 0

        for images, labels in tqdm(train_loader):

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        scheduler.step(val_acc)

        # ===== Early Stopping + Best Model Save =====
        if val_acc > best_acc:
            best_acc = val_acc
            trigger_times = 0
            torch.save(model.state_dict(), "/workspace/best_model.pth")
            print("Best model saved!")
        else:
            trigger_times += 1
            print("Early stopping counter:", trigger_times)

            if trigger_times >= patience:
                print("Early stopping triggered!")
                break

        # ===== Save Checkpoint Every Epoch =====
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if use_amp else None,
            'best_acc': best_acc,
            'trigger_times': trigger_times
        }, checkpoint_path)

        wandb.log({
            "train_loss": avg_loss,
            "val_accuracy": val_acc,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

        print(f"Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()