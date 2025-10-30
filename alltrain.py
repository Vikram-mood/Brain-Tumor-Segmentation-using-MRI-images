import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import sys
from io import StringIO
from unet_plus_plus import UNetPlusPlus
from resnet_unet import ResNetUNet
from attention_resnet_unet import AttentionResNetUNet
import UNetPP
import AttentionUNet
import UNet
from utils import resize_input, dice_coef, BraTSDataset

# Redirect print statements to a log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def plot_images(images, true_masks, pred_masks, epoch, model_name, wandb_obj, num_samples=2):
    os.makedirs("plots", exist_ok=True)
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    for i in range(num_samples):
        img = images[i].mean(dim=0).cpu().numpy()
        true = true_masks[i].argmax(dim=0).cpu().numpy()
        pred = pred_masks[i].argmax(dim=0).cpu().numpy()

        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(true, cmap='jet')
        axes[i, 1].set_title("True Mask")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='jet')
        axes[i, 2].set_title("Predicted Mask")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f"plots/{model_name}_epoch_{epoch}.png")
    plt.close(fig)
    try:
        wandb_obj.log({f"{model_name} Segmentation Examples": wandb_obj.Image(f"plots/{model_name}_epoch_{epoch}.png")})
    except:
        pass

def train_model(model, model_name, train_loader, val_loader, num_epochs=5, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_dice = 0.0
    checkpoint_dir = f"checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb as dummy by default
    wandb_obj = type('Dummy', (), {'log': lambda x: None, 'Image': lambda x: None, 'finish': lambda: None})()
    try:
        import wandb
        wandb_obj = wandb
        wandb_obj.init(project=f"{model_name}-brats", config={"epochs": num_epochs, "lr": 1e-4, "batch_size": 4})
    except Exception as e:
        print(f"Wandb failed with error: {str(e)}")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_dice = 0, 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            data = resize_input(data, target_size=(224, 224))
            target = resize_input(target, target_size=(224, 224))

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.argmax(dim=1))
            dice = dice_coef(target, torch.softmax(output, dim=1))

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_dice += dice.item()

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)

        model.eval()
        val_loss, val_dice = 0, 0
        val_images = None
        val_true_masks = None
        val_pred_masks = None
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                data = resize_input(data, target_size=(224, 224))
                target = resize_input(target, target_size=(224, 224))

                output = model(data)
                val_loss += criterion(output, target.argmax(dim=1)).item()
                val_dice += dice_coef(target, torch.softmax(output, dim=1)).item()

                if val_images is None:
                    val_images = data[:2].clone()
                    val_true_masks = target[:2].clone()
                    val_pred_masks = torch.softmax(output, dim=1)[:2].clone()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)

        wandb_obj.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice
        })

        print(f'Epoch {epoch+1}/{num_epochs} ({model_name}):')
        print(f'Training Loss: {train_loss:.4f}, Training Dice: {train_dice:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Dice: {val_dice:.4f}')

        if epoch == 0 or (epoch + 1) % 5 == 0:
            plot_images(val_images, val_true_masks, val_pred_masks, epoch + 1, model_name, wandb_obj)

        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_loss': val_loss
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth'))

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'best_model_{model_name}.pth'))
            print(f"Saved best model with Val Dice: {best_val_dice:.4f}")

    wandb_obj.finish()
    return best_val_dice

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    num_epochs = 10

    # Load paths
    with open("train.txt", "r") as f:
        train_paths = [line.strip() for line in f if line.strip()]
    with open("val.txt", "r") as f:
        val_paths = [line.strip() for line in f if line.strip()]
    with open("test.txt", "r") as f:
        test_paths = [line.strip() for line in f if line.strip()]

    print(f"Number of training images: {len(train_paths)}")
    print(f"Number of validation images: {len(val_paths)}")
    print(f"Number of test images: {len(test_paths)}")

    # Create datasets
    train_dataset = BraTSDataset(train_paths)
    val_dataset = BraTSDataset(val_paths)
    test_dataset = BraTSDataset(test_paths)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Define models
    models = [
        (UNetPlusPlus(in_channels=4, num_classes=3), "unet_plus_plus"),
        (AttentionUNet(in_channels=4, num_classes=3, pretrained=True), "attention_unet"),
        (UNet(in_channels=4, num_classes=3, pretrained=True), "unet")
        # (ResNetUNet(in_channels=4, num_classes=3, pretrained=True), "resnet_unet"),
        # (AttentionResNetUNet(in_channels=4, num_classes=3, pretrained=True), "attention_resnet_unet")
    ]

    # Train each model
    for model, model_name in models:
        model = model.to(device)
        print(f"\nTraining {model_name}...")
        best_val_dice = train_model(model, model_name, train_loader, val_loader, num_epochs=num_epochs, device=device)
        print(f'Best Validation Dice for {model_name}: {best_val_dice:.4f}')

if __name__ == "__main__":
    main()