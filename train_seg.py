import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dsets_seg import LunaSegDataset
from model_seg import UNet
import datetime
import numpy as np

BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 20
NUM_WORKERS = 6

DATA_DIR = "./datasets/LUNA16"
CSV_PATH = "./datasets/LUNA16/candidates_V2.csv"
CACHE_DIR = "./LUNA16/cache"

class LunaDiceLoss(nn.Module):
    # Dice Loss = 1 - (2 * Intersection) / (Union + epsilon)
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits, targets):
        # logits: (B, 1, D, H, W) - the original model output
        # targets: (B, 1, D, H, W) - the truth Mask (0, 1)

        # 1. Activation function
        probs = torch.sigmoid(logits)

        # 2. Flatten
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        # 3. Intersection
        intersection = (probs_flat * targets_flat).sum()

        # 4. Union
        union = probs_flat.sum() + targets_flat.sum()

        # 5. Dice coefficient
        dice_score = (2.0 * intersection) / (union + self.epsilon)

        # 6. Dice Loss = 1 - Dice Score
        return 1.0 - dice_score

def training_loop():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"./runs/seg_{time_str}")

    print("⏳ Loading Segmentation Dataset...")
    train_ds = LunaSegDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH, is_val_set=False, val_stride=10,
        cache_dir=CACHE_DIR, augmentation=True
    )
    val_ds = LunaSegDataset(
        data_dir=DATA_DIR, csv_path=CSV_PATH, is_val_set=True, val_stride=10,
        cache_dir=CACHE_DIR, augmentation=False
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True)

    model = UNet(in_channels=1, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dice_loss_fn = LunaDiceLoss().to(device)
    # bce_loss_fn = nn.BCEWithLogitsLoss()

    print(f"✅ Training Start (Batch Size: {BATCH_SIZE})")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        # === Training ===
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, masks) in enumerate(train_loader):
            inputs, masks = inputs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(inputs)

            loss = dice_loss_fn(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)

        # === Evaluation ===
        model.eval()
        val_loss = 0.0
        val_dice_score = 0.0

        # Visualize
        vis_input = None
        vis_mask = None
        vis_pred = None

        with torch.no_grad():
            for batch_idx, (inputs, masks) in enumerate(val_loader):
                inputs, masks = inputs.to(device), masks.to(device)

                logits = model(inputs)
                loss = dice_loss_fn(logits, masks)
                val_loss += loss.item()

                # Dice Score = 1 - Loss
                val_dice_score += (1.0 - loss.item())

                if batch_idx == 0:
                    vis_input = inputs[0]
                    vis_mask = masks[0]
                    vis_pred = torch.sigmoid(logits[0])

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_score / len(val_loader)

        print(f"\tTrain Loss: {avg_train_loss:.4f}")
        print(f"\tVal Loss:   {avg_val_loss:.4f}")
        print(f"\tVal Dice:   {avg_val_dice:.4f} (The higher, the better)")

        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metric/Dice', avg_val_dice, epoch)

        # TensorBoard Image Visualize
        ct_slice = vis_input[:, 24, :, :].cpu()
        mask_slice = vis_mask[:, 24, :, :].cpu()
        pred_slice = vis_pred[:, 24, :, :].cpu()

        montage = torch.cat([ct_slice, mask_slice, pred_slice], dim=2)

        writer.add_image(f'Visual/Epoch_{epoch}', montage, epoch)

        torch.save(model.state_dict(), "luna_unet_latest.pth")

    writer.close()
    print(f"✅: Segmentation Training Over")

if __name__ == '__main__':
    training_loop()
