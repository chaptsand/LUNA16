import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dsets import LunaDataset
from model import LunaModel
from logconf import compute_metrics
import datetime
import numpy as np

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 10
NUM_WORKERS = 8

DATA_DIR = "./datasets/LUNA16"
CSV_PATH = "./datasets/LUNA16/candidates_V2.csv"
CACHE_DIR = "./LUNA16/cache"

def training_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"./runs/luna_{time_str}")

    print("⏳ Loading Dataset...")
    train_ds = LunaDataset(data_dir=DATA_DIR,
                           csv_path=CSV_PATH,
                           is_val_set=False,
                           val_stride=10,
                           cache_dir=CACHE_DIR,
                           augmentation=True)

    val_ds = LunaDataset(data_dir=DATA_DIR,
                         csv_path=CSV_PATH,
                         is_val_set=True,
                         val_stride=10,
                         cache_dir=CACHE_DIR,
                         augmentation=False)

    train_loader = DataLoader(train_ds,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_ds,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=False,
                            pin_memory=True)

    model = LunaModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    class_weights = torch.tensor([1.0, 8.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"✅ Training Start (TensorBoard logs enabled)")
    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{EPOCHS} ---")

        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device).long()

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            global_step = (epoch - 1) * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)

        model.eval()
        val_loss = 0.0

        conf_mat = np.zeros((2,2))

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    conf_mat[int(t), int(p)] += 1

        avg_val_loss = val_loss / len(val_loader)
        acc, prec, recall, f1 = compute_metrics(conf_mat)

        print(f"\tTrain Loss: {avg_train_loss:.4f}")
        print(f"\tVal Loss: {avg_val_loss:.4f}")
        print(f"\tVal Acc: {acc*100:.2f}%")
        print(f"\tRecall: {recall*100:.2f}%")
        print(f"\tPrecision: {prec*100:.2f}%")
        print(f"\tF1 Score: {f1:.4f}")
        print(f"\tConfusion Matrix: TP={conf_mat[1,1]}, FN={conf_mat[1,0]}")

        writer.add_scalar('Loss/Val', avg_val_loss, epoch)
        writer.add_scalar('Metrics/Accuracy', acc, epoch)
        writer.add_scalar('Metrics/Recall', recall, epoch)
        writer.add_scalar('Metrics/Precision', prec, epoch)
        writer.add_scalar('Metrics/F1', f1, epoch)

    writer.close()
    print("✅ Training Over")

if __name__ == '__main__':
    training_loop()
