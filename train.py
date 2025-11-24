import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dsets import LunaDataset
from model import LunaModel

BATCH_SIZE = 256
LR = 1e-4
EPOCHS = 10
NUM_WORKERS = 14

DATA_DIR = "./datasets/LUNA16"
CSV_PATH = "./datasets/LUNA16/candidates_V2.csv"

def training_loop():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\tLoadding train datasets...")
    train_ds = LunaDataset(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        is_val_set=False,
        val_stride=10
    )

    print("\tLoadding valiation datasets...")
    val_ds = LunaDataset(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        is_val_set=True,
        val_stride=10
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = LunaModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    class_weights = torch.tensor([1.0, 8.0]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"Model initial done! Training start...")
    print(f"Loaded {len(train_ds)} train simples, {len(val_ds)} val simples")

    for epoch in range(1, EPOCHS+1):
        print(f"\t--- Epoch {epoch}/{EPOCHS} starting ---")
        
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

            if batch_idx % 10 == 0:
                print(f"\tBatch {batch_idx}/{len(train_loader)} Loss:{loss.item():.4f}")
            
        avg_train_loss = train_loss / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).long()

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
            
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"--- Epoch {epoch} result ---")
        print(f"\tTrain Loss: {avg_train_loss:.4f}")
        print(f"\tVal Loss: {avg_val_loss} | Val ACC: {val_accuracy:.2f}%")


if __name__ == '__main__':
    training_loop()