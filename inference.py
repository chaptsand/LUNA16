import torch
import random
import matplotlib.pyplot as plt
from dsets_seg import LunaSegDataset
from model_seg import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ds = LunaSegDataset(
    data_dir="./datasets/LUNA16",
    csv_path="./datasets/LUNA16/candidates_V2.csv",
    is_val_set=True,
    val_stride=10,
    cache_dir="./LUNA16/cache",
    augmentation=False
)

model = UNet(in_channels=1, n_classes=1).to(device)
model.load_state_dict(torch.load("luna_unet_latest.pth", map_location=device))
model.eval()

print("🔍 正在抽取幸运观众...")
while True:
    idx = random.randint(0, len(ds)-1)
    sample_ct, sample_mask = ds[idx]
    if sample_mask.sum() > 0:
        break

print(f"✅ 选中样本 Index: {idx}")

input_tensor = sample_ct.unsqueeze(0).to(device)
with torch.no_grad():
    logits = model(input_tensor)
    pred_mask = torch.sigmoid(logits)

ct_slice = sample_ct[0, 24, :, :].cpu().numpy()
mask_slice = sample_mask[0, 24, :, :].cpu().numpy()
pred_slice = pred_mask[0, 0, 24, :, :].cpu().numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("CT Input")
plt.imshow(ct_slice, cmap='gray')

plt.subplot(1, 3, 2)
plt.title("Ground Truth")
plt.imshow(mask_slice, cmap='gray')

plt.subplot(1, 3, 3)
plt.title("AI Prediction")
plt.imshow(pred_slice, cmap='gray')

plt.show()
