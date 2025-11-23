from dsets import LunaDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义超参数 (HYPERPARAMETERS)
# BATCH_SIZE = ...
# LEARNING_RATE = ...
# EPOCHS = ...

# 2. 初始化 Datasets 和 DataLoaders
# train_ds = LunaDataset(...)
# train_loader = DataLoader(...)

# 3. 初始化模型、损失函数、优化器
# model = YourModelName()
# loss_fn = nn.BCEWithLogitsLoss() # (一个好的分类损失)
# optimizer = optim.Adam(...)

# 4. 训练循环
# for epoch in range(EPOCHS):
#   ... (我们稍后会填写这个)

print("训练脚本已准备好。")