from torch.utils.data import DataLoader
from testds import LunaDataset

DATA_DIR_LUNA16 = "./datasets/LUNA16"
CSV_PATH_CANDIDATES = "./datasets/LUNA16/candidates_V2.csv"

train_ds = LunaDataset(
    data_dir=DATA_DIR_LUNA16, 
    csv_path=CSV_PATH_CANDIDATES,
    is_val_set=False,
    val_stride=10
)

train_loader = DataLoader(
    train_ds,
    batch_size=16,
    num_workers=4,
    shuffle=True
)

try:
    first_batch_chunks, first_batch_labels = next(iter(train_loader))
    
    print(f"\n--- DataLoader 测试 ---")
    print(f"成功获取一个批次！")
    print(f"图像块 (Chunks) 的形状: {first_batch_chunks.shape}")
    print(f"标签 (Labels) 的形状: {first_batch_labels.shape}")

except Exception as e:
    print(f"\n--- DataLoader 测试失败 ---")
    print(f"错误: {e}")
    print("请检查你的文件路径 (DATA_DIR_LUNA16, CSV_PATH_CANDIDATES)")
    print("以及 `build_uid_to_path_dict` 中的路径假设！")