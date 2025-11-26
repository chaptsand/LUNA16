import argparse
import os
import sys
import time
from torch.utils.data import Dataset, DataLoader
from dsets import LunaDataset
from tqdm import tqdm # 强烈建议安装: pip install tqdm

# 1. 定义一个专门用于缓存的 Dataset
class PrecacheDataset(Dataset):
    def __init__(self, luna_ds):
        self.luna_ds = luna_ds
        # 获取所有唯一的 seriesuid (去重)
        self.series_uids = sorted(self.luna_ds.seriesuid_to_path.keys())
        print(f"总共有 {len(self.series_uids)} 个唯一的 CT 扫描需要检查/处理。")

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, index):
        # 这里我们只关心“触发缓存计算”，不需要返回具体的大数组
        uid = self.series_uids[index]
        try:
            # 调用 dsets.py 中的核心逻辑
            # 如果缓存存在，它会直接返回；如果不存在，它会计算并保存
            self.luna_ds.get_resampled_ct(uid)
            return True # 返回一个简单的标志
        except Exception as e:
            print(f"Error processing {uid}: {e}")
            return False

def main():
    # --- 配置区域 ---
    # 你的 CPU 有 20 个逻辑核心。
    # 留几个给系统，开 12-16 个是比较合理的。
    # 注意：开太多会导致内存压力过大 (每个进程都要加载一个 CT)
    NUM_WORKERS = 12
    BATCH_SIZE = 1 # 处理是以 CT 为单位的，Batch Size 设为 1 即可，方便看进度

    DATA_DIR = "./datasets/LUNA16"
    CSV_PATH = "./datasets/LUNA16/candidates_V2.csv"
    # ----------------

    print(f"🚀 启动多进程缓存预处理 (Workers: {NUM_WORKERS})...")

    # 1. 初始化原始 Dataset (为了获取路径和逻辑)
    # val_stride=1 确保我们在内部拿到所有的 UID
    base_ds = LunaDataset(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        is_val_set=True,
        val_stride=1
    )

    # 2. 包装成按 CT 遍历的 Dataset
    pre_ds = PrecacheDataset(base_ds)

    # 3. 使用 DataLoader 实现多进程
    # collate_fn=None: 因为我们返回的是简单的 True/False，不需要复杂的拼接
    loader = DataLoader(
        pre_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False, # 顺序处理即可
        collate_fn=lambda x: x # 简单的占位符，不进行 tensor 转换
    )

    # 4. 开始循环
    start_time = time.time()

    # 使用 tqdm 显示进度条
    # desc: 进度条左边的文字
    # unit: 单位
    success_count = 0
    with tqdm(total=len(pre_ds), desc="Caching CTs", unit="scan") as pbar:
        for results in loader:
            # results 是一个 batch 的 True/False 列表
            success_count += sum(results)
            pbar.update(len(results))

    duration = (time.time() - start_time) / 60
    print(f"\n✅ 完成！")
    print(f"耗时: {duration:.2f} 分钟")
    print(f"成功处理: {success_count}/{len(pre_ds)}")

if __name__ == '__main__':
    # Windows/Linux 多进程必须在 __main__ 保护下运行
    main()
