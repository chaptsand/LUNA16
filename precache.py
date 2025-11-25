import os
import time
from dsets import LunaDataset
from torch.utils.data import DataLoader

DATA_DIR = "./datasets/LUNA16"
CSV_PATH = "./datasets/LUNA16/candidates_V2.csv"

def main():
    print("Prechching...")
    ds = LunaDataset(
        data_dir=DATA_DIR,
        csv_path=CSV_PATH,
        is_val_set=True,
        val_stride=1
    )

    unique_uids = sorted(ds.candidate_list['seriesuid'].unique())

    total_uids = len(unique_uids)
    print(f"{total_uids} CT images are scheduled for resampling...")

    start_time = time.time()

    for i, uid in enumerate(unique_uids):
        loop_start = time.time()

        ds.get_resampled_ct(uid)

        duration = time.time() - loop_start
        print(f"[{i+1}/{total_uids}] done: {uid} (duration:{duration:.2f}s)")

    total_time = (time.time() - start_time) / 60
    print(f"All done! Total time:{total_time:.2f}m")

if __name__ == "__main__":
    main()
