import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from scipy.ndimage import zoom

AIR_HU = -1024
MIN_HU = -1000
MAX_HU = 400

def world_to_voxel(world_coords_mm, origin_mm, spacing_mm):
    voxel_coords = (world_coords_mm - origin_mm) / spacing_mm
    return np.round(voxel_coords).astype(int)

class LunaDataset(Dataset):
    def __init__(self, data_dir, csv_path, is_val_set=False, val_stride=10, cache_dir="./LUNA16/cache"):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.is_val_set = is_val_set
        self.cache_dir = cache_dir

        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        all_candidates_df = pd.read_csv(self.csv_path)

        self.seriesuid_to_path = self.build_uid_to_path_dict(
            all_candidates_df['seriesuid'].unique(),
            subsets_to_check=['subset0', 'subset1']
        )

        available_uids = set(self.seriesuid_to_path.keys())
        all_candidates_df_available = all_candidates_df[
            all_candidates_df['seriesuid'].isin(available_uids)
        ]

        all_seriesuid_available = sorted(list(all_candidates_df_available['seriesuid'].unique()))

        val_series_uids = all_seriesuid_available[::val_stride]
        train_series_uids = [uid for uid in all_seriesuid_available if uid not in val_series_uids]

        if self.is_val_set:
            current_series_uids = val_series_uids
        else:
            current_series_uids = train_series_uids

        self.candidate_list = all_candidates_df_available[
            all_candidates_df_available['seriesuid'].isin(current_series_uids)
        ]

        self.candidate_list = self.candidate_list.reset_index(drop=True)

        self.cached_seriesuid = None
        self.cached_ct_array = None
        self.cached_origin_mm = None
        self.cached_spacing_mm = None

    def __len__(self):
        return len(self.candidate_list)

    def build_uid_to_path_dict(self, all_uids, subsets_to_check):
        uid_to_path = {}
        for uid in all_uids:
            found_path = None
            for subset_name in subsets_to_check:
                path = os.path.join(self.data_dir, subset_name, f"{uid}.mhd")
                if os.path.exists(path):
                    found_path = path
                    break
            if found_path:
                uid_to_path[uid] = found_path
        return uid_to_path

    def __getitem__(self, index):
        candidate_info = self.candidate_list.iloc[index]
        seriesuid = candidate_info['seriesuid']
        label = candidate_info['class']
        world_coords_mm = np.array((candidate_info['coordX'],
                                    candidate_info['coordY'],
                                    candidate_info['coordZ']))

        if seriesuid != self.cached_seriesuid:
            ct_array, new_origin_mm, new_spacing_mm = self.get_resampled_ct(seriesuid)
            self.cached_ct_array = ct_array
            self.cached_origin_mm = new_origin_mm
            self.cached_spacing_mm = new_spacing_mm
            self.cached_seriesuid = seriesuid

        voxel_coords_xyz = world_to_voxel(world_coords_mm,
                                          self.cached_origin_mm,
                                          self.cached_spacing_mm)

        center_zyx = (voxel_coords_xyz[2], voxel_coords_xyz[1], voxel_coords_xyz[0])

        chunk_size_zyx = (48, 48, 48)

        chunk_array = self.extract_chunk(self.cached_ct_array,
                                         center_zyx,
                                         chunk_size_zyx)

        chunk_array = np.clip(chunk_array, MIN_HU, MAX_HU)
        chunk_array = (chunk_array - MIN_HU) / (MAX_HU - MIN_HU)
        chunk_tensor = torch.from_numpy(chunk_array).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return chunk_tensor, label_tensor

    def get_resampled_ct(self, seriesuid):
        cache_path = None
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{seriesuid}.npy")
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    return data[0], data[1], data[2]
                except Exception as e:
                    print(f"Loading cache file failed {cache_path}: {e}, Resampling...")

        mhd_path = self.seriesuid_to_path[seriesuid]
        sitk_img = sitk.ReadImage(mhd_path)

        original_origin_mm = np.array(sitk_img.GetOrigin())
        original_spacing_mm = np.array(sitk_img.GetSpacing())
        original_ct_array = sitk.GetArrayFromImage(sitk_img)

        target_spacing_mm = np.array([1.0, 1.0, 1.0])
        spacing_ratio = original_spacing_mm / target_spacing_mm
        zoom_factor = [spacing_ratio[2], spacing_ratio[1], spacing_ratio[0]]

        resampled_ct_array = zoom(original_ct_array, zoom_factor, order=3, mode='nearest')

        resampled_ct_array = np.array(resampled_ct_array)
        new_origin_mm = original_origin_mm
        new_spacing_mm = target_spacing_mm

        if cache_path:
            try:
                data_to_save = np.array([resampled_ct_array, new_origin_mm, new_spacing_mm], dtype=object)
                np.save(cache_path, data_to_save)
            except Exception as e:
                print(f"Saving cache file failed: {e}")

        return resampled_ct_array, new_origin_mm, new_spacing_mm

    def extract_chunk(self, ct_array, center_zyx, size_zyx):
        chunk_array = np.full(size_zyx, AIR_HU, dtype=np.float32)

        z_size, y_size, x_size = size_zyx
        z_center, y_center, x_center = center_zyx

        z_start = z_center - z_size // 2
        z_end = z_start + z_size
        y_start = y_center - y_size // 2
        y_end = y_start + y_size
        x_start = x_center - x_size // 2
        x_end = x_start + x_size

        ct_z_start = max(0, z_start)
        ct_z_end = max(0, min(ct_array.shape[0], z_end))
        ct_y_start = max(0, y_start)
        ct_y_end = max(0, min(ct_array.shape[1], y_end))
        ct_x_start = max(0, x_start)
        ct_x_end = max(0, min(ct_array.shape[2], x_end))

        chunk_z_start = max(0, -z_start)
        chunk_z_end = chunk_z_start + (ct_z_end - ct_z_start)
        chunk_y_start = max(0, -y_start)
        chunk_y_end = chunk_y_start + (ct_y_end - ct_y_start)
        chunk_x_start = max(0, -x_start)
        chunk_x_end = chunk_x_start + (ct_x_end - ct_x_start)

        chunk_array[
            chunk_z_start : chunk_z_end,
            chunk_y_start : chunk_y_end,
            chunk_x_start : chunk_x_end
        ] = ct_array[
            ct_z_start : ct_z_end,
            ct_y_start : ct_y_end,
            ct_x_start : ct_x_end
        ]

        return chunk_array
