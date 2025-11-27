import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dsets import LunaDataset, world_to_voxel

AIR_HU = -1024
MIN_HU = -1000
MAX_HU = 400

class LunaSegDataset(LunaDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ann_path = os.path.join(self.data_dir, "annotations.csv")
        self.annotations_df = pd.read_csv(self.ann_path)
        self.annotations_dict = self.build_annotations_dict()

        # self.candidate_list = self.candidate_list[self.candidate_list['class'] == 1]
        # self.candidate_list = self.candidate_list.reset_index(drop=True)
        # print(f"Dataset Filtered: Keeping only {len(self.candidate_list)} positive nodules.")

    def build_annotations_dict(self):
        ann_dict = {}
        for index, row in self.annotations_df.iterrows():
            uid = row['seriesuid']
            point = (row['coordX'], row['coordY'], row['coordZ'], row['diameter_mm'])

            if uid not in ann_dict:
                ann_dict[uid] = []
            ann_dict[uid].append(point)
        return ann_dict

    def __getitem__(self, index):
        candidate_info = self.candidate_list.iloc[index]
        seriesuid = candidate_info['seriesuid']

        if seriesuid != self.cached_seriesuid:
            ct_array, new_origin_mm, new_spacing_mm = self.get_resampled_ct(seriesuid)
            self.cached_ct_array = ct_array
            self.cached_origin_mm = new_origin_mm
            self.cached_spacing_mm = new_spacing_mm
            self.cached_seriesuid = seriesuid

        world_coords_mm = np.array((candidate_info['coordX'],
                                   candidate_info['coordY'],
                                   candidate_info['coordZ']))

        voxel_coords_xyz = world_to_voxel(world_coords_mm,
                                          self.cached_origin_mm,
                                          self.cached_spacing_mm)

        center_zyx = (voxel_coords_xyz[2], voxel_coords_xyz[1], voxel_coords_xyz[0])

        chunk_size_zyx = (48, 48, 48)
        chunk_array = self.extract_chunk(self.cached_ct_array, center_zyx, chunk_size_zyx)

        mask_array = np.zeros(chunk_size_zyx, dtype=np.float32)

        nodule_list = self.annotations_dict.get(seriesuid, [])

        for nodule_x, nodule_y, nodule_z, diameter_mm in nodule_list:
            nodule_center_mm = np.array((nodule_x, nodule_y, nodule_z))
            nodule_voxel_xyz = world_to_voxel(nodule_center_mm,
                                               self.cached_origin_mm,
                                               self.cached_spacing_mm)

            z_origin = center_zyx[0] - chunk_size_zyx[0] // 2
            y_origin = center_zyx[1] - chunk_size_zyx[1] // 2
            x_origin = center_zyx[2] - chunk_size_zyx[2] // 2

            nodule_z_in_chunk = nodule_voxel_xyz[2] - z_origin
            nodule_y_in_chunk = nodule_voxel_xyz[1] - y_origin
            nodule_x_in_chunk = nodule_voxel_xyz[0] - x_origin

            radius = diameter_mm // 2.0

            self.draw_sphere(mask_array,
                             (nodule_z_in_chunk, nodule_y_in_chunk, nodule_x_in_chunk),
                             radius)

        chunk_array = np.clip(chunk_array, MIN_HU, MAX_HU)
        chunk_array = (chunk_array - MIN_HU) / (MAX_HU - MIN_HU)

        chunk_tensor = torch.from_numpy(chunk_array).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array).float().unsqueeze(0)

        if self.augmentation:
            chunk_tensor, mask_tensor = self.augment_seg(chunk_tensor, mask_tensor)

        return chunk_tensor, mask_tensor

    def draw_sphere(self, mask_array, center_zyx, radius):
            cz, cy, cx = center_zyx
            depth, height, width = mask_array.shape

            z_grid, y_grid, x_grid = np.ogrid[:depth, :height, :width]

            dist2 = (z_grid - cz)**2 + (y_grid - cy)**2 + (x_grid - cx)**2

            mask_array[dist2 <= radius**2] = 1.0

    def augment_seg(self, chunk_tensor, mask_tensor):
        for axis in [1, 2, 3]:
            if random.random() > 0.5:
                chunk_tensor = torch.flip(chunk_tensor, [axis])
                mask_tensor = torch.flip(mask_tensor, [axis])

        if random.random() > 0.5:
            k = random.choice([1, 2, 3])
            dims = random.choice([(1, 2), (1, 3), (2, 3)])
            chunk_tensor = torch.rot90(chunk_tensor, k, dims)
            mask_tensor = torch.rot90(mask_tensor, k, dims)

        return chunk_tensor, mask_tensor
