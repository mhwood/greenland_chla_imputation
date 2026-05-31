
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from Unet.ChlDatasetLazyRead import ChlDatasetLazy
from Unet.ChlDataset import ChlDataset

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

def plot_first_field_comparison(batch, batch_lazy):
    x = batch['x'][0]  # [C, H, W]
    x_lazy = batch_lazy['x'][0]
    # field_name = ['Chl', 'SST', 'U_wind', 'V_wind', 'U', 'V', 'Lon', 'Lat', 'DOY_sin', 'DOY_cos']
    # cmaps = ['turbo', 'RdYlBu', 'seismic', 'seismic', 'seismic', 'seismic', 'viridis', 'viridis', 'twilight_shifted',
    #          'twilight_shifted']
    field_name = ['Chl', 'SST', 'U_wind', 'V_wind', 'Lon', 'Lat', 'DOY_sin', 'DOY_cos']
    cmaps = ['turbo', 'RdYlBu', 'seismic', 'seismic', 'viridis', 'viridis', 'twilight_shifted',
             'twilight_shifted']
    all_field_names = []
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        for field in field_name:
            all_field_names.append(f"{field}_t{offset}")
    all_field_names.append('Mask')
    for i in range(x.shape[0]):
        if 'DOY' in all_field_names[i]:
            plt.figure(figsize=(15,5))

            plt.subplot(1,3,1)
            plt_grid = np.ma.masked_where(x[i].numpy() == 0, x[i].numpy())
            plt.pcolormesh(plt_grid, cmap=cmaps[i % 8])
            plt.colorbar()
            plt.title(f"Field: {all_field_names[i]}")

            plt.subplot(1, 3, 2)
            plt_grid_lazy = np.ma.masked_where(x_lazy[i].numpy() == 0, x_lazy[i].numpy())
            plt.pcolormesh(plt_grid_lazy, cmap=cmaps[i % 8])
            plt.colorbar()
            plt.title(f"Lazy Field: {all_field_names[i]}")

            plt.subplot(1, 3, 3)
            diff = x[i].numpy() - x_lazy[i].numpy()
            diff_plt_grid = np.ma.masked_where(diff==0, diff)
            plt.pcolormesh(diff_plt_grid, cmap='seismic')
            plt.colorbar()
            plt.title(f"Difference")

            plt.show()

def read_normalization_stats(project_folder, resolution, model_version):
    stats_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                              str(resolution)+'km_normalization_stats.txt')

    normalization_stats = {}

    with open(stats_file, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            if len(parts) == 2:
                field_name = parts[0].strip()
                normalization_stats[field_name] = {}
                mean_std = parts[1].strip().split(',')
                if len(mean_std) == 2:
                    mean = float(mean_std[0].strip().split('=')[1])
                    std = float(mean_std[1].strip().split('=')[1])
                    normalization_stats[field_name]['mean'] = mean
                    normalization_stats[field_name]['std'] = std

    normalization_stats['Chl'] = normalization_stats['Chlorophyll']

    return normalization_stats

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 15

start_year = 2019
start_month = 1

end_year = 2019
end_month = 12

# Step 1: Load in the model
print(' - Loading in the model architecture and optimizer')
days = 7

# Step 2: Load in the training and testing data

normalization_stats = read_normalization_stats(project_folder, resolution, model_version)
print("Normalization stats:")#, normalization_stats)
for field in normalization_stats:
    print('    - ',field,' Mean: ',normalization_stats[field]['mean'],' Std: ',normalization_stats[field]['std'])


print(' - Creating the read-it-all dataset')
batch_size = 10
train_dataset = ChlDataset(project_folder=project_folder, resolution=resolution,
                           model_version=model_version,
                           normalize_stats=normalization_stats, device=device,
                           start_year=start_year, end_year=end_year, training=True, printing=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

print(' - Creating the lazy dataset')
train_dataset_lazy = ChlDatasetLazy(project_folder=project_folder, resolution=resolution,
                           model_version=model_version,
                           normalize_stats=normalization_stats, device=device,
                           start_year=start_year, end_year=end_year, training=True, printing=True)
train_loader_lazy = DataLoader(train_dataset_lazy, batch_size=batch_size, shuffle=False)

# print the shapes of the first batch to verify
print("First batch shapes:")
first_batch = next(iter(train_loader))
print(f"     - x: {first_batch['x'].shape}")
print(f"     - target_log_chl: {first_batch['target_log_chl'].shape}")
print(f"     - target_mask: {first_batch['target_mask'].shape}")
print(f"     - ocean_mask: {first_batch['ocean_mask'].shape}")

# print the shapes of the first batch to verify
print("First batch shapes:")
first_batch_lazy = next(iter(train_loader_lazy))
print(f"     - x: {first_batch_lazy['x'].shape}")
print(f"     - target_log_chl: {first_batch_lazy['target_log_chl'].shape}")
print(f"     - target_mask: {first_batch_lazy['target_mask'].shape}")
print(f"     - ocean_mask: {first_batch_lazy['ocean_mask'].shape}")

# plot_first_field_comparison(first_batch, first_batch_lazy)
