
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from Unet.ChlDataset import ChlDataset
from Unet.ChlUnetModel import ChlUNet
from Unet.ChlUnetTraining import Unet_model_training_iteration, Unet_model_evaluation_iteration

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

def plot_first_field(batch):
    x = batch['x'][0]  # [C, H, W]
    field_name = ['Chl', 'SST', 'U_wind', 'V_wind', 'U', 'V', 'Lon', 'Lat', 'DOY_sin', 'DOY_cos']
    cmaps = ['turbo', 'RdYlBu', 'seismic', 'seismic', 'seismic', 'seismic', 'viridis', 'viridis', 'twilight_shifted',
             'twilight_shifted']
    all_field_names = []
    for offset in [-3, -2, -1, 0, 1, 2, 3]:
        for field in field_name:
            all_field_names.append(f"{field}_t{offset}")
    all_field_names.append('Mask')
    for i in range(x.shape[0]):
        plt.figure()
        plt_grid = np.ma.masked_where(x[i].numpy() == 0, x[i].numpy())
        plt.pcolormesh(plt_grid, cmap=cmaps[i % 10])
        plt.colorbar()
        plt.title(f"Field: {all_field_names[i]}")
        plt.show()

def write_normalization_stats_to_file(project_folder, resolution, model_version, normalization_stats):
    stats_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version,
                              model_version+'_normalization_stats.txt')
    with open(stats_file, 'w') as f:
        for var_name, stats in normalization_stats.items():
            f.write(f"{var_name}: mean={stats['mean']}, std={stats['std']}\n")


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet1'
resolution = 15

output_summary = ''
summary_path = os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version,
                                f'{model_version}_training_summary.txt')

start_year = 1998
start_month = 1

end_year = 2020
end_month = 12

# Step 1: Load in the model
days = 7
channels_per_day = 10
in_channels = days * channels_per_day+1
model = ChlUNet(in_channels=in_channels, out_channels=1, base=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)


# Step 2: Load in the training and testing data
batch_size = 10
train_dataset = ChlDataset(project_folder=project_folder, resolution=resolution,
                           normalize_stats={}, device=device,
                           start_year=start_year, end_year=end_year, training=True, printing=True)
normlization_stats = train_dataset.normalize_stats
write_normalization_stats_to_file(project_folder, resolution, model_version, normlization_stats)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# print the shapes of the first batch to verify
print("First batch shapes:")
first_batch = next(iter(train_loader))
print(f"x: {first_batch['x'].shape}")
print(f"target_log_chl: {first_batch['target_log_chl'].shape}")
print(f"target_mask: {first_batch['target_mask'].shape}")
print(f"ocean_mask: {first_batch['ocean_mask'].shape}")

# for every field in the first x of the first batch, make a plot of the first channel
# plot_first_field(first_batch)

# test_dataset = ChlDataset(project_folder=project_folder, resolution=resolution, normalize_stats=train_dataset.normalize_stats,
#                             start_year=start_year, end_year=end_year, training=False, printing=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Step 3: Train the model
epochs = 10

all_training_losses = []
all_testing_losses = []

for epoch in range(epochs):

    # training loop
    train_batch_counter = 0
    epoch_train_losses = []
    for batch in train_loader:
        train_loss = Unet_model_training_iteration(model, optimizer, batch)
        epoch_train_losses.append(train_loss)
        message = f"Epoch {epoch+1}/{epochs} | Batch {train_batch_counter+1}/{len(train_loader)} | Train Loss: {train_loss:.4f}"
        print(message)
        output_summary += message + '\n'
        train_batch_counter += 1

    # # evaluation loop
    # eval_batch_counter = 0
    # epoch_test_losses = []
    # for batch in test_loader:
    #     eval_loss = Unet_model_evaluation_iteration(model, batch)
    #     epoch_test_losses.append(eval_loss)
    #     message= f"Epoch {epoch+1}/{epochs} | Batch {eval_batch_counter+1}/{len(test_loader)} | Test Loss: {eval_loss:.4f}"
    #     print(message)
    #     output_summary += message + '\n'
    #     eval_batch_counter += 1

    all_training_losses += epoch_train_losses
    # all_testing_losses += epoch_test_losses

    # # print epoch summary
    # message = f"Epoch {epoch+1}/{epochs} | Train Loss: {np.mean(epoch_train_losses):.4f} | Test Loss: {np.mean(epoch_test_losses):.4f}"

    # save model checkpoint
    checkpoint_path = os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, 'Checkpoints',
                                   f'{model_version}_checkpoint_epoch_{epoch+1}.pth')
    torch.save(model.state_dict(), checkpoint_path)

# save training summary to text file
with open(summary_path, 'w') as f:
    f.write(output_summary)



