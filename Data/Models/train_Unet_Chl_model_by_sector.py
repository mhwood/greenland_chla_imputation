
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4

import torch
from torch.utils.data import DataLoader

from Unet.ChlDatasetSector import ChlDatasetSector as ChlDataset
from Unet.ChlUnetModel import ChlUNet
from Unet.ChlUnetTraining import Unet_model_training_iteration, Unet_model_evaluation_iteration

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

def read_domain_from_nc(project_folder, resolution):
    output_file = os.path.join(project_folder, 'Data', '1km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

    ds = nc4.Dataset(output_file)

    X = ds.variables['X'][:,:]
    Y = ds.variables['Y'][:, :]
    Depth = ds.variables['Depth'][:, :]

    ds.close()

    return(X,Y, Depth)

def compute_sectors_dict(Depth,subset_size):
    
    rows = np.shape(Depth)[0]
    cols = np.shape(Depth)[1]

    n_row_tiles = np.ceil(rows/subset_size[0])+1
    n_col_tiles = np.ceil(cols/subset_size[1])+1

    row_starts = np.round(np.linspace(0, rows-subset_size[0], int(n_row_tiles))).astype(int)
    col_starts = np.round(np.linspace(0, cols-subset_size[1], int(n_col_tiles))).astype(int)

    subset_dict = {}

    for r in range(len(row_starts)):
        for c in range(len(col_starts)):
            depth_subset = Depth[row_starts[r]:row_starts[r]+subset_size[0], col_starts[c]:col_starts[c]+subset_size[1]]
            if np.any(depth_subset!=0):
                subset_dict[(int(row_starts[r]),int(col_starts[c]))] = depth_subset
    
    return(subset_dict)

def plot_sectors_on_map(project_folder,Depth,sectors_dict,subset_size):

    fig = plt.figure(figsize=(8,10))

    rows = np.arange(np.shape(Depth)[0])
    cols = np.arange(np.shape(Depth)[1])

    plt.pcolormesh(cols, rows, Depth, cmap='Blues')

    counter = 0
    for subset in sectors_dict:
        min_row = subset[0]
        min_col = subset[1]
        max_row = subset[0]+subset_size[0]
        max_col = subset[1]+subset_size[1]
        color = plt.rcParams['axes.prop_cycle'].by_key()['color'][counter % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
        plt.plot([min_col,min_col],[min_row,max_row],color=color)
        plt.plot([max_col,max_col],[min_row,max_row],color=color)
        plt.plot([min_col,max_col],[min_row,min_row],color=color)
        plt.plot([min_col,max_col],[max_row,max_row],color=color)
        plt.text((min_col+max_col)/2, (min_row+max_row)/2, str(subset), ha='center', va='center', color=color)

        counter+=1
    
    plt.savefig(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version,
                                f'{model_version}_sectors.png'))
    plt.close(fig)


def plot_first_field(batch):
    x = batch['x'][0]  # [C, H, W]
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
        plt.figure()
        plt_grid = np.ma.masked_where(x[i].numpy() == 0, x[i].numpy())
        plt.pcolormesh(plt_grid, cmap=cmaps[i % 8])
        plt.colorbar()
        plt.title(f"Field: {all_field_names[i]}")
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

home_folder = os.path.expanduser('~')
project_folder = home_folder+'/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 1
subset_size = (256,256)

if resolution == 15:
    data_folder = project_folder
else:
    data_folder = '/Volumes/CoOL/Projects/Greenland Chl Imputation'
    # data_folder = '/Volumes/ilulissat/Research/Projects/Greenland Chl Imputation'



start_year = 2001
start_month = 1

end_year = 2001
end_month = 12

X, Y, Depth = read_domain_from_nc(project_folder, resolution)

sectors_dict = compute_sectors_dict(Depth,subset_size)

# plot_sectors_on_map(project_folder,Depth,sectors_dict,subset_size)

sector_list = list(sectors_dict.keys())

for sector in sector_list[34:]:

    print(' - Working on '+str(sector))

    output_folder_subset = f'Sector_{sector[0]:02d}_{sector[1]:02d}'
    output_folder = os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version)
    if output_folder_subset not in os.listdir(output_folder):
        os.mkdir(os.path.join(output_folder,output_folder_subset))
    
    if 'Checkpoints' not in os.listdir(os.path.join(output_folder,output_folder_subset)):
        os.mkdir(os.path.join(output_folder,output_folder_subset,'Checkpoints'))

    output_summary = ''
    summary_path = os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, output_folder_subset,
                                f'{model_version}_training_summary.txt')

    # Step 1: Load in the model
    print(' - Loading in the model architecture and optimizer')
    days = 7
    if model_version == 'Unet1':
        channels_per_day = 10
    else:
        channels_per_day = 8
    in_channels = days * channels_per_day+1
    model = ChlUNet(in_channels=in_channels, out_channels=1, base=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # Step 2: Load in the training and testing data
    print(' - Creating the data loader for the training set')
    normalization_stats = read_normalization_stats(project_folder, resolution, model_version)
    print("Normalization stats:")#, normalization_stats)
    for field in normalization_stats:
        print('    - ',field,' Mean: ',normalization_stats[field]['mean'],' Std: ',normalization_stats[field]['std'])
    batch_size = 10
    train_dataset = ChlDataset(project_folder=project_folder, resolution=resolution,
                               model_version=model_version, time_window=int((days-1)/2),
                               normalize_stats=normalization_stats, device=device, data_folder=data_folder,
                               sector_ll_row=sector[0], sector_ll_col=sector[1], subset_size=subset_size,
                               start_year=start_year, end_year=end_year, training=True, printing=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # print the shapes of the first batch to verify
    print("First batch shapes:")
    first_batch = next(iter(train_loader))
    print(f"     - x: {first_batch['x'].shape}")
    print(f"     - target_log_chl: {first_batch['target_log_chl'].shape}")
    print(f"     - target_mask: {first_batch['target_mask'].shape}")
    print(f"     - ocean_mask: {first_batch['ocean_mask'].shape}")


    # Step 3: Train the model
    epochs = 5

    all_training_losses = []
    all_testing_losses = []

    for epoch in range(epochs):

        # training loop
        train_batch_counter = 0
        epoch_train_losses = []
        for batch in train_loader:
            train_loss, test_loss = Unet_model_training_iteration(model, optimizer, batch)
            epoch_train_losses.append(train_loss)
            message = f"Epoch {epoch+1}/{epochs} | Batch {train_batch_counter+1}/{len(train_loader)} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}"
            print(message)
            output_summary += message + '\n'
            train_batch_counter += 1

        all_training_losses += epoch_train_losses

        # save model checkpoint
        checkpoint_path = os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, output_folder_subset, 'Checkpoints',
                                       f'{model_version}_checkpoint_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)

    # save training summary to text file
    with open(summary_path, 'w') as f:
        f.write(output_summary)



