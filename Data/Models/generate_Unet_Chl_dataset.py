
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4

import torch
from torch.utils.data import DataLoader

from Unet.ChlDataset import ChlDataset
from Unet.ChlUnetModel import ChlUNet

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

def load_model_checkpoint(project_folder, model_version, epoch, device):

    checkpoint_file = os.path.join(project_folder,'Data', str(resolution)+'km Model',
                              model_version, 'Checkpoints',
                              f'{model_version}_checkpoint_epoch_{epoch}.pth')

    days = 7
    if model_version == 'Unet1':
        channels_per_day = 10
    if model_version == 'Unet2':
        channels_per_day = 8
    in_channels = days * channels_per_day + 1
    model = ChlUNet(in_channels=in_channels, out_channels=1, base=64).to(device)

    checkpoint = torch.load(checkpoint_file)#, map_location='mps')

    model.load_state_dict(checkpoint)

    return model

def read_model_grid(project_folder, resolution):
    grid_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

    ds = nc4.Dataset(grid_file)

    var_grids = {}
    var_grids['X'] = ds.variables['X'][:]
    var_grids['Y'] = ds.variables['Y'][:]
    var_grids['Depth'] = ds.variables['Depth'][:]

    ds.close()

    return var_grids

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


def run_file_check(project_folder, resolution, model_version, year_month, check_if_exists):
    year = year_month[0]
    month = year_month[1]

    if str(year) not in os.listdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version)):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, str(year)))

    output_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                               'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')

    if check_if_exists and os.path.isfile(output_file):
        print(' - File already exists for ' + str(year) + '-' + '{:02d}'.format(month) + ', skipping...')
        return False
    else:
        print(' - Generating modeled Chlorophyll dataset for ' + str(year) + '-' + '{:02d}'.format(month))
        return True

def write_Chl_dataset_to_nc(project_folder, model_version, year, month, var_grids, chl_grid):

    days = chl_grid.shape[0]
    print('Days:',days)

    if str(year) not in os.listdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version)):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, str(year)))

    output_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                               'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')

    ds = nc4.Dataset(output_file, 'w')

    ds.createDimension('x', np.shape(var_grids['Depth'])[1])
    ds.createDimension('y', np.shape(var_grids['Depth'])[0])
    ds.createDimension('days', days)

    dvar = ds.createVariable('days', 'f4', ('days',))
    dvar[:] = np.arange(1, days + 1)
    xvar = ds.createVariable('x', 'f4', ('x',))
    xvar[:] = var_grids['X'][0, :]
    xvar = ds.createVariable('y', 'f4', ('y',))
    xvar[:] = var_grids['Y'][:, 0]
    xvar = ds.createVariable('X', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['X'][:,:]
    xvar = ds.createVariable('Y', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['Y'][:,:]
    xvar = ds.createVariable('Depth', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['Depth']

    xvar = ds.createVariable('Chl', 'f4', ('days', 'y', 'x'))
    xvar[:, :, :] = chl_grid

    ds.close()



project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
epoch = 5
resolution = 15

start_year = 2019
end_year = 2019

check_if_exists = False

print(' - Loading the model checkpoint')
model = load_model_checkpoint(project_folder, model_version, epoch, device)

print(' - Reading in the domain grids' )
domain_grids = read_model_grid(project_folder, resolution)
depth_mask = domain_grids['Depth'] == 0

print(' - Reading in the normalization stats' )
normalization_stats = read_normalization_stats(project_folder, resolution, model_version)

print(' - Generating the monthly grids')
model.eval()

for year in range(start_year, end_year + 1):

    if year %4 ==0:
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    eval_dataset = ChlDataset(project_folder=project_folder, resolution=resolution,
                              model_version=model_version,
                              normalize_stats=normalization_stats, device=device,
                               start_year=year, end_year=year, training=False, printing=False)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    chl_grids = []
    days_counted = 0
    month = 1

    for batch in eval_loader:
        x = batch["x"]  # [B, C, 184, 103]
        ocean_mask = batch["ocean_mask"]  # [B, 1, 184, 103]
        seaice_mask = batch["seaice_mask"]  # [B, 1, 184, 103]

        # if days_counted == 0:
        #     SST_grid = x[0, 1, :, :].squeeze().cpu().numpy()
        #     SST_grid *= normalization_stats['SST']['std']
        #     SST_grid += normalization_stats['SST']['mean']
        #     # print(np.min(SST_grid), np.max(SST_grid))
        #
        #     plt.pcolormesh(SST_grid, cmap='viridis')
        #     plt.colorbar()
        #     plt.show()

        # convert seaice mask to numpy
        seaice_mask_np = seaice_mask.squeeze().cpu().numpy()

        with torch.no_grad():
            pred = model(x, ocean_mask)

            # convert pred to numpy and append to chl_grids
            chl_grid = pred.squeeze().cpu().numpy()

            # de-normalize the chl grid
            chl_grid *= normalization_stats['Chl']['std']
            chl_grid += normalization_stats['Chl']['mean']

            # de logify the chl grid
            chl_grid = np.exp(chl_grid)

            # mask out the sea ice regions in the chl grid
            chl_grid[seaice_mask_np == 0] = np.nan

            # mask out the land regions in the chl grid
            chl_grid[depth_mask] = np.nan

            chl_grids.append(chl_grid)

        days_counted += 1

        if days_counted == days_in_month[month-1]:
            print(' - Finished month ' + str(month) + ' of year ' + str(year))
            chl_grids = np.stack(chl_grids, axis=0)
            write_Chl_dataset_to_nc(project_folder, model_version, year, month, domain_grids, chl_grids)
            month += 1
            days_counted = 0
            chl_grids = []




