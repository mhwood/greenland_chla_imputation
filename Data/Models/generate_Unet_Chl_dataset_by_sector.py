
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4

import torch
from torch.utils.data import DataLoader

from Unet.ChlDatasetSector import ChlDatasetSector as ChlDataset
from Unet.ChlUnetModel import ChlUNet

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"
device = torch.device(device_name)

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

def load_model_checkpoint(project_folder, model_version, epoch, device, sector):

    output_folder_subset = f'Sector_{sector[0]:02d}_{sector[1]:02d}'
    checkpoint_file = os.path.join(project_folder,'Data', str(resolution)+'km Model',
                              model_version, output_folder_subset,
                              'Checkpoints',
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



home_folder = os.path.expanduser('~')
project_folder = home_folder+'/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
epoch = 5
subset_size = (256,256)
resolution=1

if resolution == 15:
    data_folder = project_folder
else:
    data_folder = '/Volumes/CoOL/Projects/Greenland Chl Imputation'
    # data_folder = '/Volumes/ilulissat/Research/Projects/Greenland Chl Imputation'


start_year = 2001
end_year = 2001

check_if_exists = False

sectors = [(0,0)]
subset_size = (256,256)

print(' - Reading in the domain grids' )
domain_grids = read_model_grid(project_folder, resolution)
# depth_mask = domain_grids['Depth'] == 0

print(' - Reading in the normalization stats' )
normalization_stats = read_normalization_stats(project_folder, resolution, model_version)


sectors_dict = compute_sectors_dict(domain_grids['Depth'],subset_size)
sectors = list(sectors_dict.keys())

for year in range(start_year, end_year + 1):

    if year %4 ==0:
        days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for month in range(5,6):#13):

            days_in_month = days_in_month[month-1]

            month_grid = np.zeros((days_in_month, np.shape(domain_grids['Depth'])[0], np.shape(domain_grids['Depth'])[1]))

            for sector in sectors[:33]:

                sector_depth = sectors_dict[sector]
                depth_mask = sector_depth == 0


                print(' - Loading the model checkpoint')
                model = load_model_checkpoint(project_folder, model_version, epoch, device, sector)

                model.eval()

                eval_dataset = ChlDataset(project_folder=project_folder, resolution=resolution,
                                        model_version=model_version,
                                        normalize_stats=normalization_stats, device=device, data_folder=data_folder,
                                        start_year=year, end_year=year, 
                                        start_month=month, end_month=month,
                                        sector_ll_row=sector[0], sector_ll_col=sector[1], subset_size=subset_size,
                                        training=False, printing=True)
                eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

                days_counted = 0

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

                        ll_row = sector[0]
                        ll_col = sector[1]
                        month_grid[days_counted, ll_row:ll_row+subset_size[0], ll_col:ll_col+subset_size[1]] = chl_grid

                    days_counted += 1

                print(' - Finished month ' + str(month) + ' of year ' + str(year))
                chl_grids = month_grid
                write_Chl_dataset_to_nc(project_folder, model_version, year, month, domain_grids, chl_grids)

                days_counted = 0




