
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from matplotlib.gridspec import GridSpec


def compute_coverage_map(project_folder, resolution, year_range):

    if resolution==15:
        rows = 184
        cols = 103

    sum_map = np.zeros((rows, cols))
    count_map = np.zeros((rows, cols))
    N = 0

    for year in range(year_range[0], year_range[1] + 1):
        for month in range(1, 13):
            data_file = os.path.join(project_folder, 'Data', f'{resolution}km Interpolated', 'Chlorophyll', str(year), f'Chlorophyll_{year}{month:02d}.nc')
            if not os.path.isfile(data_file):
                # print(f'File {data_file} does not exist. Skipping.')
                continue

            with nc4.Dataset(data_file, 'r') as ds:
                chl_data = ds.variables['Chl'][:,:,:]
                X = ds.variables['X'][:, :]
                Y = ds.variables['Y'][:, :]
                N += chl_data.shape[0]

                for i in range(chl_data.shape[0]):
                    chl_slice = chl_data[i,:,:]
                    valid_mask = np.logical_and(~np.isnan(chl_slice),                                                 chl_slice > 0,
                                                 chl_slice >0)
                    sum_map[valid_mask] += chl_slice[valid_mask]
                    count_map[valid_mask] += 1

    coverage_map = count_map / N
    mean_map = np.zeros((rows, cols))
    valid_mask = count_map > 0
    mean_map[valid_mask] = sum_map[valid_mask] / count_map[valid_mask]
    mean_map[~valid_mask] = np.nan

    return mean_map, coverage_map, X, Y

def read_bedmachine_to_grid(data_folder):
    bm_file = os.path.join(data_folder,'Greenland','Bathymetry','BedMachine','BedMachineGreenland-v5.nc')
    ds = nc4.Dataset(bm_file)
    bm_x = ds.variables['x'][:]
    bm_y = ds.variables['y'][:]
    bed = ds.variables['bed'][:,:]
    surface = ds.variables['surface'][:,:]
    ds.close()

    bed[surface>0]=0

    # print('    - Interpolating the bathymetry to the grid')
    # bm_X, bm_Y = np.meshgrid(bm_x,bm_y)
    # points = np.column_stack([bm_X.ravel(), bm_Y.ravel()])
    # interp_bed = griddata(points, bed.ravel(), (X,Y), method='nearest')


    bm_X, bm_Y = np.meshgrid(bm_x, bm_y)

    skip = 5
    bm_X = bm_X[::skip,::skip]
    bm_Y = bm_Y[::skip,::skip]
    bed = bed[::skip,::skip]

    bm_Y = np.flipud(bm_Y)
    bed = np.flipud(bed)

    return(bm_X, bm_Y, bed)

def plot_coverage_maps(project_folder, X, Y, mean_maps, coverage_maps, year_ranges,
                       bm_X, bm_Y, land_mask):

    fig = plt.figure(figsize=(12, 10))

    plot_width = 5
    plot_height = 10
    h_spacing = 1
    colorbar_width = 1
    fontsize = 14

    chl_min = 0
    chl_max = 5
    cov_min = 0
    cov_max = 0.2

    gs = GridSpec(3*plot_height, 4*plot_width+h_spacing, figure=fig,
                  left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace=4, wspace=0.3)

    for j in range(len(year_ranges)):
        if j==0:
            i=0
        else:
            i = j+1
        plot_row = i//2
        plt_col = i%2
        ax = fig.add_subplot(gs[plot_row*plot_height:(plot_row+1)*plot_height, plt_col*plot_width:(plt_col+1)*plot_width])
        ax.pcolormesh(X, Y, mean_maps[j], vmin=chl_min, vmax=chl_max, cmap='turbo')
        ax.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        ax.contour(bm_X, bm_Y, land_mask, levels=[0.999], colors='k', linewidths=0.5)
        ax.text(np.mean(X), np.mean(Y),  f'{year_ranges[j][0]}\n-\n{year_ranges[j][1]}',
                zorder=11, ha='center', va='bottom', fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

        plot_row = i//2
        plt_col = i%2 + 2
        ax = fig.add_subplot(gs[plot_row*plot_height:(plot_row+1)*plot_height, plt_col*plot_width+h_spacing:(plt_col+1)*plot_width+h_spacing])
        ax.pcolormesh(X, Y, coverage_maps[j], vmin=cov_min, vmax=cov_max, cmap='viridis')
        ax.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        ax.contour(bm_X, bm_Y, land_mask, levels=[0.999], colors='k', linewidths=0.5)
        ax.text(np.mean(X), np.mean(Y), f'{year_ranges[j][0]}\n-\n{year_ranges[j][1]}', zorder=11,
                        ha = 'center', va = 'bottom', fontsize = fontsize)
        ax.set_xticks([])
        ax.set_yticks([])

    # make some custom colorbars
    cax1 = fig.add_subplot(gs[1:plot_height-1, plot_width+h_spacing:plot_width+h_spacing+colorbar_width])
    cx = np.array([0,1])
    cy = np.linspace(chl_min, chl_max,100)
    CX, CY = np.meshgrid(cx, cy)
    cax1.pcolormesh(CX, CY, CY, vmin=chl_min, vmax=chl_max, cmap='turbo')
    cax1.set_xticks([])
    cax1.set_ylabel('Mean Chlorophyll\n(mg/m$^3$)', fontsize = fontsize)
    for tick in cax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    cax1.yaxis.set_label_position("right")
    cax1.yaxis.tick_right()


    cax2 = fig.add_subplot(gs[1:plot_height-1, 3*plot_width+2*h_spacing:3*plot_width+2*h_spacing+colorbar_width])
    cx = np.array([0,1])
    cy = np.linspace(cov_min, cov_max,100)
    CX, CY = np.meshgrid(cx, cy)
    cax2.pcolormesh(CX, CY, CY, vmin=cov_min, vmax=cov_max, cmap='viridis')
    cax2.set_xticks([])
    cax2.set_ylabel('Data Coverage\n(%)', fontsize = fontsize)
    for tick in cax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize)
    cax2.yaxis.set_label_position("right")
    cax2.yaxis.tick_right()


    output_file = os.path.join(project_folder, 'Figures','Data Availability', 'Chlorophyll_Data_Coverage_Maps.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

home_folder = os.path.expanduser('~')
bedmachine_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

resolution = 15

year_ranges = [[1978,2026], [1978, 1986], [1997, 2006], [2007, 2016], [2015, 2026]]

print(' - Reading in the bedmachine data')
bm_X, bm_Y, bed = read_bedmachine_to_grid(bedmachine_folder)
land_mask = (bed == 0.0).astype(int)
land_mask = np.ma.masked_where(land_mask==0, land_mask)

print(' - Reading in the data coverage')
mean_maps = []
mean_coverage_maps = []

for year_range in year_ranges:
    mean_map, mean_coverage_map, X, Y = compute_coverage_map(project_folder, resolution, year_range)
    mean_maps.append(mean_map)
    mean_coverage_maps.append(mean_coverage_map)

print(' - Making the figure')
plot_coverage_maps(project_folder, X, Y, mean_maps, mean_coverage_maps, year_ranges,
                   bm_X, bm_Y, land_mask)











