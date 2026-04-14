
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

    return mean_map, coverage_map


def plot_coverage_maps(project_folder, mean_maps, coverage_maps, year_ranges):

    fig = plt.figure(figsize=(12, 10))

    plot_width = 5
    plot_height = 10
    h_spacing = 1
    colorbar_width = 1

    chl_min = 0
    chl_max = 1
    cov_min = 0
    cov_max = 0.1

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
        ax.pcolormesh(mean_maps[j], vmin=chl_min, vmax=chl_max, cmap='turbo')
        ax.set_title(f'{year_ranges[j][0]}-{year_ranges[j][1]}')
        ax.set_xticks([])
        ax.set_yticks([])

        plot_row = i//2
        plt_col = i%2 + 2
        ax = fig.add_subplot(gs[plot_row*plot_height:(plot_row+1)*plot_height, plt_col*plot_width+h_spacing:(plt_col+1)*plot_width+h_spacing])
        ax.pcolormesh(coverage_maps[j], vmin=cov_min, vmax=cov_max, cmap='viridis')
        ax.set_title(f'{year_ranges[j][0]}-{year_ranges[j][1]}')
        ax.set_xticks([])
        ax.set_yticks([])

    # make some custom colorbars
    cax1 = fig.add_subplot(gs[1:plot_height-1, plot_width+h_spacing:plot_width+h_spacing+colorbar_width])
    cx = np.array([0,1])
    cy = np.linspace(chl_min, chl_max,100)
    CX, CY = np.meshgrid(cx, cy)
    cax1.pcolormesh(CX, CY, CY, vmin=chl_min, vmax=chl_max, cmap='turbo')
    cax1.set_xticks([])
    cax1.set_ylabel('Mean Chlorophyll\n(mg/m$^3$)')
    cax1.yaxis.set_label_position("right")
    cax1.yaxis.tick_right()

    cax2 = fig.add_subplot(gs[1:plot_height-1, 3*plot_width+2*h_spacing:3*plot_width+2*h_spacing+colorbar_width])
    cx = np.array([0,1])
    cy = np.linspace(cov_min, cov_max,100)
    CX, CY = np.meshgrid(cx, cy)
    cax2.pcolormesh(CX, CY, CY, vmin=cov_min, vmax=cov_max, cmap='viridis')
    cax2.set_xticks([])
    cax2.set_ylabel('Data Coverage\n(%)')
    cax2.yaxis.set_label_position("right")
    cax2.yaxis.tick_right()

    output_file = os.path.join(project_folder, 'Figures', 'Chlorophyll_Data_Coverage_Maps.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology'

resolution = 15

year_ranges = [[1978,2026], [1978, 1986], [1997, 2006], [2007, 2016], [2015, 2026]]

mean_maps = []
mean_coverage_maps = []

for year_range in year_ranges:
    mean_map, mean_coverage_map = compute_coverage_map(project_folder, resolution, year_range)
    mean_maps.append(mean_map)
    mean_coverage_maps.append(mean_coverage_map)


plot_coverage_maps(project_folder, mean_maps, mean_coverage_maps, year_ranges)











