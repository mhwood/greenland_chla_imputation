
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from matplotlib.gridspec import GridSpec


def read_stats_grids_from_nc(project_folder, resolution, model_version, start_year, end_year, month='all'):
    file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,'Statistics',
                               model_version+'spatial_statistics_' + str(start_year) + '_' + str(end_year) + '_' + str(month) + '.nc')
    ds = nc4.Dataset(file_path)

    mean_grid = ds.variables['Mean_Chl'][:, :]
    RMSE_grid = ds.variables['RMSE_Chl'][:, :]
    count_grid = ds.variables['Count'][:, :]
    X = ds.variables['X'][:, :]
    Y = ds.variables['Y'][:, :]

    ds.close()

    return(mean_grid, RMSE_grid, count_grid, X, Y)

def plot_summary_statistics(project_folder, resolution, model_version, start_year, end_year):
    plt.figure(figsize=(10, 8))

    plt.subplot(1, 3, 1)
    C = plt.pcolormesh(X, Y, mean_grid, cmap='turbo', vmin=0, vmax=5)
    plt.colorbar(C)
    plt.title('Mean of Modeled Chl (mg/m^3)')

    plt.subplot(1, 3, 2)
    C = plt.pcolormesh(X, Y, RMSE_grid, vmin=0, vmax=1)
    plt.colorbar(C)
    plt.title('RMSE of Modeled Chl (mg/m^3)')

    plt.subplot(1, 3, 3)
    C = plt.pcolormesh(X, Y, count_grid)
    plt.colorbar(C)
    plt.title('Number of Valid Observations')

    plt.show()

def plot_monthly_statistics(project_folder, statistic, resolution, model_version, start_year, end_year):

    fig = plt.figure(figsize=(10,12))

    panel_width = 4
    panel_height = 6
    spacing = 1
    colorbar_height = 1

    if statistic=='mean':
        vmin = 0
        vmax = 5
        cmap = 'turbo'
    elif statistic=='RMSE':
        vmin = 0
        vmax = 1
        cmap = 'viridis'
    elif statistic=='count':
        vmin = 0
        vmax = 31
        cmap = 'plasma'

    gs = GridSpec(nrows=3*panel_height+spacing+colorbar_height, ncols=panel_width*4, figure=fig, wspace=0.1, hspace=0.1,
                  left=0.05, right=0.95, top=0.95, bottom=0.05)

    for month in range(1,13):
        mean_grid, RMSE_grid, count_grid, X, Y = read_stats_grids_from_nc(project_folder, resolution, model_version,
                                                                          start_year, end_year, month)
        if statistic=='mean':
            plot_grid = mean_grid
        elif statistic=='RMSE':
            plot_grid = RMSE_grid
        elif statistic=='count':
            plot_grid = count_grid

        print(month)
        print('rows',0, panel_height)
        print('cols',(month-1)*panel_width,(month)*panel_width)

        if month<=4:
            ax = fig.add_subplot(gs[:panel_height, (month-1)*panel_width:(month)*panel_width])
            C = ax.pcolormesh(X, Y, plot_grid, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title('Month: ' + str(month))
        elif month<=8 and month>4:
            ax = fig.add_subplot(gs[panel_height:2*panel_height, (month-5)*panel_width:(month-4)*panel_width])
            C = ax.pcolormesh(X, Y, plot_grid, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title('Month: ' + str(month))
        else:
            ax = fig.add_subplot(gs[2*panel_height+spacing:3*panel_height+spacing, (month-9)*panel_width:(month-8)*panel_width])
            C = ax.pcolormesh(X, Y, plot_grid, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title('Month: ' + str(month))


    cax = fig.add_subplot(gs[3*panel_height+spacing:3*panel_height+spacing+colorbar_height, panel_width:-panel_width])
    cx = np.linspace(vmin, vmax, 100)
    cy = np.array([0,1])
    CX, CY = np.meshgrid(cx, cy)
    C = cax.pcolormesh(CX, CY, CX, cmap=cmap, vmin=vmin, vmax=vmax)
    cax.set_yticks([])
    cax.set_xlabel(statistic)

    output_file = os.path.join(project_folder, 'Figures', model_version+'_monthly_' + statistic + '_statistics_' + str(start_year) + '_' + str(end_year) + '.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

start_year = 2006
end_year = 2009

for statistic in ['mean','RMSE','count']:
    plot_monthly_statistics(project_folder, statistic, resolution, model_version, start_year, end_year)


