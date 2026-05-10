

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from sklearn.metrics import r2_score


def read_R2_values(project_folder, resolution, model_version, month='all'):

    if month == 'all':
        start_month = 1
        end_month = 12
    else:
        start_month = month
        end_month = month

    depths = np.arange(5,19)
    r2s_train = np.zeros(len(depths))
    r2s_test = np.zeros(len(depths))

    for max_depth in depths:
        print(f" - Working on max_depth={max_depth}")

        pred_file = os.path.join(project_folder, 'Data', str(resolution) + 'km Model', model_version,
                                   'Predictions', model_version + '_Predictions_' + str(max_depth) + '.nc')
        ds = nc4.Dataset(pred_file)
        y_train_pred = ds.variables['y_train_pred'][:]
        y_test_pred = ds.variables['y_test_pred'][:]
        ds.close()

        obs_file = os.path.join(project_folder, 'Data', str(resolution) + 'km Model', model_version,
                                      'XGB1_Training_Testing_Sets.nc')
        ds = nc4.Dataset(obs_file)
        y_train_obs = ds.variables['y_train'][:]
        y_test_obs = ds.variables['y_test'][:]
        ds.close()

        r2 = r2_score(y_test_obs, y_test_pred)
        r2s_test[depths==max_depth] = r2

        r2 = r2_score(y_train_obs, y_train_pred)
        r2s_train[depths==max_depth] = r2

    return(depths, r2s_train, r2s_test)

def plot_model_R2(project_folder, resolution, model_version, depths, r2s_train, r2s_test):
    fig = plt.figure(figsize=(9, 4))
    plt.plot(depths, r2s_train, 'k.-', label='Training set')
    plt.plot(depths, r2s_test, 'g.-', label='Testing set')
    plt.legend()

    plt.xlabel('Model Maximum Depth')
    plt.ylabel('R$^2$')

    plt.grid(linestyle='--', linewidth=0.5)

    output_file = os.path.join(project_folder, 'Figures', model_version + '_R2_vs_Depth.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

depths, r2s_train, r2s_test = read_R2_values(project_folder, resolution, model_version)

plot_model_R2(project_folder, resolution, model_version, depths, r2s_train, r2s_test)


