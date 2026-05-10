
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import xgboost as xgb
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec

def read_training_testing_set_comparisons(project_folder, resolution, model_version, max_depth=18):

    first_file = True

    y_train_obs_sets = {}
    y_train_pred_sets = {}
    y_test_obs_sets = {}
    y_test_pred_sets = {}

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
    X_train = ds.variables['X_train'][:,:]
    X_test = ds.variables['X_test'][:,:]
    ds.close()

    # features = sst, uwind, vwind, u, v, DOY_sin, DOY_cos, X, Y
    doy_sin_train = X_train[:,5]
    doy_cos_train = X_train[:,6]
    doy_sin_test = X_test[:,5]
    doy_cos_test = X_test[:,6]

    # plt.plot(doy_sin_train, doy_cos_train, 'k.', label='Train Set')
    # plt.show()

    y_train_pred_sets['all'] = y_train_pred
    y_test_pred_sets['all'] = y_test_pred
    y_train_obs_sets['all'] = y_train_obs
    y_test_obs_sets['all'] = y_test_obs

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for month in range(1,13):
        min_day = sum(days_in_month[:month-1])
        max_day = min_day + days_in_month[month-1]
        min_doy_sin = np.sin(2*np.pi*min_day/365)
        max_doy_sin = np.sin(2*np.pi*max_day/365)
        if min_doy_sin > max_doy_sin:
            min_doy_sin, max_doy_sin = max_doy_sin, min_doy_sin
        min_doy_cos = np.cos(2*np.pi*min_day/365)
        max_doy_cos = np.cos(2*np.pi*max_day/365)
        if min_doy_cos > max_doy_cos:
            min_doy_cos, max_doy_cos = max_doy_cos, min_doy_cos

        # print(min_day, max_day, np.sin(2*np.pi*min_day/365), np.sin(2*np.pi*max_day/365), np.cos(2*np.pi*min_day/365), np.cos(2*np.pi*max_day/365))
        month_mask_train_sin = (doy_sin_train >= min_doy_sin) & (doy_sin_train < max_doy_sin)
        month_mask_train_cos = (doy_cos_train >= min_doy_cos) & (doy_cos_train < max_doy_cos)
        month_mask_test_sin = (doy_sin_test >= min_doy_sin) & (doy_sin_test < max_doy_sin)
        month_mask_test_cos = (doy_cos_test >= min_doy_cos) & (doy_cos_test < max_doy_cos)

        train_day_percent_in_month = np.sum(month_mask_train_sin & month_mask_train_cos) / len(y_train_obs)
        print('   - Month: ' + str(month) + ', Train Percent in Month: ' + str(round(train_day_percent_in_month*100, 2)) +
              '%, Test Percent in Month: ' + str(round(np.sum(month_mask_test_sin & month_mask_test_cos) / len(y_test_obs)*100, 2)) + '%')

        y_train_obs_month = y_train_obs[month_mask_train_sin & month_mask_train_cos]
        y_train_pred_month = y_train_pred[month_mask_train_sin & month_mask_train_cos]
        y_test_obs_month = y_test_obs[month_mask_test_sin & month_mask_test_cos]
        y_test_pred_month = y_test_pred[month_mask_test_sin & month_mask_test_cos]

        y_train_pred_sets[month] = y_train_pred_month
        y_test_pred_sets[month] = y_test_pred_month
        y_train_obs_sets[month] = y_train_obs_month
        y_test_obs_sets[month] = y_test_obs_month

    return(y_train_obs_sets, y_test_obs_sets, y_train_pred_sets, y_test_pred_sets)

def plot_monthly_R2_results(project_dir, model_version, subset, y_train_obs_sets, y_test_obs_sets, y_train_pred_sets, y_test_pred_sets):

    if subset == 'train':
        y_obs_sets = y_train_obs_sets
        y_pred_sets = y_train_pred_sets
    elif subset == 'test':
        y_obs_sets = y_test_obs_sets
        y_pred_sets = y_test_pred_sets

    y_min = -2
    y_max = 2

    fig = plt.figure(figsize=(10,9))

    panel_width = 1
    panel_height = 1

    gs = GridSpec(nrows=3*panel_height, ncols=panel_width*4, figure=fig, wspace=0.1, hspace=0.2,
                  left=0.05, right=0.95, top=0.95, bottom=0.05)

    for month in range(1,13):

        y_obs = y_obs_sets[month]
        y_pred = y_pred_sets[month]

        if month<=4:
            ax = fig.add_subplot(gs[:panel_height, (month-1)*panel_width:(month)*panel_width])
        elif month<=8 and month>4:
            ax = fig.add_subplot(gs[panel_height:2*panel_height, (month-5)*panel_width:(month-4)*panel_width])
        else:
            ax = fig.add_subplot(gs[2*panel_height:3*panel_height, (month-9)*panel_width:(month-8)*panel_width])

        ax.plot([y_min, y_max], [y_min, y_max], '--',color='silver')

        C = ax.plot(y_obs, y_pred, 'k.')
        ax.set_title('Month: ' + str(month))

        if month<9:
            ax.set_xticklabels([])
        if month not in [1,5,9]:
            ax.set_yticklabels([])

        ax.text(0.05, 0.95, 'N: ' + str(len(y_obs)), transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.90, 'RMSE: ' + str(round(np.sqrt(np.mean((y_obs - y_pred) ** 2)), 2)), transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.85, 'R2: ' + str(round(r2_score(y_obs, y_pred), 2)), transform=ax.transAxes, verticalalignment='top')

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)

    output_file = os.path.join(project_folder, 'Figures', model_version+'_monthly_' + subset + '_R2.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

print(' - Reading the training and testing set comparisons')
y_train_obs_sets, y_test_obs_sets, y_train_pred_sets, y_test_pred_sets = read_training_testing_set_comparisons(project_folder, resolution, model_version)

print(' - Plotting the model results')
for subset in ['train', 'test']:
    plot_monthly_R2_results(project_folder, model_version, subset, y_train_obs_sets, y_test_obs_sets, y_train_pred_sets, y_test_pred_sets)

