
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import xgboost as xgb
from sklearn.metrics import r2_score

def read_training_testing_set_comparisons(project_folder, resolution, model_version, month='all'):

    first_file = True

    if month == 'all':
        start_month = 1
        end_month = 12
    else:
        start_month = month
        end_month = month

    training_file = os.path.join(project_folder, 'Data', str(resolution) + 'km Model', model_version,
                                 'XGB1_Training_Testing_Sets.nc')
    ds = nc4.Dataset(training_file)
    X_train = ds.variables['X_train'][:,:]
    X_test = ds.variables['X_test'][:,:]
    y_train_obs = ds.variables['y_train'][:]
    y_test_obs = ds.variables['y_test'][:]
    ds.close()

    model_file = os.path.join(project_folder, 'Data', str(resolution) + 'km Model', model_version,
                              model_version + '_Model_Checkpoint.json')
    model = xgb.XGBRegressor()
    model.load_model(model_file)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)


    return(y_train_obs, y_test_obs, y_train_pred, y_test_pred)


def plot_model_R2_results(project_dir, model_version, y_train_obs, y_test_obs, y_train_pred, y_test_pred):

    y_min = min(np.min(y_train_obs), np.min(y_test_obs), np.min(y_train_pred), np.min(y_test_pred))
    y_max = max(np.max(y_train_obs), np.max(y_test_obs), np.max(y_train_pred), np.max(y_test_pred))

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(y_train_obs, y_train_pred, 'k.')
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=1)
    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.ylabel('Modeled log10(Chl)')
    plt.xlabel('True log10(Chl)')
    plt.title(f"Training set, R$^2$= {r2_score(y_train_obs, y_train_pred):.2f}")

    plt.subplot(1, 2, 2)
    plt.plot(y_test_obs, y_test_pred, 'k.')
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=1)
    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('True log10(Chl)')
    plt.title(f"Testing set, R$^2$= {r2_score(y_test_obs, y_test_pred):.2f}")

    fig_path = os.path.join(project_dir, 'Figures', 'Chl_Model_' + (model_version) + '_R2.png')
    plt.savefig(fig_path)
    plt.close(fig)


def plot_model_R2_results_with_distance(project_folder, X, Y, Depth, model,
                                        X_train, X_test, Y_train, Y_test, Loc_train, Loc_test):

    distances = np.zeros((np.size(Depth),))
    distances_train = np.zeros((np.shape(Loc_train)[0],))
    distances_test = np.zeros((np.shape(Loc_test)[0],))

    depth = Depth.ravel()
    x = X.ravel()
    y = Y.ravel()
    x_train = Loc_train[:,0]
    x_test = Loc_test[:,0]
    y_train = Loc_train[:,1]
    y_test = Loc_test[:,1]
    depth_train = Loc_train[:, 2]
    depth_test = Loc_test[:, 2]

    ocean_indices = depth>0
    x_land = x[~ocean_indices]
    y_land = y[~ocean_indices]

    ocean_indices_train = depth_train > 0
    x_land_train = x_train[~ocean_indices_train]
    y_land_train = y_train[~ocean_indices_train]

    ocean_indices_test = depth_test > 0
    x_land_test = x_test[~ocean_indices_test]
    y_land_test = y_test[~ocean_indices_test]

    for d in range(len(depth)):
        if ocean_indices[d]:
            dist = (x[d]-x_land)**2 + (y[d]-y_land)**2
            distances[d] = np.min(dist)

    for d in range(len(depth_train)):
        if ocean_indices_train[d]:
            dist = (x_train[d] - x_land_train) ** 2 + (y_train[d] - y_land_train) ** 2
            distances_train[d] = np.min(dist)

    for d in range(len(depth_test)):
        if ocean_indices_test[d]:
            dist = (x_test[d] - x_land_test) ** 2 + (y_test[d] - y_land_test) ** 2
            distances_test[d] = np.min(dist)

    bins = np.linspace(0,5e11,101)

    binned_r2_train = np.zeros(100)
    binned_r2_test = np.zeros(100)
    count_train = np.zeros(100)
    count_test = np.zeros(100)

    for b in range(100):
        dist_indices = np.logical_and(distances>bins[b], distances<=bins[b+1])
        dist_indices_train = np.logical_and(distances_train > bins[b], distances_train <= bins[b + 1])
        dist_indices_test = np.logical_and(distances_test > bins[b], distances_test <= bins[b + 1])

        if np.any(dist_indices_train):
            X_train_subset = X_train[dist_indices_train]
            Y_train_subset = Y_train[dist_indices_train]
            Y_pred_train_subset = model.predict(X_train_subset)
            binned_r2_train[b] = r2_score(Y_train_subset, Y_pred_train_subset)
            count_train[b] = np.sum(dist_indices_train)

        if np.any(dist_indices_test):
            X_test_subset = X_test[dist_indices_test]
            Y_test_subset = Y_test[dist_indices_test]
            Y_pred_test_subset = model.predict(X_test_subset)
            binned_r2_test[b] = r2_score(Y_test_subset, Y_pred_test_subset)
            count_test[b] = np.sum(dist_indices_test)

    plt.subplot(2,2,1)
    plt.plot(count_train)
    plt.subplot(2, 2, 2)
    plt.plot(count_test)
    plt.subplot(2, 2, 3)
    plt.plot(binned_r2_train)
    plt.subplot(2, 2, 4)
    plt.plot(binned_r2_test)
    plt.show()

    # distances = distances.reshape(np.shape(Depth))

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

print(' - Reading the training and testing set comparisons')
y_train_obs, y_test_obs, y_train_pred, y_test_pred = read_training_testing_set_comparisons(project_folder, resolution, model_version)

print(' - Plotting the model results')
plot_model_R2_results(project_folder, model_version, y_train_obs, y_test_obs, y_train_pred, y_test_pred)

