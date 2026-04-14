
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4

import xgboost as xgb
from sklearn.metrics import r2_score
import shap
from sklearn.model_selection import train_test_split

def read_all_data_subsets(project_folder, year, month):

    var_grids = {}

    for data_subset in ['SST','SSS','Sea_Ice','Surface_Wind','Velocity','Chlorophyll']:

        file_path = os.path.join(project_folder, '', '15km Interpolated',
                                 data_subset, data_subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')
        ds = nc4.Dataset(file_path)
        if data_subset=='SST':
            var_grids['SST'] = ds.variables['SST'][:,:,:]
            var_grids['X'] = ds.variables['X'][:,:]
            var_grids['Y'] = ds.variables['Y'][:, :]
            var_grids['Depth'] = ds.variables['Depth'][:, :]
            # print('SST',np.nanmin(var_grids['SST']), np.nanmax(var_grids['SST']))
        if data_subset=='SSS':
            var_grids['SSS'] = ds.variables['SSS'][:,:,:]
            # print('SSS',np.nanmin(var_grids['SSS']), np.nanmax(var_grids['SSS']))
        if data_subset=='Sea_Ice':
            var_grids['Sea_Ice'] = ds.variables['seaice_fraction'][:,:,:]
        if data_subset=='Surface_Wind':
            var_grids['U_wind'] = ds.variables['U_wind'][:,:,:]
            var_grids['V_wind'] = ds.variables['V_wind'][:, :, :]
            # print('U_wind',np.nanmin(var_grids['U_wind']), np.nanmax(var_grids['U_wind']))
        if data_subset=='Velocity':
            var_grids['U'] = ds.variables['U'][:,:,:]
            var_grids['V'] = ds.variables['V'][:, :, :]
            # print('V', np.nanmin(var_grids['V']), np.nanmax(var_grids['V']))
        if data_subset=='Chlorophyll':
            var_grids['Chl'] = ds.variables['Chl'][:,:,:]

    return(var_grids)

def create_training_and_testing_sets(var_grids, year, month, days, model_version):

    day_Xs = []
    day_ys = []
    Depth = var_grids['Depth'][:, :].ravel()
    for day in range(len(days)):
        chl_day = np.log10(var_grids['Chl'][day, :, :]).ravel()
        sst_day = var_grids['SST'][day, :, :].ravel()
        sss_day = var_grids['SSS'][day, :, :].ravel()
        seaice_day = var_grids['Sea_Ice'][day, :, :].ravel()
        uwind_day = var_grids['U_wind'][day, :, :].ravel()
        vwind_day = var_grids['V_wind'][day, :, :].ravel()
        u_day = var_grids['U'][day, :, :].ravel()
        v_day = var_grids['V'][day, :, :].ravel()
        day_X = var_grids['X'][:, :].ravel()
        day_Y = var_grids['Y'][:, :].ravel()
        day_Depth = var_grids['Depth'][:, :].ravel()

        if model_version=='XGB1':
            good_indices = ~np.isnan(chl_day) & \
                           ~np.isnan(sst_day) & \
                           ~np.isnan(seaice_day) & \
                           ~np.isnan(uwind_day) & \
                           ~np.isnan(vwind_day) & \
                           ~np.isnan(u_day) & \
                           ~np.isnan(v_day)
            X_day = np.column_stack([day_X.ravel(),
                                     day_Y.ravel(),
                                     day_Depth.ravel(),
                                     sst_day.ravel(),
                                     seaice_day.ravel(),
                                     uwind_day.ravel(),
                                     vwind_day.ravel(),
                                     u_day.ravel(),
                                     v_day.ravel()])

        if model_version=='XGB2':
            good_indices = ~np.isnan(chl_day) & \
                           ~np.isnan(sst_day) & \
                           ~np.isnan(sss_day) & \
                           ~np.isnan(seaice_day) & \
                           ~np.isnan(uwind_day) & \
                           ~np.isnan(vwind_day) & \
                           ~np.isnan(u_day) & \
                           ~np.isnan(v_day)
            X_day = np.column_stack([day_X.ravel(),
                                     day_Y.ravel(),
                                     day_Depth.ravel(),
                                     sst_day.ravel(),
                                     sss_day.ravel(),
                                     seaice_day.ravel(),
                                     uwind_day.ravel(),
                                     vwind_day.ravel(),
                                     u_day.ravel(),
                                     v_day.ravel()])

        y_day = chl_day.ravel().reshape((np.size(chl_day), 1))

        X_day = X_day[good_indices, :]
        y_day = y_day[good_indices, :]

        day_Xs.append(X_day)
        day_ys.append(y_day)

    X = np.vstack(day_Xs)
    y = np.vstack(day_ys)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

    Loc_train = X_train[:, :3]
    Loc_test = X_test[:, :3]

    X_train = X_train[:,3:]
    X_test = X_test[:,3:]

    return(X_train, X_test, y_train, y_test, Loc_train, Loc_test)

def generate_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=51
    )
    model.fit(X_train, y_train)

    # ckpt_path = os.path.join(project_folder, 'Model', 'Remote Sensing Model', 'Model_Checkpoint_1.ckpt')
    # model.save(ckpt_path)

    return(model)

def plot_model_R2_results(project_dir, model_version, model, X_train, X_test, y_train, y_test):
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    y_min = np.min(y_train)
    y_max = np.max(y_train)

    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(y_train, y_pred_train, 'k.')
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=1)
    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.ylabel('Modeled log10(Chl)')
    plt.xlabel('True log10(Chl)')
    plt.title(f"Training set, R$^2$= {r2_score(y_train, y_pred_train):.2f}")

    plt.subplot(1, 2, 2)
    plt.plot(y_test, y_pred_test, 'k.')
    plt.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=1)
    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.xlabel('True log10(Chl)')
    plt.title(f"Testing set, R$^2$= {r2_score(y_test, y_pred_test):.2f}")

    fig_path = os.path.join(project_dir, '../Figures', 'Remote Sensing Model', 'Chl_Model_' + (model_version) + '_R2.png')
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


def generate_modeled_Chl_dataset(project_folder, model_version, model, year, month, days, var_grids):

    Chl = np.zeros((len(days), np.shape(var_grids['X'])[0], np.shape(var_grids['Y'])[1]))

    Depth = var_grids['Depth'][:, :].ravel()
    for day in range(len(days)):
        sst_day = var_grids['SST'][day, :, :].ravel()
        sss_day = var_grids['SSS'][day, :, :].ravel()
        seaice_day = var_grids['Sea_Ice'][day, :, :].ravel()
        uwind_day = var_grids['U_wind'][day, :, :].ravel()
        vwind_day = var_grids['V_wind'][day, :, :].ravel()
        u_day = var_grids['U'][day, :, :].ravel()
        v_day = var_grids['V'][day, :, :].ravel()

        # good_indices = ~np.isnan(sst_day) & \
        #                ~np.isnan(sss_day) & \
        #                ~np.isnan(seaice_day) & \
        #                ~np.isnan(uwind_day) & \
        #                ~np.isnan(vwind_day) & \
        #                ~np.isnan(u_day) & \
        #                ~np.isnan(v_day)
        #
        # X_day = np.column_stack([sst_day.ravel(),
        #                          sss_day.ravel(),
        #                          seaice_day.ravel(),
        #                          uwind_day.ravel(),
        #                          vwind_day.ravel(),
        #                          u_day.ravel(),
        #                          v_day.ravel()])

        if model_version == 'XGB1':
            good_indices = ~np.isnan(sst_day) & \
                           ~np.isnan(seaice_day) & \
                           ~np.isnan(uwind_day) & \
                           ~np.isnan(vwind_day) & \
                           ~np.isnan(u_day) & \
                           ~np.isnan(v_day)
            X_day = np.column_stack([sst_day.ravel(),
                                     seaice_day.ravel(),
                                     uwind_day.ravel(),
                                     vwind_day.ravel(),
                                     u_day.ravel(),
                                     v_day.ravel()])

        if model_version == 'XGB2':
            good_indices = ~np.isnan(sst_day) & \
                           ~np.isnan(sss_day) & \
                           ~np.isnan(seaice_day) & \
                           ~np.isnan(uwind_day) & \
                           ~np.isnan(vwind_day) & \
                           ~np.isnan(u_day) & \
                           ~np.isnan(v_day)
            X_day = np.column_stack([sst_day.ravel(),
                                     sss_day.ravel(),
                                     seaice_day.ravel(),
                                     uwind_day.ravel(),
                                     vwind_day.ravel(),
                                     u_day.ravel(),
                                     v_day.ravel()])

        Chl_predicted = model.predict(X_day)
        Chl_predicted = 10**Chl_predicted
        Chl_predicted[~good_indices]=0
        Chl_predicted = Chl_predicted.reshape(np.shape(var_grids['X']))

        # C = plt.pcolormesh(Chl_predicted)
        # plt.colorbar(C)
        # plt.show()

        Chl[day,:,:] = Chl_predicted

    if 'Chlorophyll_Modeled_' + model_version not in os.listdir(os.path.join(project_folder, '15km Interpolated')):
        os.mkdir(os.path.join(project_folder, '15km Interpolated', 'Chlorophyll_Modeled_' + model_version))

    output_file = os.path.join(project_folder, '15km Interpolated', 'Chlorophyll_Modeled_' + model_version,
                               'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')

    ds = nc4.Dataset(output_file, 'w')

    ds.createDimension('x', np.shape(var_grids['X'])[1])
    ds.createDimension('y', np.shape(var_grids['X'])[0])
    ds.createDimension('days', len(days))

    dvar = ds.createVariable('days', 'f4', ('days',))
    dvar[:] = days
    xvar = ds.createVariable('x', 'f4', ('x',))
    xvar[:] = var_grids['X'][0, :]
    xvar = ds.createVariable('y', 'f4', ('y',))
    xvar[:] = var_grids['Y'][:, 0]
    xvar = ds.createVariable('X', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['X']
    xvar = ds.createVariable('Y', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['Y']
    xvar = ds.createVariable('Depth', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['Depth']

    xvar = ds.createVariable('Chl_Modeled', 'f4', ('days', 'y', 'x'))
    xvar[:, :, :] = Chl

    ds.close()

def run_shap_analysis(project_dir, model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap_values.feature_names = ["SST", "SSS", "Sea Ice", "U Wind", "V Wind", "U", "V"]

    # need to figure out how to save this
    shap.plots.bar(shap_values)

    # fig = plt.figure()
    # plt.savefig(os.path.join(project_dir,'Figures','Remote Sensing Model','Chl_XGBoost_SHAP.png'))
    # plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology/Data'

model_version = 'XGB1'

year = 2025
month = 8
days = np.arange(1,32).tolist()

print(' - Reading in the var grid')
var_grids = read_all_data_subsets(project_folder, year, month)

print(' - Generating the training sets')
X_train, X_test, y_train, y_test, Loc_train, Loc_test = create_training_and_testing_sets(var_grids, year, month, days, model_version)

print(' - Generating the model')
model = generate_xgboost_model(X_train, y_train)

print(' - Plotting the model results')
plot_model_R2_results(project_folder, model_version,  model, X_train, X_test, y_train, y_test)
# plot_model_R2_results_with_distance(project_folder, var_grids['X'], var_grids['Y'], var_grids['Depth'],
#                                     model, X_train, X_test, y_train, y_test, Loc_train, Loc_test)

print(' - Generating the modeled results')
generate_modeled_Chl_dataset(project_folder, model_version, model, year, month, days, var_grids)

print(' - Running the SHAP Analysis')
# run_shap_analysis(project_folder, model, X_train, X_test)


