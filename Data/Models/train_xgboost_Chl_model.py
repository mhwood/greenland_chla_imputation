
import os
import numpy as np
import netCDF4 as nc4
import xgboost as xgb
from sklearn.model_selection import train_test_split

def generate_year_months(start_year, start_month, end_year, end_month):
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            year_months.append((year, month))
    return year_months

def read_all_data_subsets(project_folder, year_months, resolution):

    var_grids = {}

    grid_started = False

    for year_month in year_months:
        year = year_month[0]
        month = year_month[1]

        for data_subset in ['SST', 'Sea_Ice', 'Wind', 'Velocity', 'Chlorophyll']:
            file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                     data_subset, str(year), data_subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')
            ds = nc4.Dataset(file_path)
            if data_subset=='SST':
                SST_grid = ds.variables['SST'][:,:,:]
                X_grid = ds.variables['X'][:,:]
                Y_grid = ds.variables['Y'][:, :]
                # Depth_grid = ds.variables['Depth'][:, :]

                # stack X in to a 3D grid to match the dimensions of the other variables
                X_grid_big = np.zeros((np.shape(SST_grid)[0], np.shape(X_grid)[0], np.shape(X_grid)[1]))
                Y_grid_big = np.zeros((np.shape(SST_grid)[0], np.shape(Y_grid)[0], np.shape(Y_grid)[1]))
                for day in range(np.shape(SST_grid)[0]):
                    X_grid_big[day,:,:] = X_grid
                    Y_grid_big[day,:,:] = Y_grid

                DOY_sin = np.zeros((np.shape(SST_grid)[0], np.shape(X_grid)[0], np.shape(X_grid)[1]))
                DOY_cos = np.zeros((np.shape(SST_grid)[0], np.shape(X_grid)[0], np.shape(X_grid)[1]))
                for day in range(np.shape(SST_grid)[0]):
                    DOY_sin[day,:,:] = np.sin(2*np.pi*day/np.shape(SST_grid)[0])
                    DOY_cos[day,:,:] = np.cos(2*np.pi*day/np.shape(SST_grid)[0])

                if not grid_started:
                    var_grids['X'] = X_grid_big
                    var_grids['Y'] = Y_grid_big
                    var_grids['DOY_sin'] = DOY_sin
                    var_grids['DOY_cos'] = DOY_cos
                    # var_grids['Depth'] = Depth_grid
                    var_grids['SST'] = SST_grid
                else:
                    var_grids['SST'] = np.concatenate((var_grids['SST'], SST_grid), axis=0)
                    var_grids['X'] = np.concatenate((var_grids['X'], X_grid_big), axis=0)
                    var_grids['Y'] = np.concatenate((var_grids['Y'], Y_grid_big), axis=0)
                    var_grids['DOY_sin'] = np.concatenate((var_grids['DOY_sin'], DOY_sin), axis=0)
                    var_grids['DOY_cos'] = np.concatenate((var_grids['DOY_cos'], DOY_cos), axis=0)
                # print('SST',np.nanmin(var_grids['SST']), np.nanmax(var_grids['SST']))
            if data_subset=='SSS':
                SSS_grid = ds.variables['SSS'][:,:,:]
                if not grid_started:
                    var_grids['SSS'] = SSS_grid
                else:
                    var_grids['SSS'] = np.concatenate((var_grids['SSS'], SSS_grid), axis=0)
                # print('SSS',np.nanmin(var_grids['SSS']), np.nanmax(var_grids['SSS']))
            if data_subset=='Sea_Ice':
                Sea_ice_grid = ds.variables['seaice_fraction'][:,:,:]
                if not grid_started:
                    var_grids['Sea_Ice'] = Sea_ice_grid
                else:
                    var_grids['Sea_Ice'] = np.concatenate((var_grids['Sea_Ice'], Sea_ice_grid), axis=0)
            if data_subset=='Wind':
                U_wind_grid = ds.variables['U_wind'][:,:,:]
                V_wind_grid = ds.variables['V_wind'][:, :, :]
                if not grid_started:
                    var_grids['U_wind'] = U_wind_grid
                    var_grids['V_wind'] = V_wind_grid
                else:
                    var_grids['U_wind'] = np.concatenate((var_grids['U_wind'], U_wind_grid), axis=0)
                    var_grids['V_wind'] = np.concatenate((var_grids['V_wind'], V_wind_grid), axis=0)
                # print('U_wind',np.nanmin(var_grids['U_wind']), np.nanmax(var_grids['U_wind']))
            if data_subset=='Velocity':
                U_grid = ds.variables['U'][:,:,:]
                V_grid = ds.variables['V'][:, :, :]
                if not grid_started:
                    var_grids['U'] = U_grid
                    var_grids['V'] = V_grid
                else:
                    var_grids['U'] = np.concatenate((var_grids['U'], U_grid), axis=0)
                    var_grids['V'] = np.concatenate((var_grids['V'], V_grid), axis=0)
                # print('V', np.nanmin(var_grids['V']), np.nanmax(var_grids['V']))
            if data_subset=='Chlorophyll':
                Chl_grid = ds.variables['Chl'][:,:,:]
                if not grid_started:
                    var_grids['Chl'] = Chl_grid
                else:
                    var_grids['Chl'] = np.concatenate((var_grids['Chl'], Chl_grid), axis=0)

        if not grid_started:
            grid_started = True


    return(var_grids)

def create_training_and_testing_sets(var_grids, model_version):

    days = np.shape(var_grids['SST'])[0]
    print('    - Identified '+str(days)+ ' days of data for training/testing')

    all_Xs = []
    all_ys = []
    for day in range(days):
        chl_day = np.log10(var_grids['Chl'][day, :, :]).ravel()
        sst_day = var_grids['SST'][day, :, :].ravel()
        # sss_day = var_grids['SSS'][day, :, :].ravel()
        seaice_day = var_grids['Sea_Ice'][day, :, :].ravel()
        uwind_day = var_grids['U_wind'][day, :, :].ravel()
        vwind_day = var_grids['V_wind'][day, :, :].ravel()
        u_day = var_grids['U'][day, :, :].ravel()
        v_day = var_grids['V'][day, :, :].ravel()
        X_day = var_grids['X'][day, :, :].ravel()
        Y_day = var_grids['Y'][day, :, :].ravel()
        DOY_sin_day = var_grids['DOY_sin'][day, :, :].ravel()
        DOY_cos_day = var_grids['DOY_cos'][day, :, :].ravel()


        if model_version=='XGB1':
            good_indices = ~np.isnan(chl_day) & \
                           ~np.isnan(sst_day) & \
                           ~np.isnan(seaice_day) & \
                           ~np.isnan(uwind_day) & \
                           ~np.isnan(vwind_day) & \
                           ~np.isnan(u_day) & \
                           ~np.isnan(v_day)
            seaice_indices = np.logical_and(seaice_day>=0, seaice_day<=0.15)
            X_sub = np.column_stack([sst_day.ravel(),
                                     uwind_day.ravel(),
                                     vwind_day.ravel(),
                                     u_day.ravel(),
                                     v_day.ravel(),
                                     DOY_sin_day.ravel(),
                                     DOY_cos_day.ravel(),
                                     X_day.ravel(),
                                     Y_day.ravel()])

        y_sub = chl_day.ravel().reshape((np.size(chl_day), 1))

        model_indices = good_indices & seaice_indices
        X_sub = X_sub[model_indices, :]
        y_sub = y_sub[model_indices, :]

        all_Xs.append(X_sub)
        all_ys.append(y_sub)

    X = np.vstack(all_Xs)
    y = np.vstack(all_ys)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

    # Loc_train = X_train[:, :3]
    # Loc_test = X_test[:, :3]
    #
    # X_train = X_train[:,3:]
    # X_test = X_test[:,3:]

    return(X_train, X_test, y_train, y_test)

def save_training_testing_to_nc(project_folder, resolution, model_version, X_train, X_test, y_train, y_test):
    if str(resolution)+'km Model' not in os.listdir(os.path.join(project_folder,'Data')):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model'))

    if model_version not in os.listdir(os.path.join(project_folder,'Data', str(resolution)+'km Model')):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version))

    ckpt_path = os.path.join(project_folder,'Data', str(resolution)+'km Model', model_version, model_version+'_Training_Testing_Sets.nc')

    ds=nc4.Dataset(ckpt_path, 'w', format='NETCDF4')

    ds.createDimension('train_samples', X_train.shape[0])
    ds.createDimension('test_samples', X_test.shape[0])
    ds.createDimension('features', X_train.shape[1])

    X_train_var = ds.createVariable('X_train', 'f4', ('train_samples', 'features'))
    y_train_var = ds.createVariable('y_train', 'f4', ('train_samples',))
    X_test_var = ds.createVariable('X_test', 'f4', ('test_samples', 'features'))
    y_test_var = ds.createVariable('y_test', 'f4', ('test_samples',))

    X_train_var[:,:] = X_train
    y_train_var[:] = y_train[:,0]
    X_test_var[:,:] = X_test
    y_test_var[:] = y_test[:,0]

    ds.close()

def generate_xgboost_model(X_train, y_train, resolution, model_version, max_depth):
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=51
    )
    model.fit(X_train, y_train)

    if str(resolution)+'km Model' not in os.listdir(os.path.join(project_folder,'Data')):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model'))

    if model_version not in os.listdir(os.path.join(project_folder,'Data', str(resolution)+'km Model')):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version))

    if 'Checkpoints' not in os.listdir(os.path.join(project_folder,'Data', str(resolution)+'km Model', model_version)):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, 'Checkpoints'))

    ckpt_path = os.path.join(project_folder,'Data', str(resolution)+'km Model', model_version, 'Checkpoints',
                             model_version+'_Model_Checkpoint_'+str(max_depth)+'.json')
    model.save_model(ckpt_path)

    return(model)

def save_model_predictions_to_nc(project_folder, resolution, model_version,
                                 max_depth, y_train_pred, y_test_pred):

    if str(resolution)+'km Model' not in os.listdir(os.path.join(project_folder,'Data')):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model'))

    if model_version not in os.listdir(os.path.join(project_folder,'Data', str(resolution)+'km Model')):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version))

    if 'Predictions' not in os.listdir(os.path.join(project_folder,'Data', str(resolution)+'km Model', model_version)):
        os.mkdir(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version, 'Predictions'))

    ds = nc4.Dataset(os.path.join(project_folder,'Data', str(resolution)+'km Model', model_version, 'Predictions',
                                    model_version+'_Predictions_'+str(max_depth)+'.nc'), 'w', format='NETCDF4')

    ds.createDimension('train_samples', len(y_train_pred))
    ds.createDimension('test_samples', len(y_test_pred))

    y_train_pred_var = ds.createVariable('y_train_pred', 'f4', ('train_samples',))
    y_test_pred_var = ds.createVariable('y_test_pred', 'f4', ('test_samples',))

    y_train_pred_var[:] = y_train_pred
    y_test_pred_var[:] = y_test_pred

    ds.close()




project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

start_year = 1998
start_month = 1

end_year = 2020
end_month = 12

year_months = generate_year_months(start_year, start_month, end_year, end_month)

print(' - Reading in the monthly datasets')
var_grids = read_all_data_subsets(project_folder, year_months, resolution)

print(' - Generating the training sets')
X_train, X_test, y_train, y_test = \
    create_training_and_testing_sets(var_grids, model_version)

print('    - Training data shape: ',np.shape(X_train))
print('    - Testing data shape: ', np.shape(X_test))

save_training_testing_to_nc(project_folder, resolution, model_version, X_train, X_test, y_train, y_test)

print(' - Generating the model')
for max_depth in range(20,21):
    model = generate_xgboost_model(X_train, y_train, resolution, model_version, max_depth)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    save_model_predictions_to_nc(project_folder, resolution, model_version, max_depth, y_train_pred, y_test_pred)



