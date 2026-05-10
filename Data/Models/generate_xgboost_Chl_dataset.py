
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4

import xgboost as xgb
from sklearn.metrics import r2_score
import shap
from sklearn.model_selection import train_test_split

def load_model_checkpoint(project_folder, model_version, max_depth):
    model_file = os.path.join(project_folder,'Data', str(resolution)+'km Model',
                              model_version, 'Checkpoints', model_version+'_Model_Checkpoint_' + str(max_depth) + '.json')
    model = xgb.XGBRegressor()
    model.load_model(model_file)

    return model

def generate_year_months(start_year, start_month, end_year, end_month):
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            year_months.append((year, month))
    return year_months

def read_all_data_subset(project_folder, year_month, resolution):

    var_grids = {}

    year = year_month[0]
    month = year_month[1]

    for data_subset in ['SST', 'Wind', 'Velocity', 'Sea_Ice']:
        file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                 data_subset, str(year), data_subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')
        ds = nc4.Dataset(file_path)
        if data_subset=='SST':
            SST_grid = ds.variables['SST'][:,:,:]
            X_grid = ds.variables['X'][:,:]
            Y_grid = ds.variables['Y'][:, :]
            Depth_grid = ds.variables['Depth'][:, :]

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

            var_grids['X'] = X_grid_big
            var_grids['Y'] = Y_grid_big
            var_grids['DOY_sin'] = DOY_sin
            var_grids['DOY_cos'] = DOY_cos
            var_grids['Depth'] = Depth_grid
            var_grids['SST'] = SST_grid

        if data_subset=='SSS':
            SSS_grid = ds.variables['SSS'][:,:,:]
            var_grids['SSS'] = SSS_grid

        if data_subset=='Sea_Ice':
            Sea_ice_grid = ds.variables['seaice_fraction'][:,:,:]
            var_grids['Sea_Ice'] = Sea_ice_grid

        if data_subset=='Wind':
            U_wind_grid = ds.variables['U_wind'][:,:,:]
            V_wind_grid = ds.variables['V_wind'][:, :, :]
            var_grids['U_wind'] = U_wind_grid
            var_grids['V_wind'] = V_wind_grid

        if data_subset=='Velocity':
            U_grid = ds.variables['U'][:,:,:]
            V_grid = ds.variables['V'][:, :, :]
            var_grids['U'] = U_grid
            var_grids['V'] = V_grid

        if data_subset=='Chlorophyll':
            Chl_grid = ds.variables['Chl'][:,:,:]
            var_grids['Chl'] = Chl_grid

    return(var_grids)

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

def generate_modeled_Chl_dataset(project_folder, model_version, model, year_month, var_grids):

    year = year_month[0]
    month = year_month[1]
    days = np.shape(var_grids['SST'])[0]

    Chl = np.zeros((days, np.shape(var_grids['Depth'])[0], np.shape(var_grids['Depth'])[1]))

    Depth = var_grids['Depth'][:, :].ravel()

    for day in range(days):
        sst_day = var_grids['SST'][day, :, :].ravel()
        uwind_day = var_grids['U_wind'][day, :, :].ravel()
        vwind_day = var_grids['V_wind'][day, :, :].ravel()
        u_day = var_grids['U'][day, :, :].ravel()
        v_day = var_grids['V'][day, :, :].ravel()
        X_day = var_grids['X'][day, :, :].ravel()
        Y_day = var_grids['Y'][day, :, :].ravel()
        DOY_sin_day = var_grids['DOY_sin'][day, :, :].ravel()
        DOY_cos_day = var_grids['DOY_cos'][day, :, :].ravel()

        if model_version == 'XGB1':

            X = np.column_stack([sst_day.ravel(),
                                     uwind_day.ravel(),
                                     vwind_day.ravel(),
                                     u_day.ravel(),
                                     v_day.ravel(),
                                     DOY_sin_day.ravel(),
                                     DOY_cos_day.ravel(),
                                     X_day.ravel(),
                                     Y_day.ravel()])

        Chl_predicted = model.predict(X)
        Chl_predicted = 10**Chl_predicted

        # mask where depth is 0
        Chl_predicted[Depth == 0] = np.nan

        # mask where seaice is more than 0.15
        Chl_predicted[var_grids['Sea_Ice'][day, :, :].ravel() > 0.15] = np.nan

        Chl_predicted = Chl_predicted.reshape(np.shape(var_grids['Depth']))

        if year == 2006 and month == 1 and day == 0:
            C = plt.pcolormesh(Chl_predicted)
            plt.colorbar(C)
            plt.show()

        Chl[day,:,:] = Chl_predicted

    output_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                               'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')

    ds = nc4.Dataset(output_file, 'w')

    ds.createDimension('x', np.shape(var_grids['Depth'])[1])
    ds.createDimension('y', np.shape(var_grids['Depth'])[0])
    ds.createDimension('days', days)

    dvar = ds.createVariable('days', 'f4', ('days',))
    dvar[:] = np.arange(1, days + 1)
    xvar = ds.createVariable('x', 'f4', ('x',))
    xvar[:] = var_grids['X'][0, 0, :]
    xvar = ds.createVariable('y', 'f4', ('y',))
    xvar[:] = var_grids['Y'][0, :, 0]
    xvar = ds.createVariable('X', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['X'][0,:,:]
    xvar = ds.createVariable('Y', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['Y'][0,:,:]
    xvar = ds.createVariable('Depth', 'f4', ('y', 'x'))
    xvar[:, :] = var_grids['Depth']

    xvar = ds.createVariable('Chl', 'f4', ('days', 'y', 'x'))
    xvar[:, :, :] = Chl

    ds.close()



project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15
max_depth = 18

start_year = 2003
start_month = 1

end_year = 2020
end_month = 12

check_if_exists = False

print(' - Loading the model checkpoint' )
model = load_model_checkpoint(project_folder, model_version, max_depth)

print(' - Generating the monthly grids')
year_months = generate_year_months(start_year, start_month, end_year, end_month)

for year_month in year_months:

    generate_file = run_file_check(project_folder, resolution, model_version, year_month, check_if_exists)

    if generate_file:

        var_grids = read_all_data_subset(project_folder, year_month, resolution)

        generate_modeled_Chl_dataset(project_folder, model_version, model, year_month, var_grids)





