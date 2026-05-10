
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4


def compute_maps(project_folder, resolution, model_version, start_year, end_year, month='all'):

    first_file = True

    if month == 'all':
        start_month = 1
        end_month = 12
    else:
        start_month = month
        end_month = month

    for year in range(start_year, end_year + 1):
        for month in range(start_month, end_month + 1):
            print(' - Working on ' + str(year) + '/' + str(month))

            chl_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                               'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')
            ds = nc4.Dataset(chl_file)
            X = ds.variables['X'][:,:]
            Y = ds.variables['Y'][:,:]
            Chl_Model = ds.variables['Chl'][:,:,:]
            ds.close()

            chl_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                     'Chlorophyll', str(year), 'Chlorophyll_' + str(year) +'{:02d}'.format(month) +'.nc')
            ds = nc4.Dataset(chl_file)
            Chl_Obs = ds.variables['Chl'][:, :, :]
            ds.close()

            if first_file:
                first_file=False
                count_grid = np.zeros(np.shape(Chl_Model)[1:])
                R2_grid = np.zeros(np.shape(Chl_Model)[1:])
                sum_grid = np.zeros(np.shape(Chl_Model)[1:])

            for day in range(np.shape(Chl_Model)[0]):
                model_day = Chl_Model[day,:,:]
                obs_day = Chl_Obs[day,:,:]

                valid_mask = ~np.isnan(model_day) & ~np.isnan(obs_day)

                count_grid[valid_mask] += 1

                R2_grid[valid_mask] += (model_day[valid_mask] - obs_day[valid_mask])**2
                sum_grid[valid_mask] += obs_day[valid_mask]

    RMSE_grid = np.sqrt(R2_grid / count_grid)
    RMSE_grid[count_grid < 10] = np.nan

    mean_grid = sum_grid / count_grid

    return(mean_grid, RMSE_grid, count_grid, X, Y)

def save_fields_to_nc(project_folder, month, mean_grid, RMSE_grid, count_grid, X, Y):

    output_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,'Statistics',
                               model_version+'spatial_statistics_' + str(start_year) + '_' + str(end_year) + '_' + str(month) + '.nc')

    ds = nc4.Dataset(output_file, 'w', format='NETCDF4')

    ds.createDimension('X', X.shape[1])
    ds.createDimension('Y', Y.shape[0])

    X_var = ds.createVariable('X', 'f4', ('Y','X'))
    Y_var = ds.createVariable('Y', 'f4', ('Y','X'))
    mean_var = ds.createVariable('Mean_Chl', 'f4', ('Y','X'))
    RMSE_var = ds.createVariable('RMSE_Chl', 'f4', ('Y','X'))
    count_var = ds.createVariable('Count', 'i4', ('Y','X'))

    X_var[:,:] = X
    Y_var[:,:] = Y
    mean_var[:,:] = mean_grid
    RMSE_var[:,:] = RMSE_grid
    count_var[:,:] = count_grid

    ds.close()


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

start_year = 2006
end_year = 2009

for month in range(1,13):
    mean_grid, RMSE_grid, count_grid, X, Y = compute_maps(project_folder, resolution, model_version, start_year, end_year, month=month)
    save_fields_to_nc(project_folder, month, mean_grid, RMSE_grid, count_grid, X, Y)

mean_grid, RMSE_grid, count_grid, X, Y = compute_maps(project_folder, resolution, model_version, start_year, end_year, month='all')
save_fields_to_nc(project_folder, 'all', mean_grid, RMSE_grid, count_grid, X, Y)


