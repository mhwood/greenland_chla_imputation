

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from pyproj import Transformer
import datetime

def reproject_polygon(polygon_array,inputCRS,outputCRS,x_column=0,y_column=1,run_test = True):

    transformer = Transformer.from_crs('EPSG:' + str(inputCRS), 'EPSG:' + str(outputCRS))

    # There seems to be a serious problem with pyproj
    # The x's and y's are mixed up for these transformations
    #       For 4326->3413, you put in (y,x) and get out (x,y)
    #       Foe 3413->4326, you put in (x,y) and get out (y,x)
    # Safest to run check here to ensure things are outputting as expected with future iterations of pyproj

    if inputCRS == 4326 and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,y_column], polygon_array[:,x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    elif inputCRS == 3413 and outputCRS == 4326:
        y2, x2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    elif str(inputCRS)[:3] == '326' and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,y_column], polygon_array[:,x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
        run_test = False
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon

def reproject_locations(locations):

    points = []
    for location in locations:
        points.append(locations[location])
    points = np.array(points)

    points = reproject_polygon(points, 4326, 3413)
    return(points)

def YMD_to_DecYr(year,month,day,hour=0,minute=0,second=0):
    date = datetime.datetime(year,month,day,hour,minute,second)
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    decimal_fraction = float(date.toordinal() - start) / year_length
    dec_yr = year+decimal_fraction
    return(dec_yr)

def generate_year_months(start_year, start_month, end_year, end_month):
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            year_months.append((year, month))
    return year_months

def read_Chl_timeseries(project_folder, locations, points, year_months, infill_with_obs = True):

    first_file = True

    location_names = list(locations.keys())

    all_model_timeseries = {}
    all_obs_timeseries = {}

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    for year_month in year_months:
        year = year_month[0]
        month = year_month[1]
        print(' - Working on ' + str(year) + '/' + str(month))

        chl_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                           'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')
        ds = nc4.Dataset(chl_file)
        X = ds.variables['X'][:,:]
        Y = ds.variables['Y'][:,:]
        Chl_Model = ds.variables['Chl'][:,:,:]
        ds.close()

        nan_chl = False
        if year == 1986 and month >= 7:
            nan_chl = True
        if year > 1986 and year < 1997:
            nan_chl = True
        if year == 1997 and month <= 8:
            nan_chl = True

        if nan_chl:
            print('    - No Chlorophyll data for', year, month, '- filling with NaNs')
            n_days_month = days_in_month[month - 1]
            if year % 4 == 0 and month == 2:
                n_days_month = 29
            Chl_Obs = np.full((n_days_month, 184, 103),
                               np.nan, dtype=np.float32)
        else:
            chl_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                     'Chlorophyll', str(year), 'Chlorophyll_' + str(year) +'{:02d}'.format(month) +'.nc')
            ds = nc4.Dataset(chl_file)
            Chl_Obs = ds.variables['Chl'][:, :, :]
            ds.close()

        if infill_with_obs:
            model_nan_mask = np.isnan(Chl_Model)
            obs_nan_mask = np.isnan(Chl_Obs)

            # if the model is NaN but the obs is not, fill in with the obs
            infill_mask = model_nan_mask & ~obs_nan_mask

            Chl_Model[infill_mask] = Chl_Obs[infill_mask]

        days = np.arange(1,np.shape(Chl_Model)[0]+1).tolist()
        dec_yrs = [YMD_to_DecYr(year,month,day) for day in days]

        for ll in range(np.shape(points)[0]):
            x = points[ll,0]
            y = points[ll,1]
            row = np.argmin(np.abs(Y[:,0]-y))
            col = np.argmin(np.abs(X[0,:]-x))
            # print(row,col)
            month_timeseries = np.column_stack([dec_yrs,Chl_Model[:,row,col]])
            obs_timeseries = np.column_stack([dec_yrs,Chl_Obs[:,row,col]])

            if first_file:
                all_model_timeseries[location_names[ll]] = month_timeseries
                all_obs_timeseries[location_names[ll]] = obs_timeseries
            else:
                all_model_timeseries[location_names[ll]] = np.vstack([all_model_timeseries[location_names[ll]],
                                                                month_timeseries])
                all_obs_timeseries[location_names[ll]] = np.vstack([all_obs_timeseries[location_names[ll]],
                                                                      obs_timeseries])

        if first_file:
            first_file=False

    return(all_model_timeseries, all_obs_timeseries)

def save_timeseries_to_nc(project_folder, resolution, model_version, locations, model_timeseries, obs_timeseries):

    if 'Model Timeseries' not in os.listdir(os.path.join(project_folder,'Data',str(resolution)+'km Model',model_version)):
        os.mkdir(os.path.join(project_folder,'Data',str(resolution)+'km Model',model_version,'Model Timeseries'))

    ds = nc4.Dataset(os.path.join(project_folder,'Data',str(resolution)+'km Model',model_version,'Model Timeseries',
                                  'Chl Timeseries at Sample Locations.nc'),'w')

    ds.createDimension('time', np.shape(model_timeseries[list(model_timeseries.keys())[0]])[0])

    dvar = ds.createVariable('time', 'f4', ('time',))
    dvar[:] = model_timeseries[list(model_timeseries.keys())[0]][:,0]

    for location in list(model_timeseries.keys()):
        grp = ds.createGroup(location)
        chl_var = grp.createVariable('Chl_Modeled', 'f4', ('time',))
        chl_var[:] = model_timeseries[location][:,1]
        chl_var = grp.createVariable('Chl_Obs', 'f4', ('time',))
        chl_var[:] = obs_timeseries[location][:, 1]
        grp.longitude = locations[location][0]
        grp.latitude = locations[location][1]

    ds.close()

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 15

start_year = 1982
start_month = 1

end_year = 2020
end_month = 12

locations = {'Disko Bay':(-55.579, 68.747),
            'Melville Bay':(-60.955,75.465),
            'Sermilik Fjord':(-38.086, 65.239),
             'Labrador Sea':(-52.09,60.83)}

points = reproject_locations(locations)

year_months = generate_year_months(start_year, start_month, end_year, end_month)

model_timeseries, obs_timeseries = read_Chl_timeseries(project_folder, locations, points, year_months, infill_with_obs = False)

save_timeseries_to_nc(project_folder, resolution, model_version, locations, model_timeseries, obs_timeseries)



