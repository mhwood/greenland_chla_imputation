
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4


def read_timeseries_from_nc(project_folder, model_version, resolution):

    ds = nc4.Dataset(os.path.join(project_folder, 'Data', str(resolution)+'km Model', model_version,
                                  'Model Timeseries',
                                  'Chl Timeseries at Sample Locations.nc'))

    dec_yrs = ds.variables['time'][:]

    model_timeseries = {}
    obs_timeseries = {}

    for location in list(ds.groups.keys()):
        grp = ds.groups[location]
        model_timeseries[location] = np.column_stack([dec_yrs, grp.variables['Chl_Modeled']])
        obs_timeseries[location] = np.column_stack([dec_yrs, grp.variables['Chl_Obs']])

    ds.close()

    return(model_timeseries, obs_timeseries)


def plot_sample_timeseries(project_folder, model_version, resolution, model_timeseries, obs_timeseries):

    fig = plt.figure(figsize=(12,10))

    location_names = list(model_timeseries.keys())

    for ll, location in enumerate(location_names):
        plt.subplot(4,1,ll+1)
        plt.plot(model_timeseries[location][:,0], model_timeseries[location][:,1],'g-')
        # plt.plot(obs_timeseries[location][:,0], obs_timeseries[location][:,1],'kd')
        plt.title(location)

    plt.savefig(os.path.join(project_folder,'Figures','Timeseries',
                             'Modeled Chl at Sample Locations ('+model_version+', '+str(resolution)+'km).png'))
    plt.close(fig)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 15

model_timeseries, obs_timeseries = read_timeseries_from_nc(project_folder, model_version, resolution)

plot_sample_timeseries(project_folder, model_version, resolution, model_timeseries, obs_timeseries)