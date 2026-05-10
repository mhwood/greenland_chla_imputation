
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4


def read_timeseries_from_nc(project_folder):

    ds = nc4.Dataset(os.path.join(project_folder, 'Data', 'Model Timeseries',
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


def plot_sample_timeseries(project_folder, model_timeseries, obs_timeseries):

    fig = plt.figure(figsize=(8,8))

    location_names = list(model_timeseries.keys())

    for ll, location in enumerate(location_names):
        plt.subplot(3,1,ll+1)
        plt.plot(model_timeseries[location][:,0], model_timeseries[location][:,1],'g-')
        plt.plot(obs_timeseries[location][:,0], obs_timeseries[location][:,1],'kd')
        plt.title(location)

    plt.savefig(os.path.join(project_folder,'Figures','Remote Sensing Model',
                             'Modeled XGBoost Chl vs Obs.png'))
    plt.close(fig)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology'

model_timeseries, obs_timeseries = read_timeseries_from_nc(project_folder)

plot_sample_timeseries(project_folder, model_timeseries, obs_timeseries)