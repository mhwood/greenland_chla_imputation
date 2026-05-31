
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import datetime

def YMD_to_DecYr(year,month,day,hour=0,minute=0,second=0):
    date = datetime.datetime(year,month,day,hour,minute,second)
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    decimal_fraction = float(date.toordinal() - start) / year_length
    dec_yr = year+decimal_fraction
    return(dec_yr)

def read_timeseries_from_nc(project_folder, model_version, resolution, month):

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

    if month == 'August':
        month_num = 8
        days_in_month = 31

    years = np.arange(1982,2021,1)
    monthly_chl_timeseries = {}
    for location in list(ds.groups.keys()):
        monthly_chl_timeseries[location] = np.zeros_like(years).astype(float)
    for y in range(len(years)):
        year = years[y]
        min_dec_yr = YMD_to_DecYr(year, month_num, 1)
        max_dec_yr = YMD_to_DecYr(year, month_num, days_in_month)
        for location in model_timeseries:
            model_chl_in_month = model_timeseries[location][(model_timeseries[location][:,0] >= min_dec_yr) &
                                                            (model_timeseries[location][:,0] <= max_dec_yr), 1]
            # obs_chl_in_month = obs_timeseries[location][(obs_timeseries[location][:,0] >= min_dec_yr) &
            #                                             (obs_timeseries[location][:,0] <= max_dec_yr), 1]

            if len(model_chl_in_month) > 0:
                monthly_chl_timeseries[location][y] = np.nanmean(model_chl_in_month)
            else:
                monthly_chl_timeseries[location][y] = np.nan

    return(years, monthly_chl_timeseries)


def plot_sample_timeseries(project_folder, model_version, resolution, years, model_timeseries, month):

    fig = plt.figure(figsize=(12,10))

    location_names = list(model_timeseries.keys())

    for ll, location in enumerate(location_names):
        plt.subplot(4,1,ll+1)
        plt.plot(years, model_timeseries[location],'g-')
        plt.title(location)

    plt.savefig(os.path.join(project_folder,'Figures','Timeseries',
                             'Modeled Chl '+month+' at Sample Locations '+model_version+' '+str(resolution)+'km.png'))
    plt.close(fig)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 15

month = 'August'

years, model_timeseries = read_timeseries_from_nc(project_folder, model_version, resolution, month)

plot_sample_timeseries(project_folder, model_version, resolution, years, model_timeseries, month)