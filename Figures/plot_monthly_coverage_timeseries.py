
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

def read_daily_Chl_coverage_timeseries(project_folder):

    first_file = True

    dec_yrs = []
    coverage_fraction = []

    for year in range(2025, 2026):
        for month in range(8, 9):
            print(' - Working on ' + str(year) + '/' + str(month))

            sif_file = os.path.join(project_folder, 'Data', '15km Interpolated', 'Sea_Ice',
                               'Sea_Ice_' + str(year) + '{:02d}'.format(month) + '.nc')
            ds = nc4.Dataset(sif_file)
            SIF = ds.variables['seaice_fraction'][:,:,:]
            Depth = ds.variables['Depth'][:,:]
            ds.close()

            chl_file = os.path.join(project_folder, 'Data', '15km Interpolated', 'Chlorophyll',
                                    'Chlorophyll_' + str(year) + '{:02d}'.format(month) + '.nc')
            ds = nc4.Dataset(chl_file)
            Chl = ds.variables['Chl'][:, :, :]
            ds.close()

            for day in range(np.shape(Chl)[0]):
                dec_yrs.append(YMD_to_DecYr(year,month,day+1))
                open_indices = np.logical_and(Depth>0, SIF[day,:,:]<0.15)
                Chl_day = Chl[day,:,:]
                fraction = np.sum(~np.isnan(Chl_day[open_indices]))/np.sum(open_indices)
                coverage_fraction.append(fraction)
                print(np.sum(~np.isnan(Chl_day[open_indices])), np.sum(open_indices),np.size(SIF[day,:,:]))

    return(dec_yrs, coverage_fraction)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology'

dec_yrs, coverage_fraction = read_daily_Chl_coverage_timeseries(project_folder)

plt.plot(dec_yrs, coverage_fraction)
plt.show()