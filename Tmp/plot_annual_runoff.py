
import os
import numpy as np
import netCDF4 as nc4
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


data_dir = '/Volumes/CoOL/Data_Repository/Greenland/SMB/RU'

home_folder = os.path.expanduser('~')
project_folder = os.path.join(home_folder, 'Documents', 'Research',
                              'Projects', 'Greenland Chl Imputation')

start_year = 1980
end_year = 2019

runoff_timeseries = np.zeros((end_year - start_year + 1,2))

for year in range(start_year, end_year+1):
    print('Reading in year '+str(year))
    file_path = os.path.join(data_dir, 'MARv3_11_20km_daily_ERA_10km_' + str(year) + '.nc')

    ds = nc4.Dataset(file_path)
    runoff_timeseries[year-start_year,0] = year
    runoff_grid = ds.variables['RU'][:,:,:]
    area = ds.variables['AREA'][:,:]
    total_runoff = 0
    for i in range(runoff_grid.shape[0]):
        day_runoff = runoff_grid[i,:,:]
        day_runoff = day_runoff*area/1000 # Convert from mm water to m^3 water
        day_runoff=day_runoff[day_runoff>0]
        total_runoff += np.sum(day_runoff)
    runoff_timeseries[year-start_year,1] = total_runoff

    ds.close()

runoff_timeseries[:,1] = runoff_timeseries[:,1]/1e6

start_year_mean_1 = 1980
end_year_mean_1 = 1999
start_year_mean_2 = 2000
end_year_mean_2 = 2020

subset_1 = runoff_timeseries[(runoff_timeseries[:,0]>=start_year_mean_1) & (runoff_timeseries[:,0]<=end_year_mean_1),1]
subset_2 = runoff_timeseries[(runoff_timeseries[:,0]>=start_year_mean_2) & (runoff_timeseries[:,0]<=end_year_mean_2),1]

mean_1 = np.mean(subset_1)
mean_2 = np.mean(subset_2)

print(ttest_ind(subset_1, subset_2, equal_var=False))


plt.style.use('dark_background')
fig = plt.figure(figsize=(10,4))

plt.bar(runoff_timeseries[:,0], runoff_timeseries[:,1],color='white')

plt.plot([start_year_mean_1-0.5, end_year_mean_1+0.5], [mean_1, mean_1], '--', label='1980-1985 Mean', color='dodgerblue')
plt.plot([start_year_mean_2-0.5, end_year_mean_2+0.5], [mean_2, mean_2], '--', label='2085-2090 Mean', color='salmon')

plt.text(start_year_mean_1-0.5, mean_1+0.1*mean_1, f'{mean_1:.2f}x10$^6$ m$^3$', color='dodgerblue',ha='left')
plt.text(start_year_mean_2-0.5, mean_2+0.1*mean_2, f'{mean_2:.2f}x10$^6$ m$^3$', color='salmon',ha='left')

plt.ylabel('Total Runoff (10$^6$ m$^3$)')
plt.title('Total Annual Greenland Runoff')

plt.savefig(os.path.join(project_folder, 'Figures', 'Runoff', 'Total Annual Runoff.png'), dpi=300)
plt.close(fig)




