import netCDF4 as nc4
import numpy as np
import xarray as xr



folder= '/Volumes/CoOL/Data_Repository/Greenland/SMB'
file_path = folder+'/MARv3_11_20km_daily_ERA_10km_1979_2019.nc'

print('Reading in the big file')
# ds=nc4.Dataset(file_path)
# time = ds.variables['TIME'][:]
# Lon = ds.variables['LON'][:,:]
# Lat = ds.variables['LAT'][:,:]
# SMB = ds.variables['SMB']#[:, :, :]
# SMB = np.array(SMB)
# ds.close()

year = 1979
ds = xr.open_dataset(file_path)
ds.drop_vars(['SECTOR', 'SECTOR1_1', 'ATMLAY3_3', 'SH', 'SOL', 'SRF', 'MSK', 'FRV', 'VEG',# 'AREA',
              'SF', 'RF', 'SMB', 'ME', 'SU', 'TT', 'ST2', 'SWD', 'LWD', 'SHF', 'LHF', 'AL2', 'SHSN3', 'SHSN0'])

ds = ds[['TIME','LON','LAT','AREA','RU']]
print(ds)


for year in np.arange(1979,2020).tolist():
    print('Subsetting to year ' + str(year))
    yr_ds = ds.sel(TIME=slice(str(year)+'-01-01', str(year)+'-12-31'))
    yr_ds.to_netcdf(folder+'/RU/MARv3_11_20km_daily_ERA_10km_'+str(year)+'.nc')