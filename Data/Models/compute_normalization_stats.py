


import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4


def generate_year_months(start_year, start_month, end_year, end_month):
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            year_months.append((year, month))
    return year_months

def read_variable_stats(project_folder, resolution, year_months, var_name):

    if var_name=='Chlorophyll':
        read_name = 'Chl'
    else:
        read_name = var_name

    ###################################################################
    # Compute the mean first

    if var_name in ['Chlorophyll','SST']:
        sum_for_mean = 0
    else:
        u_sum_for_mean = 0
        v_sum_for_mean = 0
    counter = 0

    stats_dict = {}

    for year, month in year_months:
        file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',var_name,str(year),
                                 f'{var_name}_{year}{month:02d}.nc')
        if os.path.exists(file_path):
            # print(file_path)
            with nc4.Dataset(file_path, 'r') as ds:
                if var_name in ['Chlorophyll']:
                    var_data = ds.variables[read_name][:]
                    # print(np.size(var_data), np.sum(np.isnan(var_data)))
                    var_data = var_data[~np.isnan(var_data)]
                    if np.size(var_data)>0:
                        # var_data = np.array(var_data)
                        var_data = np.log10(var_data[var_data>0])
                        if np.any(np.isfinite(var_data)):
                            sum_for_mean += np.sum(var_data)
                            counter += np.size(var_data)
                elif var_name in ['SST']:
                    var_data = ds.variables[read_name][:]
                    var_data = var_data[~np.isnan(var_data)]
                    sum_for_mean += np.sum(var_data)
                    counter += np.size(var_data)
                elif var_name=='Wind':
                    u_var_data = ds.variables['U_wind'][:]
                    u_var_data = u_var_data[~np.isnan(u_var_data)]
                    u_sum_for_mean += np.sum(u_var_data)
                    v_var_data = ds.variables['V_wind'][:]
                    v_var_data = v_var_data[~np.isnan(v_var_data)]
                    v_sum_for_mean += np.sum(v_var_data)
                    counter += np.size(u_var_data)

    if var_name in ['Chlorophyll', 'SST']:
        stat_mean = sum_for_mean/counter
        stats_dict[var_name] = {'mean':stat_mean}
    elif var_name=='Wind':
        u_stat_mean = u_sum_for_mean / counter
        stats_dict['U_wind'] = {'mean': u_stat_mean}
        v_stat_mean = v_sum_for_mean / counter
        stats_dict['V_wind'] = {'mean': v_stat_mean}

    ###################################################################
    # Compute the std second

    if var_name in ['Chlorophyll','SST']:
        sum_of_squares_for_std = 0
    else:
        u_sum_of_squares_for_std = 0
        v_sum_of_squares_for_std = 0
    counter = 0

    for year, month in year_months:
        file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',var_name,str(year),
                                 f'{var_name}_{year}{month:02d}.nc')
        
        if os.path.exists(file_path):
            # print(file_path)
            print('     - Reading '+var_name+' in '+str(year)+'/'+str(month))
            with nc4.Dataset(file_path, 'r') as ds:
                if var_name in ['Chlorophyll']:
                    var_data = ds.variables[read_name][:]
                    var_data = var_data[~np.isnan(var_data)]
                    if np.size(var_data) > 0:
                        var_data = np.log10(var_data[var_data > 0])
                        if np.any(np.isfinite(var_data)):
                            sum_of_squares_for_std += np.sum((var_data-stat_mean)**2)
                            counter += np.size(var_data)
                elif var_name in ['SST']:
                    var_data = ds.variables[read_name][:]
                    var_data = var_data[~np.isnan(var_data)]
                    sum_of_squares_for_std += np.sum((var_data-stat_mean)**2)
                    counter += np.size(var_data)
                elif var_name=='Wind':
                    u_var_data = ds.variables['U_wind'][:]
                    u_var_data = u_var_data[~np.isnan(u_var_data)]
                    u_sum_of_squares_for_std += np.sum((u_var_data-u_stat_mean) ** 2)
                    v_var_data = ds.variables['V_wind'][:]
                    v_var_data = v_var_data[~np.isnan(v_var_data)]
                    v_sum_of_squares_for_std += np.sum((v_var_data-v_stat_mean) ** 2)
                    counter += np.size(u_var_data)

    if var_name in ['Chlorophyll', 'SST']:
        stat_std = np.sqrt(sum_of_squares_for_std/counter)
        stats_dict[var_name] = {'mean':stat_mean, 'std':stat_std}
    elif var_name=='Wind':
        u_stat_std = np.sqrt(u_sum_of_squares_for_std / counter)
        stats_dict['U_wind']['std'] = u_stat_std
        v_stat_std = np.sqrt(v_sum_of_squares_for_std / counter)
        stats_dict['V_wind']['std'] = v_stat_std

    return(stats_dict)

def compute_location_stats(project_folder, resolution):
    output_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                               'Greenland_domain_' + str(resolution) + 'km.nc')

    ds = nc4.Dataset(output_file)
    Lon = ds.variables['Lon'][:, :]
    Lat = ds.variables['Lat'][:, :]
    ds.close()

    stats_dict = {}
    stats_dict['Lon'] = {'mean':np.mean(Lon), 'std':np.std(Lon)}
    stats_dict['Lat'] = {'mean': np.mean(Lat), 'std': np.std(Lat)}


    return stats_dict

def compute_DOY_stats(year_months):

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # first compute how many days there are
    n_days = 0
    for year_month in year_months:
        year = year_month[0]
        month = year_month[1]

        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            num_days = 29
        else:
            num_days = days_in_month[month - 1]
        n_days += num_days

    # then fill in the DOYs in a big array
    doys = np.zeros((n_days,), dtype=np.float32)
    days_counted = 0

    for year_month in year_months:
        year = year_month[0]
        month = year_month[1]

        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            num_days = 29
        else:
            num_days = days_in_month[month - 1]

        if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            days_in_year = 366
        else:
            days_in_year = 365

        start_day = np.sum(days_in_month[:month - 1]) + 1
        end_day = start_day + num_days - 1
        doys[days_counted:days_counted + num_days] = np.arange(start_day, end_day + 1) / days_in_year
        days_counted += num_days

    doy_sin = np.sin(2 * np.pi * doys)  # already divided by days in year to get 0-1 range above
    doy_cos = np.cos(2 * np.pi * doys)

    stats_dict = {}
    stats_dict['DOY_sin'] = {'mean':np.mean(doy_sin), 'std':np.mean(doy_sin)}
    stats_dict['DOY_cos'] = {'mean': np.mean(doy_cos), 'std': np.mean(doy_cos)}
    return(stats_dict)

def write_normalization_stats_to_file(project_folder, resolution, normalization_stats):
    stats_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                              str(resolution)+'km_normalization_stats.txt')
    print(stats_file)
    with open(stats_file, 'w') as f:
        for var_name, stats in normalization_stats.items():
            f.write(f"{var_name}: mean={stats['mean']}, std={stats['std']}\n")

home_folder = os.path.expanduser('~')
project_folder = home_folder+'/Documents/Research/Projects/Greenland Chl Imputation'

resolution = 1

if resolution == 15:
    data_folder = project_folder
else:
    # data_folder = '/Volumes/CoOL/Projects/Greenland Chl Imputation'
    data_folder = '/Volumes/ilulissat/Research/Projects/Greenland Chl Imputation'


start_year = 2001
start_month = 1

end_year = 2001
end_month = 12

year_months = generate_year_months(start_year, start_month, end_year, end_month)

var_names = ['Chlorophyll', 'SST', 'Wind']#, 'DOY_sin','DOY_cos']

normalization_stats = {}

for var_name in var_names:
    print(' - Reading in the stats for '+var_name)
    stats_dict = read_variable_stats(data_folder, resolution, year_months, var_name)
    if var_name in ['Chlorophyll','SST']:
        normalization_stats[var_name] = stats_dict[var_name]
        print('       - Mean:',stats_dict[var_name]['mean'],'- Std:',stats_dict[var_name]['std'])
    elif var_name == 'Wind':
        normalization_stats['U_wind'] = stats_dict['U_wind']
        normalization_stats['V_wind'] = stats_dict['V_wind']
        print('       - Mean:', stats_dict['U_wind']['mean'], '- Std:', stats_dict['U_wind']['std'])
        print('       - Mean:', stats_dict['V_wind']['mean'], '- Std:', stats_dict['V_wind']['std'])

location_stats = compute_location_stats(project_folder, resolution)
normalization_stats['Lon'] = location_stats['Lon']
normalization_stats['Lat'] = location_stats['Lat']

doy_stats = compute_DOY_stats(year_months)
normalization_stats['DOY_sin'] = doy_stats['DOY_sin']
normalization_stats['DOY_cos'] = doy_stats['DOY_cos']

write_normalization_stats_to_file(project_folder, resolution, normalization_stats)






