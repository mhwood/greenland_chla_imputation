
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

def read_data_distribution(project_folder, resolution, year_months):

    var_name = 'Chlorophyll'
    read_name = 'Chl'

    ###################################################################

    first_file = True

    for year, month in year_months:
        file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',var_name,str(year),
                                 f'{var_name}_{year}{month:02d}.nc')
        if os.path.exists(file_path):
            with nc4.Dataset(file_path, 'r') as ds:
                if var_name in ['Chlorophyll']:
                    var_data = ds.variables[read_name][:]
                    # print(np.size(var_data), np.sum(np.isnan(var_data)))
                    var_data = var_data[~np.isnan(var_data)]
                    if np.size(var_data)>0:
                        # var_data = np.array(var_data)
                        var_data = np.log10(var_data[var_data>0])
                        if np.any(np.isfinite(var_data)):

                            if first_file:
                                all_points = var_data
                            else:
                                all_points = np.concatenate([all_points,var_data])

    return(all_points)

def read_model_distribution(project_folder, resolution, year_months, model_version):

    var_name = 'Chlorophyll'
    read_name = 'Chl'

    ###################################################################

    first_file = True

    for year, month in year_months:
        file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                                 'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')
        if os.path.exists(file_path):
            with nc4.Dataset(file_path, 'r') as ds:
                if var_name in ['Chlorophyll']:
                    var_data = ds.variables[read_name][:]
                    # print(np.size(var_data), np.sum(np.isnan(var_data)))
                    var_data = var_data[~np.isnan(var_data)]
                    if np.size(var_data)>0:
                        # var_data = np.array(var_data)
                        var_data = np.log10(var_data[var_data>0])
                        if np.any(np.isfinite(var_data)):

                            if first_file:
                                all_points = var_data
                            else:
                                all_points = np.concatenate([all_points,var_data])

    return(all_points)

def plot_distribution_comparision(project_folder, model_version, resolution, data_points, model_points):

    fig = plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.hist(data_points,density=True, bins=100)
    plt.title(f'Data Distribution\nMean:{np.mean(data_points):.3f}, StD:{np.std(data_points):.3f}')
    plt.xlabel('log$_{10}$(Chl-a)')
    plt.ylabel('Normalized Count')

    plt.subplot(1, 2, 2)
    plt.hist(model_points, density=True, bins=100)
    plt.title(f'Model Distribution\nMean:{np.mean(model_points):.3f}, StD:{np.std(model_points):.3f}')
    plt.xlabel('log$_{10}$(Chl-a)')

    output_file = os.path.join(project_folder, 'Figures', 'Model Assessment',
                               f'{model_version}_model_vs_data_distribution.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'Unet2'
resolution = 15

start_year = 2019
start_month = 1

end_year = 2019
end_month = 12

year_months = generate_year_months(start_year, start_month, end_year, end_month)

data_points = read_data_distribution(project_folder, resolution, year_months)

model_points = read_model_distribution(project_folder, resolution, year_months, model_version)

plot_distribution_comparision(project_folder, model_version, resolution, data_points, model_points)















