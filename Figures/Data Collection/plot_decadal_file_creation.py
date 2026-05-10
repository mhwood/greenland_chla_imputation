
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_expected_year_months(var_name, start_year, end_year):
    year_months = []
    if var_name=='Chlorophyll':
        sst_start_year = 1978
        sst_start_month = 10
        sst_end_year = 2025
        sst_end_month = 12
    if var_name=='SST':
        sst_start_year = 1981
        sst_start_month = 1
        sst_end_year = 2025
        sst_end_month = 12
    if var_name=='Sea_Ice':
        sst_start_year = 1975
        sst_start_month = 1
        sst_end_year = 2025
        sst_end_month = 12
    if var_name=='Velocity':
        sst_start_year = 1993
        sst_start_month = 1
        sst_end_year = 2025
        sst_end_month = 12
    if var_name=='Wind':
        sst_start_year = 1975
        sst_start_month = 1
        sst_end_year = 2025
        sst_end_month = 12

    for year in range(start_year, end_year):
        if year == sst_start_year:
            start_month = sst_start_month
            end_month = 12
            for month in range(start_month, end_month + 1):
                year_months.append((year, month))
        elif year == sst_end_year:
            start_month = 1
            end_month = sst_end_month
            for month in range(start_month, end_month + 1):
                year_months.append((year, month))
        elif year>sst_start_year and year<sst_end_year:
            start_month = 1
            end_month = 12
            for month in range(start_month, end_month + 1):
                year_months.append((year, month))

    if var_name=='Chlorophyll':
        for year in range(1987,1997):
            for month in range(1, 13):
                idx = year_months.index((year, month))
                year_months.pop(idx)
        for month in range(1, 10):
            idx = year_months.index((1997, month))
            year_months.pop(idx)
        for month in range(7, 13):
            idx = year_months.index((1986, month))
            year_months.pop(idx)

    return(year_months)

def check_if_files_exist(project_dir, var_name, resolution, year_months):
    files_present = []
    for year, month in year_months:
        file_name = f'{var_name}_{year}{month:02d}.nc'
        file_path = os.path.join(project_dir, 'Data', str(resolution)+'km Interpolated', var_name, str(year), file_name)

        if not os.path.isfile(file_path):
            files_present.append(False)
        else:
            files_present.append(True)

    return files_present


def plot_file_existence(project_folder, resolution, var_names, all_year_months, all_files_present):

    color_dict = {'Chlorophyll':'green', 'SST':'red',
                  'Sea_Ice':'grey', 'Velocity':'blue',
                  'Wind':'purple','All':'black'}

    fig = plt.figure(figsize=(15, 10))

    for start_year in range(1975, 2025, 10):

        plt.subplot(5, 1, (start_year - 1975) // 10 + 1)

        for v, var_name in enumerate(var_names):
            year_months = all_year_months[v]
            files_present = all_files_present[v]

            for year in range(start_year, start_year + 10):
                for month in range(1, 13):
                    if (year, month) in year_months:
                        idx = year_months.index((year, month))
                        # print((year,month), files_present[idx])
                        plt.gca().add_patch(Rectangle((year + (month-1) / 12, v + 0.1), 1 / 12, 0.8, facecolor='none', edgecolor='black'))
                        if files_present[idx]:
                            plt.gca().add_patch(Rectangle((year+(month-1)/12, v+0.1), 1/12, 0.8, color=color_dict[var_name], alpha=0.5))
                            # print((year-start_year)*12+month - 0.5, v)

        plt.ylim(0, len(var_names))
        if start_year == 1975:
            plt.title(f'File Existence for {resolution}km Resolution Model')
        plt.xlim(start_year, start_year + 10)
        y_ticks = np.arange(0.5, len(var_names) + 0.5, 1)
        plt.yticks(y_ticks, var_names)

    output_file = os.path.join(project_folder, 'Figures','Data Collection', f'{resolution}km_File_Status.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

home_folder = os.path.expanduser('~')

project_folder = os.path.join(home_folder, 'Documents', 'Research', 'Projects', 'Greenland Chl Imputation')

resolution = 15

start_year = 1975
end_year = 2025

var_names = ['Wind','SST','Sea_Ice','Velocity','Chlorophyll']
all_year_months = []
all_files_present = []

for var_name in var_names:
    year_months = get_expected_year_months(var_name, start_year, end_year)
    all_year_months.append(year_months)
    files_present = check_if_files_exist(project_folder, var_name, resolution, year_months)
    all_files_present.append(files_present)

year_months_total = []
files_present_total = []
for year in range(start_year, end_year):
    for month in range(1, 13):
        files_present = True
        data_exists = True
        for v, var_name in enumerate(var_names):
            if (year, month) not in all_year_months[v]:
                files_present = False
                data_exists = False
            else:
                idx = all_year_months[v].index((year, month))
                if not all_files_present[v][idx]:
                    files_present = False
        if data_exists:
            files_present_total.append(files_present)
            year_months_total.append((year, month))

var_names.append('All')
all_year_months.append(year_months_total)
all_files_present.append(files_present_total)


plot_file_existence(project_folder, resolution, var_names, all_year_months, all_files_present)



