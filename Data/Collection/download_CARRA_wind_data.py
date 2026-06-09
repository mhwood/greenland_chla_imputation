import cdsapi
import zipfile
import os
import shutil

repo_folder = '/Volumes/CoOL/Data_Repository/Global/Wind/ERA5'
tmp_folder = '/Users/mikewoods/Desktop/Tmp/'
repo_folder = tmp_folder

start_year = 1986
end_year = 1990

# start_year = 1991
# end_year = 1999

start_year = 2002
end_year = 2010

# start_year = 2011
# end_year = 2020

# start_year = 2015
# end_year = 2023

client = cdsapi.Client()

for year in range(start_year, end_year + 1):

    if str(year) not in os.listdir(repo_folder):
        print(f'Creating folder for year {year}...')
        os.mkdir(os.path.join(repo_folder, str(year)))

    year_files = os.listdir(os.path.join(repo_folder, str(year)))

    for month in range(1, 13):

        v_dst_file = 'v_10m_wind_'+str(year)+f"{month:02d}"+'.nc'
        if v_dst_file in year_files:
            print(f'      - File {v_dst_file} is already present. Skipping download.')
            continue
        else:
            print('   - Working on year {} and month {}...'.format(year, month))

            dataset = "reanalysis-pan-carra-means"
            request = {
                "time_aggregation": "daily",
                "level_type": "single_levels",
                "variable": [
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind"
                ],
                "product_type": "analysis_based",
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "data_format": "netcdf"
            }

            client.retrieve(dataset, request, f"{tmp_folder}wind_{year}{month:02d}.nc")#.download()

            # # then, move the files to the repo folder
            # for file_name in os.listdir(f"{tmp_folder}wind_{year}{month:02d}/"):
            #     if file_name.endswith('.nc') and file_name.startswith('10m_u_component'):
            #         dst_file_name = 'u_10m_wind_'+str(year)+f"{month:02d}"+'.nc'
            #         shutil.copyfile(os.path.join(f"{tmp_folder}wind_{year}{month:02d}/", file_name), os.path.join(repo_folder, str(year), dst_file_name))
            #     elif file_name.endswith('.nc') and file_name.startswith('10m_v_component'):
            #         dst_file_name = 'v_10m_wind_'+str(year)+f"{month:02d}"+'.nc'
            #         shutil.copyfile(os.path.join(f"{tmp_folder}wind_{year}{month:02d}/", file_name), os.path.join(repo_folder, str(year), dst_file_name))
            #
            # # finally, deleter the zip file and the unzipped folder
            # os.remove(f"{tmp_folder}wind_{year}{month:02d}.zip")
            # shutil.rmtree(f"{tmp_folder}wind_{year}{month:02d}/")