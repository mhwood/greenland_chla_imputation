import cdsapi
import zipfile
import os
import shutil

repo_folder = '/Volumes/CoOL/Data_Repository/Global/Radiation/ERA5'
tmp_folder = '/Users/mikewoods/Desktop/Tmp/'

subset = 1

start_year = 1980
end_year = 1985

start_year = 1985
end_year = 1994

start_year = 1995
end_year = 2004

start_year = 2005
end_year = 2014

# start_year = 2015
# end_year = 2025

client = cdsapi.Client()

for year in range(start_year, end_year + 1):

    if str(year) not in os.listdir(repo_folder):
        print(f'Creating folder for year {year}...')
        os.mkdir(os.path.join(repo_folder, str(year)))

    year_files = os.listdir(os.path.join(repo_folder, str(year)))

    for month in range(1, 13):

        v_dst_file = 'toa_incident_solar_radiation_'+str(year)+f"{month:02d}"+'.nc'
        if v_dst_file in year_files:
            print(f'      - File {v_dst_file} is already present. Skipping download.')
            continue
        else:
            print('   - Working on year {} and month {}...'.format(year, month))

            dataset = "derived-era5-single-levels-daily-statistics"
            request = {
                "product_type": ["reanalysis"],
                "variable": [
                    "surface_net_solar_radiation_clear_sky",
                    "surface_solar_radiation_downwards",
                    "toa_incident_solar_radiation"
                ],
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
                "daily_statistic": "daily_mean"#,
                #"time_zone": "utc+00:00",
                #"frequency": "6_hourly"
            }

            
            client.retrieve(dataset, request, f"{tmp_folder}radiation_{year}{month:02d}.zip")#.download()

            # now, unzip the file
            with zipfile.ZipFile(f"{tmp_folder}radiation_{year}{month:02d}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{tmp_folder}radiation_{year}{month:02d}/")

            # then, move the files to the repo folder
            for file_name in os.listdir(f"{tmp_folder}radiation_{year}{month:02d}/"):
                if file_name.endswith('.nc') and file_name.startswith('surface_net_solar_radiation_clear_sky'):
                    dst_file_name = 'surface_net_solar_radiation_clear_sky_'+str(year)+f"{month:02d}"+'.nc'
                    shutil.copyfile(os.path.join(f"{tmp_folder}radiation_{year}{month:02d}/", file_name), os.path.join(repo_folder, str(year), dst_file_name))
                elif file_name.endswith('.nc') and file_name.startswith('surface_solar_radiation_downwards'):
                    dst_file_name = 'surface_solar_radiation_downwards_'+str(year)+f"{month:02d}"+'.nc'
                    shutil.copyfile(os.path.join(f"{tmp_folder}radiation_{year}{month:02d}/", file_name), os.path.join(repo_folder, str(year), dst_file_name))
                elif file_name.endswith('.nc') and file_name.startswith('toa_incident_solar_radiation'):
                    dst_file_name = 'toa_incident_solar_radiation_'+str(year)+f"{month:02d}"+'.nc'
                    shutil.copyfile(os.path.join(f"{tmp_folder}radiation_{year}{month:02d}/", file_name), os.path.join(repo_folder, str(year), dst_file_name))
            
            # finally, deleter the zip file and the unzipped folder
            os.remove(f"{tmp_folder}radiation_{year}{month:02d}.zip")
            shutil.rmtree(f"{tmp_folder}radiation_{year}{month:02d}/")