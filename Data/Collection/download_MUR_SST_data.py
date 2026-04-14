
import requests
import os
from subscriber import podaac_data_downloader as pdd
from subscriber import podaac_access as pa
import argparse
import datetime
import shutil

# define an output folder
data_path = '/Volumes/CoOL/Data_Repository/Global/Sea Surface Temperature/MUR'

local_dir = '/Users/mike/Desktop/Tmp'

# define a collection e.g.:
# MUR SST: MUR25-JPL-L4-GLOB-v04.2
# SMAP SSS: SMAP_JPL_L3_SSS_CAP_MONTHLY_V5
short_name = 'MUR-JPL-L4-GLOB-v4.1'

start_year = 2003
end_year = 2024

for year in range(start_year, end_year + 1):
    if str(year) not in os.listdir(data_path):
        print(f'Creating folder for year {year}...')
        os.mkdir(os.path.join(data_path, str(year)))

for year in range(start_year, end_year + 1):

    for month in range(1, 13):
        print('   - Working on year {} and month {}...'.format(year, month))
        if year % 4 == 0:
            days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        all_files_present = True

        year_files = os.listdir(os.path.join(data_path, str(year)))
        for day in range(1, days_in_month[month-1] + 1):
            date = datetime.datetime(year, month, day)
            file_name = f'{date.strftime("%Y%m%d")}090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc'
            if file_name not in year_files:
                # print('      - File {} is not present. Will be downloaded.'.format(file_name))
                all_files_present = False

        if all_files_present:
            print(f'      - All files are already present. Skipping download.')
        else:
            print(f'      - Downloading files ...')

            # define a start time
            # must be in this format:
            start_date_time = str(year)+'-'+'{:02d}'.format(month)+'-01T00:00:00Z'

            # define a start time
            # must be in this format:
            end_date_time = str(year)+'-'+'{:02d}'.format(month)+'-'+str(days_in_month[month-1])+'T00:00:00Z'

            # create a parser object from the
            parser = pdd.create_parser()
            args = parser.parse_args(['-c',short_name, '-d',local_dir,
                                      '-sd',start_date_time, '-ed',end_date_time])

            # run the downloading script
            pdd.run(args)

            # now move the downloaded files to the correct folder
            for file in sorted(os.listdir(local_dir)):
                if file.endswith('.nc') and file not in os.listdir(os.path.join(data_path, str(year))):
                    print(f'        - Moving file {file} to folder for year {year}...')
                    shutil.copyfile(os.path.join(local_dir, file), os.path.join(data_path, str(year), file))
                    print(f'        - Removing file {file} from local directory...')
                    os.remove(os.path.join(local_dir, file))


