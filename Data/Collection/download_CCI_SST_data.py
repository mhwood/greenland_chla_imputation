
import requests
import os

import argparse
import datetime
import shutil

# define an output folder
data_path = '/Volumes/CoOL/Data_Repository/Global/Sea Surface Temperature/CCI'

start_year = 1982
end_year = 2016

for year in range(start_year, end_year + 1):
    if str(year) not in os.listdir(data_path):
        print(f'Creating folder for year {year}...')
        os.mkdir(os.path.join(data_path, str(year)))

for year in range(start_year, end_year + 1):

    SST_dir = data_path+'/'+str(year)

    for month in range(1, 13):
        if month in [1 ,3 ,5 ,7 ,8 ,10 ,12]:
            days_in_month = 31
        elif month in [4 ,6 ,9 ,11]:
            days_in_month = 30
        elif month==2:
            days_in_month = 29 if year % 4==0 else 28
        for day in range(1, days_in_month +1):
            try:
                print('Downloading file in  ' +str(year) +'/' +str(month) +'/' +str(day))
                url = 'https://data.cci.ceda.ac.uk/thredds/fileServer/esacci/sst/data/CDR_v2/Analysis/L4/' \
                        'v2.1/' +str(year) +'/' +'{:02d}'.format(month) +'/' +'{:02d}'.format(day)+ \
                        '/' +str(year) +'{:02d}'.format(month) +'{:02d}'.format(day) +'120000-' \
                                                                                'ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc'

                file_name = url.split('/')[-1]
                if file_name not in os.listdir(SST_dir):
                    output_file = os.path.join(SST_dir,file_name)

                    with requests.get(url, stream=True, allow_redirects=True) as r:
                        r.raise_for_status()
                        with open(output_file, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
            except KeyboardInterrupt:
                print(' - Keyboard Interrupt detected, stopping code execution')
                break
            except:
                print('        - Error in data for this day')

