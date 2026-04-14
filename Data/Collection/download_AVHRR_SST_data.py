
import os
import requests
import datetime
import shutil


subset = 'day'

# define an output folder
data_path = '/Volumes/CoOL/Data_Repository/Global/Sea Surface Temperature/AVHRR'
local_dir = '/Users/mike/Desktop/Tmp'

start_year = 1982
end_year = 1982

for year in range(start_year, end_year + 1):
    # url = 'https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/'+str(year)+'/data/'
    if str(year) not in os.listdir(data_path):
        print(f'Creating folder for year {year}...')
        os.mkdir(os.path.join(data_path, str(year)))

for year in range(start_year, end_year + 1):

    for month in range(1, 2):
        print('   - Working on year {} and month {}...'.format(year, month))
        if year % 4 == 0:
            days_in_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        all_files_present = True

        year_files = os.listdir(os.path.join(data_path, str(year)))
        for day in range(1, days_in_month[month-1] + 1):
            date = datetime.datetime(year, month, day)
            date_string = date.strftime("%Y%m%d")
            file_found = False
            for file_name in year_files:
                if file_name.startswith(date_string):
                    file_found = True
                    break
            if not file_found:
                all_files_present = False

        if all_files_present:
            print(f'      - All files are already present. Skipping download.')
        else:
            print(f'      - Downloading files ...')

            # get the list of files in the directory
            url_head = 'https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/'+str(year)+'/data/'
            download_urls = []
            response = requests.get(url_head)
            if response.status_code == 200:
                lines = response.text.splitlines()
                for line in lines:
                    # print(line)
                    if 'href="' in line and '">' in line and subset in line:
                        download_url = line.split('href="')[1].split('">')[0]
                        print(download_url)
                        if download_url.startswith(str(year)+'{:02d}'.format(month)) and download_url.endswith('.nc'):
                            download_urls.append(url_head + download_url)

            print(download_urls)
            #https://www.ncei.noaa.gov/data/oceans/pathfinder/Version5.3/L3C/1981/data/19810825023019-NCEI-L3C_GHRSST-SSTskin-AVHRR_Pathfinder-PFV5.3_NOAA07_G_1981237_night-v02.0-fv01.0.nc
            for download_url in download_urls:
                file_name = download_url.split('/')[-1]
                print(f'        - Downloading file {file_name}...')
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    with open(os.path.join(local_dir, file_name), 'wb') as f:
                        shutil.copyfileobj(response.raw, f)
                    print(f'        - File {file_name} downloaded successfully.')
                else:
                    print(f'        - Failed to download file {file_name}. Status code: {response.status_code}')

            # # now move the downloaded files to the correct folder
            # for file in sorted(os.listdir(local_dir)):
            #     if file.endswith('.nc') and file not in os.listdir(os.path.join(data_path, str(year))):
            #         print(f'        - Moving file {file} to folder for year {year}...')
            #         shutil.copyfile(os.path.join(local_dir, file), os.path.join(data_path, str(year), file))
            #         print(f'        - Removing file {file} from local directory...')
            #         os.remove(os.path.join(local_dir, file))













