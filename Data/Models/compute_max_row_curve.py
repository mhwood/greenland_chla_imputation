
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import datetime
import math
from pyproj import Transformer

def reproject_polygon(polygon_array,inputCRS,outputCRS,x_column=0,y_column=1,run_test = True):

    transformer = Transformer.from_crs('EPSG:' + str(inputCRS), 'EPSG:' + str(outputCRS))

    # There seems to be a serious problem with pyproj
    # The x's and y's are mixed up for these transformations
    #       For 4326->3413, you put in (y,x) and get out (x,y)
    #       Foe 3413->4326, you put in (x,y) and get out (y,x)
    # Safest to run check here to ensure things are outputting as expected with future iterations of pyproj

    if inputCRS == 4326 and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,y_column], polygon_array[:,x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    elif inputCRS == 3413 and outputCRS == 4326:
        y2, x2 = transformer.transform(polygon_array[:, x_column], polygon_array[:, y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
    elif str(inputCRS)[:3] == '326' and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,y_column], polygon_array[:,x_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
        run_test = False
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon


def read_domain_from_nc(project_folder, resolution):
    output_file = os.path.join(project_folder, 'Data', '15km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

    ds = nc4.Dataset(output_file)

    X = ds.variables['X'][:,:]
    Y = ds.variables['Y'][:, :]
    Depth = ds.variables['Depth'][:, :]

    ds.close()

    points = np.column_stack([X.ravel(), Y.ravel()])
    points = reproject_polygon(points, 3413, 4326)
    Lon = points[:,0].reshape(X.shape)
    Lat = points[:,1].reshape(Y.shape)

    return(Lon, Lat)

def read_Chl_timeseries(project_folder, resolution, model_version):

    first_file = True

    N = 184
    max_row_grid = np.zeros((N,365))
    max_rows = np.zeros(365)

    for year in range(2010, 2021):
        days_counted = 0
        for month in range(1, 13):
            print(' - Working on ' + str(year) + '/' + str(month))

            chl_file = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                     'Chlorophyll', str(year), 'Chlorophyll_' + str(year) +'{:02d}'.format(month) +'.nc')

            if os.path.exists(chl_file):
                ds = nc4.Dataset(chl_file)
                Chl_Obs = ds.variables['Chl'][:, :, :]
                ds.close()

                if month==2:
                    Chl_Obs = Chl_Obs[:28,:,:]

                for day in range(np.shape(Chl_Obs)[0]):
                    for col in range(np.shape(Chl_Obs)[2]):
                        valid_mask = ~np.isnan(Chl_Obs[day,:,col])
                        max_row = np.argmax(valid_mask)
                        max_row_grid[:max_row,day+days_counted] +=1
                        if max_row > max_rows[day+days_counted]:
                            max_rows[day+days_counted] = max_row
                days_counted += np.shape(Chl_Obs)[0]

    return(max_rows, max_row_grid)

# function from LLM overlords
def daylight_hours(latitude_deg: float, day_of_year: int) -> float:
    """
    Approximate daylight hours for a latitude and day of year.

    latitude_deg: latitude in degrees, positive north
    day_of_year: 1..365
    """
    lat = math.radians(latitude_deg)

    # Solar declination approximation, in radians
    decl = math.radians(23.44) * math.sin(
        math.radians((360 / 365) * (day_of_year - 81))
    )

    # Hour angle at sunrise/sunset
    x = -math.tan(lat) * math.tan(decl)

    # Polar day/night handling
    if x <= -1:
        return 24.0
    if x >= 1:
        return 0.0

    hour_angle = math.acos(x)

    # Convert angle to hours: 2 * hour_angle / (15 deg per hour)
    return 24 * hour_angle / math.pi

def compute_daylight_hours_on_grid(Lat):

    lat = np.arange(np.min(Lat), np.max(Lat), 0.1)
    days = np.arange(1,366)
    daylight_grid = np.zeros((len(lat), len(days)))

    for day in range(1,366):
        for ll in range(len(lat)):
            daylight_grid[ll,day-1] = daylight_hours(lat[ll], day)

    return(daylight_grid)

def compute_theoretical_max_row(Lat,hour_threshold):
    max_rows = np.zeros((365,1))

    for day in range(1,366):
        max_row = 0
        for row in range(np.shape(Lat)[0]):
            for col in range(np.shape(Lat)[1]):
                if daylight_hours(Lat[row,col], day) > hour_threshold:
                    if row > max_row:
                        max_row = row
        max_rows[day-1] = max_row

    return(max_rows)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

Lon, Lat = read_domain_from_nc(project_folder, resolution)

# daylight_grid = compute_daylight_hours_on_grid(Lat)
#
# C = plt.pcolormesh(daylight_grid)
# plt.contour(daylight_grid,levels=[1],colors='w')
# plt.colorbar(C)
# plt.show()

max_rows = compute_theoretical_max_row(Lat, hour_threshold=8)

chl_max_rows, _ = read_Chl_timeseries(project_folder, resolution, model_version)

plt.plot(chl_max_rows,'g-')
plt.plot(max_rows,'k-')
plt.show()



