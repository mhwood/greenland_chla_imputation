
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from pyproj import Transformer
import datetime
from matplotlib.pyplot import GridSpec
from matplotlib.patches import Rectangle

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

def read_bedmachine_to_grid(data_folder):
    bm_file = os.path.join(data_folder,'Greenland','Bathymetry','BedMachine','BedMachineGreenland-v5.nc')
    ds = nc4.Dataset(bm_file)
    bm_x = ds.variables['x'][:]
    bm_y = ds.variables['y'][:]
    bed = ds.variables['bed'][:,:]
    surface = ds.variables['surface'][:,:]
    ds.close()

    bed[surface>0]=0

    # print('    - Interpolating the bathymetry to the grid')
    # bm_X, bm_Y = np.meshgrid(bm_x,bm_y)
    # points = np.column_stack([bm_X.ravel(), bm_Y.ravel()])
    # interp_bed = griddata(points, bed.ravel(), (X,Y), method='nearest')


    bm_X, bm_Y = np.meshgrid(bm_x, bm_y)

    skip = 5
    bm_X = bm_X[::skip,::skip]
    bm_Y = bm_Y[::skip,::skip]
    bed = bed[::skip,::skip]

    bm_Y = np.flipud(bm_Y)
    bed = np.flipud(bed)

    return(bm_X, bm_Y, bed)

def compute_spatial_trends(project_folder, model_version, resolution, start_year, end_year, month):

    for year in range(start_year, end_year+1):
        print(' - Reading in ' + str(year) + '/' + str(month))

        chl_file = os.path.join(project_folder, 'Data', str(resolution)+'km Model',model_version,str(year),
                           'Chl_Modeled_' + model_version +'_' + str(year) + '{:02d}'.format(month) + '.nc')
        ds = nc4.Dataset(chl_file)
        if year==start_year:
            X = ds.variables['X'][:,:]
            Y = ds.variables['Y'][:,:]
        Chl_Model = ds.variables['Chl'][:,:,:]
        ds.close()

        chl_mean = np.nanmean(Chl_Model, axis=0)

        if year == start_year:
            chl_mean_stack = chl_mean[np.newaxis,:,:]
        else:
            chl_mean_stack = np.concatenate((chl_mean_stack, chl_mean[np.newaxis,:,:]), axis=0)

    # Compute the trend at each pixel
    trends = np.full(chl_mean_stack.shape[1:], np.nan)
    for i in range(chl_mean_stack.shape[1]):
        for j in range(chl_mean_stack.shape[2]):
            pixel_values = chl_mean_stack[:, i, j]
            if np.sum(~np.isnan(pixel_values)) > 3:
                x = np.arange(start_year, end_year+1)
                y = pixel_values
                mask = ~np.isnan(y)
                if np.sum(mask) > 3:
                    slope, intercept = np.polyfit(x[mask], y[mask], 1)
                    trends[i, j] = slope

    return(X, Y, trends)




def plot_sample_locations(project_folder, model_version, resolution, month, X, Y, trends, bm_X, bm_Y, bed):
    plt.style.use('dark_background')

    month_to_name = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                     7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    fig = plt.figure(figsize=(6.5, 9))

    gs2 = GridSpec(17, 14, left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05)

    # map
    ax1 = fig.add_subplot(gs2[:, :-1])
    C = plt.pcolormesh(X, Y, trends, vmin=-0.05, vmax=0.05, cmap='BrBG')
    land_mask = (bed == 0.0).astype(int)
    land_mask = np.ma.masked_where(land_mask == 0, land_mask)
    plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
    plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Chlorophyll-a Concentration Trends - Month: ' + month_to_name[month], fontsize=12)
    cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
    cbar.set_label('Trend (mg/m$^3$/yr)', rotation=90)


    output_file = os.path.join(project_folder, 'Figures', 'Maps', 'Trends',
                               'Chl_Trends_' + model_version + '_' + str(resolution) + 'km_' + month_to_name[month] + '.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

home_folder = os.path.expanduser('~')
bedmachine_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

model_version = 'Unet2'
resolution = 15


print(' - Reading in the bedmachine data')
bm_X, bm_Y, bed = read_bedmachine_to_grid(bedmachine_folder)
# bm_X = bm_Y = bed = None

for month in range(1,13):

    print(' Computing Trends')
    X, Y, trends = compute_spatial_trends(project_folder, model_version, resolution, 1982, 2020, month)

    print(' Plotting the trends')
    print(np.nanmin(trends), np.nanmax(trends))
    plot_sample_locations(project_folder, model_version, resolution, month, X, Y, trends, bm_X, bm_Y, bed)





