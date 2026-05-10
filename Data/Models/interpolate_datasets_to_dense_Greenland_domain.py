
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from pyproj import Transformer
import pickle

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
    elif inputCRS == 3411 and outputCRS == 3413:
        x2, y2 = transformer.transform(polygon_array[:,x_column], polygon_array[:,y_column])
        x2 = np.array(x2)
        y2 = np.array(y2)
        run_test = False
    else:
        raise ValueError('Reprojection with this epsg is not safe - no test for validity has been implemented')

    output_polygon=np.copy(polygon_array)
    output_polygon[:,x_column] = x2
    output_polygon[:,y_column] = y2
    return output_polygon

def create_domain(resolution):
    min_x = -652925.0
    max_x = 879625.0
    min_y = -3384425.0
    max_y = -632675.0

    x = np.arange(min_x, max_x, resolution*1000)
    y = np.arange(min_y, max_y, resolution*1000)

    X, Y = np.meshgrid(x,y)
    return(X,Y)

def read_bedmachine_to_grid(data_folder, X, Y):
    bm_file = os.path.join(data_folder,'Greenland','Bathymetry','BedMachine','BedMachineGreenland-v5.nc')
    ds = nc4.Dataset(bm_file)
    bm_x = ds.variables['x'][:]
    bm_y = ds.variables['y'][:]
    bed = ds.variables['bed'][:,:]
    surface = ds.variables['surface'][:,:]
    ds.close()
    bed[surface>0]=0

    buffer = 2000
    min_x = np.min(bm_x)+buffer
    max_x = np.max(bm_x)-buffer+1000
    min_y = np.min(bm_y)+buffer
    max_y = np.max(bm_y)-buffer+1000

    # round to the nearest 1000
    min_x = np.floor(min_x / 1000) * 1000
    max_x = np.ceil(max_x / 1000) * 1000
    min_y = np.floor(min_y / 1000) * 1000
    max_y = np.ceil(max_y / 1000) * 1000
    x = np.arange(min_x, max_x, 1000)
    y = np.arange(min_y, max_y, 1000)
    X, Y = np.meshgrid(x,y)

    Bed = np.zeros(np.shape(X))

    processing_rows = 16
    processing_cols = 16
    for row in range(processing_rows):
        for col in range(processing_cols):
            print('        - Processing chunk in row ' + str(row + 1) + '/' + str(processing_rows) + ' and col ' + str(
                col + 1) + '/' + str(processing_cols))

            # subset X and Y to the current processing chunk
            min_row = row * np.shape(X)[0] // processing_rows
            max_row = (row + 1) * np.shape(X)[0] // processing_rows
            min_col = col * np.shape(X)[1] // processing_cols
            max_col = (col + 1) * np.shape(X)[1] // processing_cols
            X_chunk = X[min_row:max_row, min_col:max_col]
            Y_chunk = Y[min_row:max_row, min_col:max_col]
            print(min_row, max_row, min_col, max_col)

            # get the subset of bedmachine around the current processing chunk
            bmx_indices = np.logical_and(bm_x >= np.min(X_chunk) - buffer, bm_x <= np.max(X_chunk) + buffer)
            bmy_indices = np.logical_and(bm_y >= np.min(Y_chunk) - buffer, bm_y <= np.max(Y_chunk) + buffer)
            bm_x_subset = bm_x[bmx_indices]
            bm_y_subset = bm_y[bmy_indices]
            bed_subset = bed[bmy_indices, :][:, bmx_indices]

            # interpolate the bed to the current processing chunk
            bm_X, bm_Y = np.meshgrid(bm_x_subset, bm_y_subset)
            points = np.column_stack([bm_X.ravel(), bm_Y.ravel()])
            interp_bed_chunk = griddata(points, bed_subset.ravel(), (X_chunk,Y_chunk), method='nearest')

            # insert the interpolated bed chunk into the full bed array
            Bed[min_row:max_row, min_col:max_col] = interp_bed_chunk


    mask = Bed == 0
    mask = 1 - mask.astype(int)

    points = np.column_stack([X.ravel(), Y.ravel()]).astype(float)
    points = reproject_polygon(points, inputCRS=3413, outputCRS=4326)
    Lon = np.reshape(points[:, 0], np.shape(X))
    Lat = np.reshape(points[:, 1], np.shape(Y))

    plt.pcolormesh(X, Y, Bed)
    plt.colorbar()
    plt.show()

    return(X, Y, Lon, Lat, Bed, mask)

def write_domain_to_nc(project_folder, resolution, X, Y, Lon, Lat, Depth, mask):
    output_file = os.path.join(project_folder, 'Data', '1km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

    ds = nc4.Dataset(output_file, 'w')

    ds.createDimension('x', np.shape(X)[1])
    ds.createDimension('y', np.shape(X)[0])

    xvar = ds.createVariable('x', 'f4', ('x',))
    xvar[:] = X[0, :]
    xvar = ds.createVariable('y', 'f4', ('y',))
    xvar[:] = Y[:, 0]
    xvar = ds.createVariable('X', 'f4', ('y', 'x'))
    xvar[:, :] = X
    xvar = ds.createVariable('Y', 'f4', ('y', 'x'))
    xvar[:, :] = Y
    xvar = ds.createVariable('Depth', 'f4', ('y', 'x'))
    xvar[:, :] = Depth
    xvar = ds.createVariable('mask', 'f4', ('y', 'x'))
    xvar[:, :] = mask
    lvar = ds.createVariable('Lon', 'f4', ('y', 'x'))
    lvar[:, :] = Lon
    lvar = ds.createVariable('Lat', 'f4', ('y', 'x'))
    lvar[:, :] = Lat

    ds.close()

def read_domain_from_nc(project_folder, resolution):
    output_file = os.path.join(project_folder, 'Data', '1km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

    ds = nc4.Dataset(output_file)

    X = ds.variables['X'][:,:]
    Y = ds.variables['Y'][:, :]
    Depth = ds.variables['Depth'][:, :]

    ds.close()

    return(X,Y, Depth)


home_folder = os.path.expanduser('~')

bedmachine_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

data_folder = '/Volumes/CoOL/Data_Repository'
#data_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

project_folder = os.path.join(home_folder, 'Documents', 'Research', 'Projects', 'Greenland Chl Imputation')

resolution = 1

X, Y = create_domain(resolution)

if True:#'Greenland_domain_'+str(resolution)+'km.nc' not in os.listdir(os.path.join(project_folder,'Data',str(resolution)+'km Interpolated')):
    X, Y = create_domain(resolution)
    print(' - Reading in the Bedmachine Grid to Domain')
    X, Y, Lon, Lat, Bed, mask = read_bedmachine_to_grid(bedmachine_folder, X, Y)
    Depth = Bed*-1
    write_domain_to_nc(project_folder, resolution, X, Y, Lon, Lat, Depth, mask)
else:
    X, Y, Depth = read_domain_from_nc(project_folder, resolution)
