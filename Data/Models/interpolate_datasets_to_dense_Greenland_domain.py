
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
    bm_file = os.path.join(data_folder,'Greenland','Bathymetry','BedMachine','BedMachineGreenland-v6.nc')
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
            # print(min_row, max_row, min_col, max_col)

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

def spread_var_horizontally_in_wet_grid_legacy(var_grid, wet_grid, verbose=False):
    """
    Fill zero values in var_grid using the nearest non-zero neighbors within wet areas.
    Legacy implementation using NumPy.

    Parameters:
        var_grid (ndarray): 2D array with variable data, where zeros represent missing values.
        wet_grid (ndarray): 2D mask where 1 indicates wet cells and 0 indicates dry.

    Returns:
        tuple: Updated var_grid and number of remaining unfilled wet cells.
    """
    rows = np.arange(np.shape(var_grid)[0])
    cols = np.arange(np.shape(var_grid)[1])
    Cols, Rows = np.meshgrid(cols, rows)

    is_remaining = np.logical_and(var_grid == 0, wet_grid == 1)
    n_remaining = np.sum(is_remaining)
    continue_iter = True

    for _ in range(n_remaining):
        if verbose:
            if n_remaining>0:
                print(f"         - Remaining cells to fill: {n_remaining}")
        if continue_iter:
            Wet_Rows = Rows[wet_grid == 1]
            Wet_Cols = Cols[wet_grid == 1]
            Wet_Vals = var_grid[wet_grid == 1]
            Wet_Rows = Wet_Rows[Wet_Vals != 0]
            Wet_Cols = Wet_Cols[Wet_Vals != 0]
            Wet_Vals = Wet_Vals[Wet_Vals != 0]

            if len(Wet_Vals) > 0:
                rows_remaining, cols_remaining = np.where(is_remaining)
                for ri in range(n_remaining):
                    row = rows_remaining[ri]
                    col = cols_remaining[ri]
                    row_col_dist = ((Wet_Rows.astype(float) - row) ** 2 + (Wet_Cols.astype(float) - col) ** 2) ** 0.5
                    closest_index = np.argmin(row_col_dist)
                    if row_col_dist[closest_index] < np.sqrt(2):
                        var_grid[row, col] = Wet_Vals[closest_index]

                is_remaining = np.logical_and(var_grid == 0, wet_grid == 1)
                n_remaining_now = np.sum(is_remaining)
                if n_remaining_now < n_remaining:
                    n_remaining = n_remaining_now
                else:
                    continue_iter = False
            else:
                continue_iter = False

    return var_grid, n_remaining

def read_month_SIF_data_to_grid(data_folder, year, month, n_days, X, Y):

    SIF = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    first_file = True

    sat_code = 'n07'
    

    for day in range(1,n_days+1):
        print('    - Reading in day '+str(day))

        if year == 1987:
            if month >= 7:
                if day >= 10:
                    sat_code = 'F08'
        if year > 1987:
            if month >= 8:
                if day >= 10:
                    sat_code = 'F08'
        if year >=1988:
            sat_code = 'F08'
        if year >= 1991:
            if month >= 12:
                if day >= 3:
                    sat_code = 'F11'
        if year > 1991:
            sat_code = 'F11'
        if year == 1995:
            if month >= 10:
                sat_code = 'F13'
        if year > 1995:
            sat_code = 'F13'
        if year > 1995:
            sat_code = 'F13'
        if year >= 2008:
            sat_code = 'F17'

        file_path = os.path.join(data_folder,'Arctic','Sea_Ice','daily','v5',str(year),
                                 'sic_psn25_'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'_'+sat_code+'_v05r00.nc')
        ds = nc4.Dataset(file_path)
        x = ds.variables['x'][:]
        y = ds.variables['y'][:]
        seaice_fraction = ds.variables['cdr_seaice_conc'][:,:,:]
        ds.close()

        # plt.pcolormesh(seaice_fraction[0,:,:])
        # plt.show()

        sif = seaice_fraction[0,:,:]

        if first_file:
            X_3411, Y_3411 = np.meshgrid(x, y)
            # plt.pcolormesh(X_3411, Y_3411, seaice)
            # plt.show()

            points = np.column_stack([X_3411.ravel(), Y_3411.ravel()])
            points = reproject_polygon(points, 3411, 3413)
            X_3413_seaice = np.reshape(points[:, 0], np.shape(X_3411))
            Y_3413_seaice = np.reshape(points[:, 1], np.shape(Y_3411))

            first_file = False

        # fill the inland points to avoid interpolation issues?
        points_nonnan = np.column_stack([X_3411.ravel(), Y_3411.ravel()])
        seaice_nonnan = np.ravel(sif)
        non_zero_locations = seaice_nonnan < 2
        points_nonnan = points_nonnan[non_zero_locations, :]
        seaice_nonnan = seaice_nonnan[non_zero_locations]
        seaice_nearest = griddata(points_nonnan, seaice_nonnan, (X_3411, Y_3411),
                                  method='nearest')

        # C=plt.pcolormesh(X_3413_seaice, Y_3413_seaice, sif)
        # plt.colorbar(C, orientation='horizontal')
        # plt.show()

        seaice_3413 = griddata(np.column_stack([X_3413_seaice.ravel(), Y_3413_seaice.ravel()]),
                               seaice_nearest.ravel(), (X, Y))
        seaice_3413[Depth <= 0] = np.nan

        # plt.subplot(1, 2, 1)
        # C = plt.pcolormesh(sif)
        # plt.colorbar(C, orientation='horizontal')
        # plt.subplot(1, 2, 2)
        # C = plt.pcolormesh(seaice_3413)
        # plt.colorbar(C, orientation='horizontal')
        # plt.show()

        SIF[day-1, :, :] = seaice_3413

    return(SIF)

def read_month_velocity_data_to_grid(project_folder, data_folder, year, month, n_days, X, Y, Depth):

    U = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    V = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    seaice_file = os.path.join(project_folder, 'Data', '1km Interpolated', 'Sea_Ice',str(year),
                                'Sea_Ice_'+str(year)+'{:02d}'.format(month)+'.nc')
    ds = nc4.Dataset(seaice_file)
    seaice = ds.variables['seaice_fraction'][:,:,:]
    ds.close()

                               

    for day in range(1,2):#n_days+1):
        print('      - Reading in day '+str(day))
        file_path = os.path.join(data_folder,'Global','Ocean Currents','Oscar',str(year),
                                 'oscar_currents_final_'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'.nc')
        ds = nc4.Dataset(file_path)
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
        u = ds.variables['u'][:,:,:]
        v = ds.variables['v'][:,:,:]
        ds.close()

        u = u[0,:,:].T
        v = v[0,:,:].T

        seaice_mask = seaice[day-1,:,:] < 0.15

        # move the longitudes from 0-360 to -180-180
        small_lon_indices = lon < 180
        lon = np.concatenate((lon[~small_lon_indices] - 360, lon[small_lon_indices]))
        u = np.concatenate((u[:, ~small_lon_indices], u[:, small_lon_indices]), axis=1)
        v = np.concatenate((v[:, ~small_lon_indices], v[:, small_lon_indices]), axis=1)

        lon_indices = np.logical_and(lon>-100, lon<20)
        lat_indices = np.logical_and(lat>50, lat<85)

        u = u[lat_indices,:]
        u = u[:, lon_indices]
        v = v[lat_indices, :]
        v = v[:, lon_indices]

        lon = lon[lon_indices]
        lat = lat[lat_indices]

        Lon, Lat = np.meshgrid(lon, lat)
        points = np.column_stack([Lon.ravel(), Lat.ravel()])
        points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

        u[u < -100] = np.nan
        v[v < -100] = np.nan
        interp_u = griddata(points, u.ravel(), (X, Y))
        interp_v = griddata(points, v.ravel(), (X, Y))

        interp_u[interp_u<-100] = np.nan
        interp_v[interp_v<-100] = np.nan

        mask = np.logical_or(seaice_mask, Depth > 0)
        interp_u[np.logical_and(np.isnan(interp_u), ~mask)] = 0
        interp_v[np.logical_and(np.isnan(interp_v), ~mask)] = 0

        # make a spread mask which is 1 where the grid cell is wet
        # and the sea ice fraction is less than 0.15, and 0 otherwise
        spread_mask = np.logical_and(seaice_mask, Depth <= 0).astype(int)

        interp_u, _ = spread_var_horizontally_in_wet_grid_legacy(interp_u, mask, verbose=True)

        plt.pcolormesh(X, Y, interp_u)
        plt.show()

        U[day-1, :, :] = interp_u
        V[day-1, :, :]= interp_v

    return(U, V)

def compute_MUR41_SST_triangulation(data_folder, year, month, day, X, Y):

    file_path = os.path.join(data_folder,'Global','Sea Surface Temperature','MUR','v4.1',str(year),
                             str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
    ds = nc4.Dataset(file_path)
    lon = np.array(ds.variables['lon'][:])
    lat = np.array(ds.variables['lat'][:])
    ds.close()

    lon_indices = np.logical_and(lon>-100, lon<20)
    lat_indices = np.logical_and(lat>50, lat<85)

    lon = lon[lon_indices]
    lat = lat[lat_indices]

    print('    - Reprojecting the coordinates to the grid')
    Lon, Lat = np.meshgrid(lon, lat)
    points = np.column_stack([Lon.ravel(), Lat.ravel()])
    reprojected_points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

    tri_dict = {}

    processing_rows = 4
    processing_cols = 4
    for row in range(processing_rows):
        for col in range(processing_cols):
            print('        - Processing chunk in row '+str(row+1)+'/'+str(processing_rows)+' and col '+str(col+1)+'/'+str(processing_cols))
            # subset X and Y to the current processing chunk
            min_row = row * np.shape(X)[0] // processing_rows
            max_row = (row + 1) * np.shape(X)[0] // processing_rows
            min_col = col * np.shape(X)[1] // processing_cols
            max_col = (col + 1) * np.shape(X)[1] // processing_cols
            X_chunk = X[min_row:max_row, min_col:max_col]
            Y_chunk = Y[min_row:max_row, min_col:max_col]

            print('            - Reprojecting the subset of points to the grid coordinates')
            subset = np.column_stack([X_chunk.ravel(), Y_chunk.ravel()])

            buff = 10000
            lon_subset_indices = np.logical_and(reprojected_points[:,0]>=np.min(subset[:,0])-buff, reprojected_points[:,0]<=np.max(subset[:,0])+buff)
            lat_subset_indices = np.logical_and(reprojected_points[:,1]>=np.min(subset[:,1])-buff, reprojected_points[:,1]<=np.max(subset[:,1])+buff)
            subset_indices = np.logical_and(lon_subset_indices, lat_subset_indices)
            points_subset = reprojected_points[subset_indices,:]

            print('            - Computing the Delaunay triangulation of the subset of points')
            tri_subset = Delaunay(np.array(points_subset))

            tri_dict[(row, col)] = {'tri':tri_subset, 'point_subset_indices': subset_indices}

    # print('    - Computing the Delaunay triangulation of the points')
    # tri = Delaunay(points)

    return(tri_dict)

def compute_CCI_SST_triangulation(data_folder, year, month, day, X, Y):

    file_path = os.path.join(data_folder,'Global','Sea Surface Temperature','CCI',str(year),
                             str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc')
    ds = nc4.Dataset(file_path)
    lon = np.array(ds.variables['lon'][:])
    lat = np.array(ds.variables['lat'][:])
    ds.close()

    lon_indices = np.logical_and(lon>-100, lon<20)
    lat_indices = np.logical_and(lat>50, lat<85)

    lon = lon[lon_indices]
    lat = lat[lat_indices]

    print('    - Reprojecting the coordinates to the grid')
    Lon, Lat = np.meshgrid(lon, lat)
    points = np.column_stack([Lon.ravel(), Lat.ravel()])
    reprojected_points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

    print('New')

    tri_dict = {}

    processing_rows = 4
    processing_cols = 4
    for row in range(processing_rows):
        for col in range(processing_cols):
            print('        - Processing chunk in row '+str(row+1)+'/'+str(processing_rows)+' and col '+str(col+1)+'/'+str(processing_cols))
            # subset X and Y to the current processing chunk
            min_row = row * np.shape(X)[0] // processing_rows
            max_row = (row + 1) * np.shape(X)[0] // processing_rows
            min_col = col * np.shape(X)[1] // processing_cols
            max_col = (col + 1) * np.shape(X)[1] // processing_cols
            X_chunk = X[min_row:max_row, min_col:max_col]
            Y_chunk = Y[min_row:max_row, min_col:max_col]

            print('            - Reprojecting the subset of points to the grid coordinates')
            subset = np.column_stack([X_chunk.ravel(), Y_chunk.ravel()])

            buff = 10000
            lon_subset_indices = np.logical_and(reprojected_points[:,0]>=np.min(subset[:,0])-buff, reprojected_points[:,0]<=np.max(subset[:,0])+buff)
            lat_subset_indices = np.logical_and(reprojected_points[:,1]>=np.min(subset[:,1])-buff, reprojected_points[:,1]<=np.max(subset[:,1])+buff)
            subset_indices = np.logical_and(lon_subset_indices, lat_subset_indices)
            points_subset = reprojected_points[subset_indices,:]

            print('            - Computing the Delaunay triangulation of the subset of points')
            tri_subset = Delaunay(np.array(points_subset))

            tri_dict[(row, col)] = {'tri':tri_subset, 'point_subset_indices': subset_indices}

    # print('    - Computing the Delaunay triangulation of the points')
    # tri = Delaunay(points)

    return(tri_dict)

def read_month_MUR41_SST_data_to_grid(data_folder, year, month, n_days, X, Y, tri):

    SST = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    # first_file = True

    for day in range(1,n_days+1):
        process_file = True
        # if year==2013 and month==9 and day in [2,3,4,5,6]: # these files aere corrupted
        #     process_file = False
        
        if process_file:
            print('    - Reading in day '+str(day))
            file_path = os.path.join(data_folder,'Global','Sea Surface Temperature','MUR','v4.1',str(year),
                                    str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'090000-JPL-L4_GHRSST-SSTfnd-MUR-GLOB-v02.0-fv04.1.nc')
            ds = nc4.Dataset(file_path)
            lon = ds.variables['lon'][:]
            lat = ds.variables['lat'][:]
            sst = ds.variables['analysed_sst'][:,:,:]
            ds.close()

            sst = sst[0,:,:]

            lon_indices = np.logical_and(lon>-100, lon<20)
            lat_indices = np.logical_and(lat>50, lat<85)

            sst = sst[lat_indices,:]
            sst = sst[:, lon_indices]

            interp_sst = np.zeros(np.shape(X))

            processing_rows = 4
            processing_cols = 4
            for row in range(processing_rows):
                for col in range(processing_cols):
                    print('        - Processing chunk in row '+str(row+1)+'/'+str(processing_rows)+' and col '+str(col+1)+'/'+str(processing_cols))
                    tri_subset = tri[(row, col)]['tri']
                    subset_indices = tri[(row, col)]['point_subset_indices']
                    sst_subset = sst.ravel()[subset_indices]

                    interp = LinearNDInterpolator(tri_subset, sst_subset)

                    min_row = row * np.shape(X)[0] // processing_rows
                    max_row = (row + 1) * np.shape(X)[0] // processing_rows
                    min_col = col * np.shape(X)[1] // processing_cols
                    max_col = (col + 1) * np.shape(X)[1] // processing_cols
                    X_chunk = X[min_row:max_row, min_col:max_col]
                    Y_chunk = Y[min_row:max_row, min_col:max_col]

                    interp_sst_chunk = interp(X_chunk, Y_chunk)
                    interp_sst[min_row:max_row, min_col:max_col] = interp_sst_chunk

            interp_sst[interp_sst<270] = np.nan

            # plt.pcolormesh(X, Y, interp_sst)
            # plt.colorbar()
            # plt.show()

            SST[day-1, :, :] = interp_sst
        
        else:
            SST[day-1, :, :] = np.nan

    return(SST)

def read_month_MUR42_SST_data_to_grid(data_folder, year, month, n_days, X, Y):

    SST = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    SIF = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    for day in range(1,n_days+1):
        process_file = True
        if year==2013 and month==9 and day==2:
            process_file = False
        
        if process_file:
            file_path = os.path.join(data_folder,'Global','SST',
                                    str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'090000-JPL-L4_GHRSST-SSTfnd-MUR25-GLOB-v02.0-fv04.2.nc')
            ds = nc4.Dataset(file_path)
            lon = ds.variables['lon'][:]
            lat = ds.variables['lat'][:]
            sst = ds.variables['analysed_sst'][:,:,:]
            seaice_fraction = ds.variables['sea_ice_fraction'][:,:,:]
            ds.close()

            sst = sst[0,:,:]
            sif = seaice_fraction[0,:,:]

            lon_indices = np.logical_and(lon>-100, lon<20)
            lat_indices = np.logical_and(lat>50, lat<85)

            sst = sst[lat_indices,:]
            sst = sst[:, lon_indices]
            sif = sif[lat_indices, :]
            sif = sif[:, lon_indices]

            lon = lon[lon_indices]
            lat = lat[lat_indices]

            Lon, Lat = np.meshgrid(lon, lat)
            points = np.column_stack([Lon.ravel(), Lat.ravel()])
            points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

            sst[sst < -100] = np.nan
            sif[sif < -100] = np.nan

            interp_sst = griddata(points, sst.ravel(), (X, Y))
            interp_sif = griddata(points, sif.ravel(), (X, Y))

            interp_sst[interp_sst<-100] = np.nan
            interp_sif[interp_sif < -100] = np.nan

            interp_sif[np.logical_and(~np.isnan(interp_sst), np.isnan(interp_sif))]=0

            SST[day-1, :, :] = interp_sst
            SIF[day - 1, :, :] = interp_sif

    return(SST, SIF)

def read_month_CCI_SST_data_to_grid(data_folder, year, month, n_days, X, Y, tri):

    SST = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    # first_file = True

    for day in range(1,n_days+1):
        process_file = True
        # if year==2013 and month==9 and day in [2,3,4,5,6]: # these files aere corrupted
        #     process_file = False
        
        if process_file:
            print('    - Reading in day '+str(day))
            file_path = os.path.join(data_folder,'Global','Sea Surface Temperature','CCI',str(year),
                                    str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'120000-ESACCI-L4_GHRSST-SSTdepth-OSTIA-GLOB_CDR2.1-v02.0-fv01.0.nc')
            ds = nc4.Dataset(file_path)
            lon = ds.variables['lon'][:]
            lat = ds.variables['lat'][:]
            sst = ds.variables['analysed_sst'][:,:,:]
            ds.close()

            sst = sst[0,:,:]

            lon_indices = np.logical_and(lon>-100, lon<20)
            lat_indices = np.logical_and(lat>50, lat<85)

            sst = sst[lat_indices,:]
            sst = sst[:, lon_indices]

            interp_sst = np.zeros(np.shape(X))

            processing_rows = 4
            processing_cols = 4
            for row in range(processing_rows):
                for col in range(processing_cols):
                    print('        - Processing chunk in row '+str(row+1)+'/'+str(processing_rows)+' and col '+str(col+1)+'/'+str(processing_cols))
                    tri_subset = tri[(row, col)]['tri']
                    subset_indices = tri[(row, col)]['point_subset_indices']
                    sst_subset = sst.ravel()[subset_indices]

                    interp = LinearNDInterpolator(tri_subset, sst_subset)

                    min_row = row * np.shape(X)[0] // processing_rows
                    max_row = (row + 1) * np.shape(X)[0] // processing_rows
                    min_col = col * np.shape(X)[1] // processing_cols
                    max_col = (col + 1) * np.shape(X)[1] // processing_cols
                    X_chunk = X[min_row:max_row, min_col:max_col]
                    Y_chunk = Y[min_row:max_row, min_col:max_col]

                    interp_sst_chunk = interp(X_chunk, Y_chunk)
                    interp_sst[min_row:max_row, min_col:max_col] = interp_sst_chunk

            interp_sst[interp_sst<270] = np.nan

            # plt.pcolormesh(X, Y, interp_sst)
            # plt.colorbar()
            # plt.show()

            SST[day-1, :, :] = interp_sst
        
        else:
            SST[day-1, :, :] = np.nan

    return(SST)

def read_month_SSS_data_to_grid(data_folder, year, month, n_days, X, Y):

    SSS = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    for day in range(1,n_days+1):
        file_path = os.path.join(data_folder,'Global','SSS',
                                 'SMAP_L3_SSS_'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'_8DAYS_V5.0.nc')
        ds = nc4.Dataset(file_path)
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        sss = ds.variables['smap_sss'][:,:]
        ds.close()

        lon_indices = np.logical_and(lon>-100, lon<20)
        lat_indices = np.logical_and(lat>50, lat<85)

        sss = sss[lat_indices,:]
        sss = sss[:, lon_indices]

        lon = lon[lon_indices]
        lat = lat[lat_indices]

        Lon, Lat = np.meshgrid(lon, lat)
        points = np.column_stack([Lon.ravel(), Lat.ravel()])
        points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

        sss[sss < -100] = np.nan

        interp_sss = griddata(points, sss.ravel(), (X, Y), method='nearest')

        interp_sss[interp_sss<-100] = np.nan

        SSS[day-1, :, :] = interp_sss

    return(SSS)

def read_month_ERA5_wind_velocity_data_to_grid(data_folder, year, month, n_days, X, Y):

    U = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    V = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    file_path = os.path.join(data_folder,'Global','Wind','ERA5',str(year),
                                'u_10m_wind_'+str(year)+'{:02d}'.format(month)+'.nc')
    ds = nc4.Dataset(file_path)
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    u = ds.variables['u10'][:,:,:]
    ds.close()

    file_path = os.path.join(data_folder,'Global','Wind','ERA5',str(year),
                                'v_10m_wind_'+str(year)+'{:02d}'.format(month)+'.nc')
    ds = nc4.Dataset(file_path)
    v = ds.variables['v10'][:,:,:]
    ds.close()

    lon_indices_r = np.logical_and(lon>260, lon<360)
    lon_indics_l = np.logical_and(lon>=0, lon<20)
    lon_indices = np.logical_or(lon_indices_r, lon_indics_l)
    lat_indices = np.logical_and(lat>50, lat<85)

    u = u[:,lat_indices,:]
    u = u[:,:, lon_indices]
    v = v[:,lat_indices, :]
    v = v[:,:, lon_indices]

    lon = lon[lon_indices]
    lat = lat[lat_indices]

    Lon, Lat = np.meshgrid(lon, lat)
    points = np.column_stack([Lon.ravel(), Lat.ravel()])
    points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

    u[u < -100] = np.nan
    v[v < -100] = np.nan

    for day in range(1,n_days+1):
        print('        - Interpolating day '+str(day))

        interp_u = griddata(points, u[day-1,:,:].ravel(), (X, Y))
        interp_v = griddata(points, v[day-1,:,:].ravel(), (X, Y))

        interp_u[interp_u<-100] = np.nan
        interp_v[interp_v<-100] = np.nan

        interp_u[interp_u==0] = np.nan
        interp_v[interp_v == 0] = np.nan

        U[day-1, :, :] = interp_u
        V[day-1, :, :]= interp_v

    return(U, V)

def read_month_AQUA_MODIS_Chl_data_to_grid(data_folder, year, month, n_days, X, Y):

    Chl = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    for day in range(1,n_days+1):
        file_path = os.path.join(data_folder,'Global','Chlorophyll',
                                 'AQUA_MODIS.'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'.L3m.DAY.CHL.chlor_a.4km.nc')
        ds = nc4.Dataset(file_path)
        lon = ds.variables['lon'][:]
        lat = ds.variables['lat'][:]
        chl = ds.variables['chlor_a'][:,:]
        ds.close()

        lon_indices = np.logical_and(lon>-100, lon<20)
        lat_indices = np.logical_and(lat>50, lat<85)

        chl = chl[lat_indices,:]
        chl = chl[:, lon_indices]

        lon = lon[lon_indices]
        lat = lat[lat_indices]

        Lon, Lat = np.meshgrid(lon, lat)
        points = np.column_stack([Lon.ravel(), Lat.ravel()])
        points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

        chl[chl < -100] = np.nan

        interp_chl = griddata(points, chl.ravel(), (X, Y), method='nearest')

        interp_chl[interp_chl<-100] = np.nan

        Chl[day-1, :, :] = interp_chl

    return(Chl)

def read_month_OCCCI_Chl_data_to_grid(data_folder, year, month, n_days, X, Y):

    Chl = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    problem_files = ((1997,10,13),(1997,10,14),(1997,10,15),(1997,10,16),(1997,10,17),(1997,10,18),(1997,10,19),(1997,10,20),
                     (1997,12,15),(1998,7,10),(1998,11,17),(1998,11,18),(1998,11,19),(1998,11,20),
                     (1998,12,17),(1999,1,1),(1999,1,25),
                     (1999,11,17),(1999,11,18),(2000,1,1),(2000,11,17),(2001,1,1),(2001,3,9),(2001,11,18),
                     (2002,4,25),(2002,1,1),(2003,1,1),(2003,4,1),(2004,1,1),
                     (2005,1,1),(2006,1,1), (2006,9,7), (2006,7,27),(2007,7,14),(2007,1,1),(2008,8,22),(2008,1,1),(2009,1,1),
                     (2010,1,2),(2010,1,6),(2010,1,13),(2010,1,18),(2010,1,20),(2010,1,21),(2010,1,22),(2010,1,23),(2010,1,24),(2010,1,25),
                     (2010,1,26),(2010,1,27),(2010,1,28),(2010,1,29),(2010,1,30),(2010,1,31),
                     (2010,2,8),(2012,10,1),(2012,1,1),(2011,1,1),(2010,1,9),(2014,1,30),
                     (2010,1,8),(2010,1,15),(2013,9,2),(2013,1,1),(2014,10,22),(2014,1,1),
                     (2015,1,1),(2016,1,1),(2017,1,1),(2017,12,12),(2018,1,1),(2019,1,1),(2020,1,1),(2020,1,6),(2020,1,7),
                     (2020,8,23),(2020,6,19),(2021,1,1))

    for day in range(1,n_days+1):
        print('    - Reading in day '+str(day))
        if (year, month, day) in problem_files:
            print('        - This file is known to be problematic, filling with NaNs')
            Chl[day-1, :, :] = np.nan
        else:
            file_path = os.path.join(data_folder,'Global','Ocean Color',
                                    'OC-CCI','dap.ceda.ac.uk','neodc','esacci','ocean_colour','data','v6.0-release',
                                    'geographic','netcdf','chlor_a','daily','v6.0',str(year),
                                    'ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'-fv6.0.nc')
            ds = nc4.Dataset(file_path)
            lon = ds.variables['lon'][:]
            lat = ds.variables['lat'][:]
            chl = ds.variables['chlor_a'][:,:, :]
            ds.close()

            chl = chl[0,:,:]

            lon_indices = np.logical_and(lon>-100, lon<20)
            lat_indices = np.logical_and(lat>50, lat<85)

            chl = chl[lat_indices,:]
            chl = chl[:, lon_indices]

            lon = lon[lon_indices]
            lat = lat[lat_indices]

            Lon, Lat = np.meshgrid(lon, lat)
            points = np.column_stack([Lon.ravel(), Lat.ravel()])
            points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

            chl[chl < -100] = np.nan
            chl[chl > 100] = np.nan

            if np.any(~np.isnan(chl)):
                print('        - Interpolating the OC-CCI Chl data')

                interp_chl_nearest = griddata(points, chl.ravel(), (X, Y), method='nearest')
                interp_chl = griddata(points, chl.ravel(), (X, Y), method='linear')
                interp_chl[np.isnan(interp_chl)] = interp_chl_nearest[np.isnan(interp_chl)]

                interp_chl[interp_chl<-100] = np.nan

                Chl[day-1, :, :] = interp_chl
            
            else:
                Chl[day-1, :, :] = np.nan
                print('        - No valid data found for this day, filling with NaNs')

        # plt.subplot(1,2,1)
        # plt.pcolormesh(chl, cmap='turbo')
        # plt.colorbar()
        # plt.subplot(1,2,2)
        # plt.pcolormesh(interp_chl, cmap='turbo')
        # plt.colorbar()
        # plt.show()

    return(Chl)

def read_month_CZCS_Chl_data_to_grid(data_folder, year, month, n_days, X, Y):

    Chl = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    data_found = False

    for day in range(1,n_days+1):
        file_path = os.path.join(data_folder,'CZCS',str(year), #'Global','Chlorophyll',
                                 'NIMBUS-7_CZCS.'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'.L3m.DAY.CHL.chlor_a.4km.nc')
        if os.path.exists(file_path):
            if not data_found:
                print('    - Interpolating the CZCS Chl data')
                data_found = True
            ds = nc4.Dataset(file_path)
            lon = ds.variables['lon'][:]
            lat = ds.variables['lat'][:]
            chl = ds.variables['chlor_a'][:,:]
            ds.close()

            lon_indices = np.logical_and(lon>-100, lon<20)
            lat_indices = np.logical_and(lat>50, lat<85)

            chl = chl[lat_indices,:]
            chl = chl[:, lon_indices]

            lon = lon[lon_indices]
            lat = lat[lat_indices]

            Lon, Lat = np.meshgrid(lon, lat)
            points = np.column_stack([Lon.ravel(), Lat.ravel()])
            points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)

            chl[chl < -100] = np.nan

            interp_chl = griddata(points, chl.ravel(), (X, Y), method='nearest')

            interp_chl[interp_chl<-100] = np.nan

            Chl[day-1, :, :] = interp_chl

    return(Chl, data_found)

def check_file_existence(project_folder, year, month, subset):

    output_file = os.path.join(project_folder, 'Data', '1km Interpolated', subset, str(year),
                               subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')

    return os.path.isfile(output_file)

def write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, subset, data_grids, data_grid_names):

    days = np.arange(1,n_days+1)

    if str(year) not in os.listdir(os.path.join(project_folder, 'Data', '1km Interpolated', subset)):
        os.mkdir(os.path.join(project_folder, 'Data', '1km Interpolated', subset, str(year)))

    output_file = os.path.join(project_folder, 'Data', '1km Interpolated', subset, str(year),
                               subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')

    ds = nc4.Dataset(output_file,'w')

    ds.createDimension('x',np.shape(X)[1])
    ds.createDimension('y', np.shape(X)[0])
    ds.createDimension('days', len(days))

    dvar = ds.createVariable('days', 'f4', ('days',))
    dvar[:] = days
    xvar = ds.createVariable('x', 'f4', ('x',))
    xvar[:] = X[0,:]
    xvar = ds.createVariable('y', 'f4', ('y',))
    xvar[:] = Y[:,0]
    xvar = ds.createVariable('X','f4',('y','x'))
    xvar[:,:] = X
    xvar = ds.createVariable('Y', 'f4', ('y', 'x'))
    xvar[:, :] = Y
    xvar = ds.createVariable('Depth', 'f4', ('y', 'x'))
    xvar[:, :] = Depth

    for v, var_name in enumerate(data_grid_names):
        xvar = ds.createVariable(var_name, 'f4', ('days','y', 'x'))
        xvar[:, :, :] = data_grids[v]

    ds.close()


home_folder = os.path.expanduser('~')

bedmachine_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

data_folder = '/Volumes/CoOL/Data_Repository'
#data_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

project_folder = os.path.join(home_folder, 'Documents', 'Research', 'Projects', 'Greenland Chl Imputation')


resolution = 1

X, Y = create_domain(resolution)

if 'Greenland_domain_'+str(resolution)+'km.nc' not in os.listdir(os.path.join(project_folder,'Data',str(resolution)+'km Interpolated')):
    X, Y = create_domain(resolution)
    print(' - Reading in the Bedmachine Grid to Domain')
    X, Y, Lon, Lat, Bed, mask = read_bedmachine_to_grid(bedmachine_folder, X, Y)
    Depth = Bed*-1
    write_domain_to_nc(project_folder, resolution, X, Y, Lon, Lat, Depth, mask)
else:
    X, Y, Depth = read_domain_from_nc(project_folder, resolution)


make_SST = True
make_sea_ice = True
make_velocity = False
make_wind = True
make_Chl = True

# C = plt.pcolormesh(X,Y,Bed)
# plt.colorbar(C)
# plt.show()

project_folder = os.path.join('/Volumes', 'CoOL', 'Projects', 'Greenland Chl Imputation')

# start_year = 1986
# end_year = 1990

# start_year = 1990
# end_year = 1999

# start_year = 2000
# end_year = 2010

start_year = 2020
end_year = 2025

for year in range(start_year, end_year+1):

    if make_SST and year>=2003:
        mur_tri_file = os.path.join(project_folder, 'Data', '1km Interpolated', 'MUR41_SST_triangulation.pkl')
        if os.path.isfile(mur_tri_file):
            print(' - Loading the MUR41 SST triangulation from file')
            with open(mur_tri_file, 'rb') as f:
                mur_41_tri = pickle.load(f)
        else:
            mur_41_tri = compute_MUR41_SST_triangulation(data_folder, 2010, 1, 1, X, Y)
            # now pickle the tri object so we don't have to compute it every time
            with open(mur_tri_file, 'wb') as f:
                pickle.dump(mur_41_tri, f)
    
    if make_SST and year<2003:
        cci_tri_file = os.path.join(project_folder, 'Data', '1km Interpolated', 'CCI_SST_triangulation.pkl')
        if os.path.isfile(cci_tri_file):
            print(' - Loading the CCI SST triangulation from file')
            with open(cci_tri_file, 'rb') as f:
                cci_tri = pickle.load(f)
        else:
            cci_tri = compute_CCI_SST_triangulation(data_folder, 1992, 1, 1, X, Y)
            # now pickle the tri object so we don't have to compute it every time
            with open(cci_tri_file, 'wb') as f:
                pickle.dump(cci_tri, f)

    for month in range(1,13):
        print(' - Working on '+str(year)+'/'+str(month))

        if month in [1, 3, 5, 7, 8, 10, 12]:
            n_days = 31
        elif month in [4, 6, 9, 11]:
            n_days = 30
        else:
            if year % 4 == 0:
                n_days = 29
            else:
                n_days = 28

        #############################################################################################
        # Sea Ice Data
        file_exists = check_file_existence(project_folder, year, month, 'Sea_Ice')
        if make_sea_ice and not file_exists:
            print('    - Interpolating the sea ice data')
            try:
                SIF = read_month_SIF_data_to_grid(data_folder, year, month, n_days, X, Y)
                write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth,
                                     'Sea_Ice', [SIF], ['seaice_fraction'])
            except KeyboardInterrupt:
                print(' - Keyboard Interrupt detected, stopping code execution')
                break
            except:
                print('        - Error in sea ice data for this month')

        #############################################################################################
        # Velocity
        file_exists = check_file_existence(project_folder, year, month, 'Velocity')
        if make_velocity and True:#not file_exists:
            print('    - Interpolating the velocity data')
            # try:
            U, V = read_month_velocity_data_to_grid(project_folder, data_folder, year, month, n_days, X, Y, Depth)
            write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Velocity', [U,V], ['U','V'])
                # plt.subplot(1,2,1)
                # C = plt.pcolormesh(X, Y, U[0,:,:])
                # plt.colorbar(C)
                # plt.subplot(1, 2, 2)
                # C = plt.pcolormesh(X, Y, V[0,:,:])
                # plt.colorbar(C)
                # plt.show()
            # except KeyboardInterrupt:
            #     print(' - Keyboard Interrupt detected, stopping code execution')
            #     break
            # except:
            #     print('        - Error in velocity data for this month')

        #############################################################################################
        # Temperature

        file_exists = check_file_existence(project_folder, year, month, 'SST')
        if make_SST and not file_exists:
            print('    - Interpolating the sea surface temperature data')
            try:
                if year<2003:
                    SST = read_month_CCI_SST_data_to_grid(data_folder, year, month, n_days, X, Y, cci_tri)
                    write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'SST', [SST], ['SST'])
    
                    # C = plt.pcolormesh(X, Y, SST[0,:,:], cmap='turbo')
                    # plt.colorbar(C)
                    # plt.show()
                else:
                    SST = read_month_MUR41_SST_data_to_grid(data_folder, year, month, n_days, X, Y, mur_41_tri)
                    write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'SST', [SST], ['SST'])
            except KeyboardInterrupt:
                print(' - Keyboard Interrupt detected, stopping code execution')
                break
            except:
                print('        - Error in temperature data for this month')

        #############################################################################################
        # Salinity

        # print('    - Interpolating the SSS data')
        # SSS = read_month_SSS_data_to_grid(data_folder, year, month, n_days, X, Y)
        # write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'SSS', [SSS], ['SSS'])

        #############################################################################################
        # Wind Velocity

        # print('    - Interpolating the wind velocity data')
        # U_wind, V_wind = read_month_wind_velocity_data_to_grid(data_folder, year, month, n_days, X, Y)
        # write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Surface_Wind', [U_wind,V_wind], ['U_wind','V_wind'])

        file_exists = check_file_existence(project_folder, year, month, 'Wind')
        if make_wind and not file_exists:
            print('    - Interpolating the wind velocity data')
            try:
                U_wind, V_wind = read_month_ERA5_wind_velocity_data_to_grid(data_folder, year, month, n_days, X, Y)
                write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Wind', [U_wind,V_wind], ['U_wind','V_wind'])

                # C = plt.pcolormesh(X, Y, U_wind[0,:,:], cmap='turbo')
                # plt.colorbar(C)
                # plt.show()
            except KeyboardInterrupt:
                print(' - Keyboard Interrupt detected, stopping code execution')
                break
            except:
                print('        - Error in wind data for this month')

        #############################################################################################
        # Chl Data

        if make_Chl:

            # Chl = read_month_AQUA_MODIS_Chl_data_to_grid(data_folder, year, month, n_days, X, Y)
            # write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Chlorophyll', [Chl], ['Chl'])

            file_exists = check_file_existence(project_folder, year, month, 'Chlorophyll')
            try:
                if year< 1987:
                    if not file_exists:
                        Chl, data_found = read_month_CZCS_Chl_data_to_grid(data_folder, year, month, n_days, X, Y)
                        if data_found:
                            write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Chlorophyll', [Chl], ['Chl'])
                            # C = plt.pcolormesh(X, Y, np.nanmean(Chl, axis=0), cmap='turbo')
                            # plt.colorbar(C)
                            # plt.show()
                if year>=1997:
                    if not file_exists:
                        Chl = read_month_OCCCI_Chl_data_to_grid(data_folder, year, month, n_days, X, Y)
                        write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Chlorophyll', [Chl], ['Chl'])
                        # C = plt.pcolormesh(X, Y, np.nanmean(Chl, axis=0), cmap='turbo')
                        # plt.colorbar(C)
                        # plt.show()
            except KeyboardInterrupt:
                print(' - Keyboard Interrupt detected, stopping code execution')
                break
            except:
                print('        - Error in chlorophyll data for this month')

        # plt.subplot(1,2,1)
        # C = plt.pcolormesh(X, Y, U_wind[0,:,:])
        # plt.colorbar(C)
        # plt.subplot(1, 2, 2)
        # C = plt.pcolormesh(X, Y, V_wind[0,:,:])
        # plt.colorbar(C)
        # plt.show()


