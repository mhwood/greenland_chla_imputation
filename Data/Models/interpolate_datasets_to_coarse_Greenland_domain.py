
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
    bm_file = '/Users/mike/Documents/Research/Data Repository/Greenland/Bathymetry/BedMachine/BedMachineGreenland-v5.nc'
    # bm_file = os.path.join(data_folder,'Greenland','Bathymetry','BedMachine','BedMachineGreenland-v5.nc')
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

    skip = 100
    bm_X, bm_Y = np.meshgrid(bm_x, bm_y)
    bm_X = bm_X[::skip,::skip]
    bm_Y = bm_Y[::skip,::skip]
    bed = bed[::skip,::skip]

    bm_Y = np.flipud(bm_Y)
    bed = np.flipud(bed)

    return(bm_X, bm_Y, bed)

def write_domain_to_nc(project_folder, resolution, X, Y, Depth):
    output_file = os.path.join(project_folder, 'Data', '15km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

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

    ds.close()

def read_domain_from_nc(project_folder, resolution):
    output_file = os.path.join(project_folder, 'Data', '15km Interpolated', 'Greenland_domain_'+str(resolution)+'km.nc')

    ds = nc4.Dataset(output_file)

    X = ds.variables['X'][:,:]
    Y = ds.variables['Y'][:, :]
    Depth = ds.variables['Depth'][:, :]

    ds.close()

    return(X,Y, Depth)

def read_month_SIF_data_to_grid(data_folder, year, month, n_days, X, Y):

    SIF = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    first_file = True

    sat_code = 'N07'
    if year == 1987:
        if month >= 8:
            sat_code = 'F08'
    if year > 1987:
        sat_code = 'F08'
    if year >= 1992:
        sat_code = 'F11'
    if year == 1995:
        if month >= 10:
            sat_code = 'F13'
    if year > 1995:
        sat_code = 'F13'
    if year >= 2008:
        sat_code = 'F17'

    for day in range(1,n_days+1):
        print('    - Reading in day '+str(day))
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

def read_month_velocity_data_to_grid(data_folder, year, month, n_days, X, Y):

    U = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    V = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    for day in range(1,n_days+1):
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

        U[day-1, :, :] = interp_u
        V[day-1, :, :]= interp_v

    return(U, V)

def compute_MUR41_SST_triangulation(data_folder, year, month, day, X, Y):

    file_path = os.path.join(data_folder,'Global','SST','MUR',str(year),
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

    print('    - Computing the Delaunay triangulation of the points')
    tri = Delaunay(points)

    return(tri)

def read_month_MUR41_SST_data_to_grid(data_folder, year, month, n_days, X, Y, tri):

    SST = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    # first_file = True

    for day in range(1,n_days+1):
        print('    - Reading in day '+str(day))
        file_path = os.path.join(data_folder,'Global','SST','MUR',str(year),
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

        # lon = lon[lon_indices]
        # lat = lat[lat_indices]

        # if first_file:
        #     print('    - Reprojecting the coordinates to the grid')
        #     Lon, Lat = np.meshgrid(lon, lat)
        #     points = np.column_stack([Lon.ravel(), Lat.ravel()])
        #     points = reproject_polygon(points, inputCRS=4326, outputCRS=3413)
        #     sst[sst < -100] = np.nan
        #
        #     print('    - Computing the Delaunay triangulation of the points')
        #     tri = Delaunay(points)
        #     first_file = False

        print('    - Interpolating the SST to the grid')
        interp = LinearNDInterpolator(tri, sst.ravel())
        interp_sst = interp(X, Y)

        interp_sst[interp_sst<-100] = np.nan

        SST[day-1, :, :] = interp_sst

    return(SST)

def read_month_MUR42_SST_data_to_grid(data_folder, year, month, n_days, X, Y):

    SST = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    SIF = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    for day in range(1,n_days+1):
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

def read_month_wind_velocity_data_to_grid(data_folder, year, month, n_days, X, Y):

    U = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))
    V = np.zeros((n_days, np.shape(X)[0], np.shape(Y)[1]))

    for day in range(1,n_days+1):
        file_path = os.path.join(data_folder,'Global','Surface Winds',
                                 'CCMP_Wind_Analysis_'+str(year)+'{:02d}'.format(month)+'{:02d}'.format(day)+'_V03.1_L4.nc')
        ds = nc4.Dataset(file_path)
        lon = ds.variables['longitude'][:]
        lat = ds.variables['latitude'][:]
        u = ds.variables['uwnd'][:,:,:]
        v = ds.variables['vwnd'][:,:,:]
        ds.close()

        u = np.mean(u,axis=0)
        v = np.mean(v,axis=0)

        lon_indices = np.logical_and(lon>260, lon<360)
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

    for day in range(1,n_days+1):
        # ESACCI-OC-L3S-CHLOR_A-MERGED-1D_DAILY_4km_GEO_PML_OCx-20000801-fv6.0.nc
        file_path = os.path.join(data_folder,#'Global','Chlorophyll',
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

        interp_chl = griddata(points, chl.ravel(), (X, Y), method='nearest')

        interp_chl[interp_chl<-100] = np.nan

        Chl[day-1, :, :] = interp_chl

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

    output_file = os.path.join(project_folder, 'Data', '15km Interpolated', subset, str(year),
                               subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')

    return os.path.isfile(output_file)

def write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, subset, data_grids, data_grid_names):

    days = np.arange(1,n_days+1)

    if str(year) not in os.listdir(os.path.join(project_folder, 'Data', '15km Interpolated', subset)):
        os.mkdir(os.path.join(project_folder, 'Data', '15km Interpolated', subset, str(year)))

    output_file = os.path.join(project_folder, 'Data', '15km Interpolated', subset, str(year),
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


bedmachine_folder = '/Users/mike/Documents/Research/Data Repository'

data_folder = '/Volumes/CoOL/Data_Repository'
data_folder = '/Users/mike/Documents/Research/Data Repository'

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology'

resolution = 15

X, Y = create_domain(resolution)

if 'Greenland_domain_'+str(resolution)+'km.nc' not in os.listdir(os.path.join(project_folder,'Data',str(resolution)+'km Interpolated')):
    X, Y = create_domain(resolution)
    print(' - Reading in the Bedmachine Grid to Domain')
    X, Y, Bed = read_bedmachine_to_grid(data_folder, X, Y)
    Depth = Bed*-1
    write_domain_to_nc(project_folder, resolution, X, Y, Depth)
else:
    X, Y, Depth = read_domain_from_nc(project_folder, resolution)


make_SST = True
make_sea_ice = False
make_velocity = False
make_Chl = False

# C = plt.pcolormesh(X,Y,Bed)
# plt.colorbar(C)
# plt.show()

if make_SST:
    mur_tri_file = os.path.join(project_folder, 'Data', '15km Interpolated', 'MUR41_SST_triangulation.pkl')
    if os.path.isfile(mur_tri_file):
        print(' - Loading the MUR41 SST triangulation from file')
        with open(mur_tri_file, 'rb') as f:
            mur_41_tri = pickle.load(f)
    else:
        mur_41_tri = compute_MUR41_SST_triangulation(data_folder, 2010, 1, 1, X, Y)
        # now pickle the tri object so we don't have to compute it every time
        with open(mur_tri_file, 'wb') as f:
            pickle.dump(mur_41_tri, f)


for year in range(2010,2021):
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
        if make_velocity and not file_exists:
            print('    - Interpolating the velocity data')
            try:
                U, V = read_month_velocity_data_to_grid(data_folder, year, month, n_days, X, Y)
                write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Velocity', [U,V], ['U','V'])
                # plt.subplot(1,2,1)
                # C = plt.pcolormesh(X, Y, U[0,:,:])
                # plt.colorbar(C)
                # plt.subplot(1, 2, 2)
                # C = plt.pcolormesh(X, Y, V[0,:,:])
                # plt.colorbar(C)
                # plt.show()
            except KeyboardInterrupt:
                print(' - Keyboard Interrupt detected, stopping code execution')
                break
            except:
                print('        - Error in velocity data for this month')

        #############################################################################################
        # Temperature

        file_exists = check_file_existence(project_folder, year, month, 'Sea Surface Temperature')
        if make_velocity and not file_exists:
            print('    - Interpolating the sea surface temperature data')
            # try:
            SST = read_month_MUR41_SST_data_to_grid(data_folder, year, month, n_days, X, Y, mur_41_tri)
            write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'SST', [SST], ['SST'])

            C = plt.pcolormesh(X, Y, SST[0,:,:], cmap='turbo')
            plt.colorbar(C)
            plt.show()

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


        #############################################################################################
        # Chl Data

        if make_Chl:

            # Chl = read_month_AQUA_MODIS_Chl_data_to_grid(data_folder, year, month, n_days, X, Y)
            # write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Chlorophyll', [Chl], ['Chl'])

            file_exists = check_file_existence(project_folder, year, month, 'Chlorophyll')
            if year< 1987:
                if not file_exists:
                    Chl, data_found = read_month_CZCS_Chl_data_to_grid(data_folder, year, month, n_days, X, Y)
                    if data_found:
                        write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Chlorophyll', [Chl], ['Chl'])
                        # C = plt.pcolormesh(X, Y, np.nanmean(Chl, axis=0), cmap='turbo')
                        # plt.colorbar(C)
                        # plt.show()
            if year>=1997:
                if True:#not file_exists:
                    Chl = read_month_OCCCI_Chl_data_to_grid(data_folder, year, month, n_days, X, Y)
                    write_data_to_nc(project_folder, year, month, n_days, X, Y, Depth, 'Chlorophyll', [Chl], ['Chl'])
                    C = plt.pcolormesh(X, Y, np.nanmean(Chl, axis=0), cmap='turbo')
                    plt.colorbar(C)
                    plt.show()

        # plt.subplot(1,2,1)
        # C = plt.pcolormesh(X, Y, U_wind[0,:,:])
        # plt.colorbar(C)
        # plt.subplot(1, 2, 2)
        # C = plt.pcolormesh(X, Y, V_wind[0,:,:])
        # plt.colorbar(C)
        # plt.show()


