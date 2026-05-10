
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
import cartopy
import cartopy.crs as ccrs
from pyproj import Transformer
import datetime
from matplotlib.pyplot import GridSpec
from matplotlib.patches import Rectangle
import cmocean.cm as cm

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

def create_monthly_panels(project_folder,year,month,check_existing,bm_X, bm_Y, bed):

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # SST
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Interpolated', 'SST',
                                str(year), f'SST_{year}{month:02d}.nc'))
    X = ds.variables['X'][:,:]
    Y = ds.variables['Y'][:,:]
    SST = ds.variables['SST'][:,:,:]
    ds.close()

    # Sea Ice
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Interpolated', 'Sea_Ice',
                                str(year), f'Sea_Ice_{year}{month:02d}.nc'))
    Sea_Ice = ds.variables['seaice_fraction'][:,:,:]
    ds.close()

    # wind
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Interpolated', 'Wind',
                                str(year), f'Wind_{year}{month:02d}.nc'))
    U_wind = ds.variables['U_wind'][:,:,:]
    V_wind = ds.variables['V_wind'][:,:,:]
    ds.close()

    # velocity
    ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Interpolated', 'Velocity',
                                str(year), f'Velocity_{year}{month:02d}.nc'))
    U = ds.variables['U'][:,:,:]
    V = ds.variables['V'][:,:,:]
    ds.close()

    # points = np.column_stack([X.ravel(), Y.ravel()])
    # reprojected_points = reproject_polygon(points, 3413, 4326)
    # longitude = reprojected_points[:,0].reshape(X.shape)
    # latitude = reprojected_points[:,1].reshape(X.shape)
    #
    # print(np.min(longitude), np.max(longitude), np.min(latitude), np.max(latitude))

    for day in range(1, days_in_month[month-1]+1):
        print(f' - Working on {year}/{month:02d}/{day:02d}')
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(8,9))

        gs2 = GridSpec(19, 17, left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05)

        land_mask = (bed == 0.0).astype(int)
        land_mask = np.ma.masked_where(land_mask == 0, land_mask)

        # sst
        ax1 = fig.add_subplot(gs2[:8, :7])
        C = plt.pcolormesh(X, Y, SST[day - 1, :, :], vmin=273, vmax=290, cmap='plasma')
        plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
        cbar.set_label('Sea Surface Temperature (K)', rotation=90)

        # sea ice
        ax1 = fig.add_subplot(gs2[:8, 9:-1])
        C = plt.pcolormesh(X, Y, Sea_Ice[day - 1, :, :], vmin=0, vmax=1, cmap=cm.ice)
        plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
        cbar.set_label('Sea Ice Concentration', rotation=90)

        # velocity
        ax2 = fig.add_subplot(gs2[9:-2, :7])
        skip = 5
        C = ax2.pcolormesh(X, Y, np.sqrt(U[day - 1, :, :]**2 + V[day - 1, :, :]**2),
                           vmin=0, vmax=1, cmap=cm.tempo_r)
        ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], U[day - 1, ::skip, ::skip], V[day - 1, ::skip, ::skip],
                     scale=10, color='white')
        plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
        ax2.set_xticks([])
        ax2.set_yticks([])
        cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
        cbar.set_label('Ocean Current Speed (m/s)', rotation=90)

        # wind
        ax2 = fig.add_subplot(gs2[9:-2, 9:-1])
        skip = 5
        C = ax2.pcolormesh(X, Y, np.sqrt(U_wind[day - 1, :, :]**2 + V_wind[day - 1, :, :]**2), vmin=0, vmax=25, cmap='viridis')
        ax2.quiver(X[::skip, ::skip], Y[::skip, ::skip], U_wind[day - 1, ::skip, ::skip], V_wind[day - 1, ::skip, ::skip],
                   scale=300, color='white')
        plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
        ax2.set_xticks([])
        ax2.set_yticks([])
        cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
        cbar.set_label('Wind Speed (m/s)', rotation=90)

        # time
        ax3 = fig.add_subplot(gs2[-1, 1:-3])
        day_of_year = day + np.sum(days_in_month[:month-1])
        rect = Rectangle((0, 0), day_of_year, 1, fc='silver', ec='white')
        ax3.add_patch(rect)
        ax3.set_xlim([0, 366 if year%4==0 else 365])
        ax3.set_ylim([0, 1])
        for i in range(1, 12):
            plt.plot([np.sum(days_in_month[:i]) + 1, np.sum(days_in_month[:i]) + 1], [0, 1], 'w--', linewidth=0.5)
        ax3.set_xticks(np.arange(15, 350, 30))
        ax3.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
        ax3.set_yticks([])

        output_file = os.path.join(project_folder, 'Figures', 'Maps','Independent Observations', 'panels',
                                   f'Independent_Obs_{year}_{month:02d}_{day:02d}.png')
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

def compile_panels_to_movie(project_folder,year):
    pwd = os.getcwd()

    panels_dir = os.path.join(project_folder, 'Figures', 'Maps','Independent Observations', 'panels')

    # get a list of the file names
    all_file_names = []
    for file_name in os.listdir(panels_dir):
        if file_name[0]!='.' and file_name[-4:]=='.png':
            all_file_names.append(file_name)
    all_file_names = sorted(all_file_names)

    # make a temporary dir where we will link all available images and go there
    panels_dir_tmp = os.path.join(project_folder, 'Figures', 'Maps','Independent Observations', 'panels_tmp')
    os.mkdir(panels_dir_tmp)
    os.chdir(panels_dir_tmp)

    # link all of the images
    for ff in range(len(all_file_names)):
        os.system('ln -s ' + '../panels/'+all_file_names[ff]+' panel_'+'{:05d}'.format(ff)+'.png')

    output_name = 'Independent_Observations_'+str(year)+'.mp4'

    os.system("ffmpeg -r 5 -i panel_%05d.png -vcodec mpeg4 -b 3M -y " + output_name)
    os.rename(output_name, os.path.join('..', output_name))

    os.chdir(pwd)
    os.system('rm -rf ' + panels_dir_tmp)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

home_folder = os.path.expanduser('~')
bedmachine_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

resolution = 15

check_existing = True

year = 2019

print(' - Reading in the bedmachine data')
bm_X, bm_Y, bed = read_bedmachine_to_grid(bedmachine_folder)

for month in range(3,10):
    create_monthly_panels(project_folder, year, month, check_existing, bm_X, bm_Y, bed)

print(' - Compiling panels into movie')
compile_panels_to_movie(project_folder,year)
