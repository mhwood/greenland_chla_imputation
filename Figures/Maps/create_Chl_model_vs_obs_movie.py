
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

def create_monthly_panels(project_folder,year,month,check_existing,bm_X, bm_Y, bed, model_version):

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Interpolated', 'Chlorophyll',
                                str(year), f'Chlorophyll_{year}{month:02d}.nc'))
    X = ds.variables['X'][:,:]
    Y = ds.variables['Y'][:,:]
    Chl_obs = ds.variables['Chl'][:,:,:]
    ds.close()

    ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Model', model_version,
                                  str(year), f'Chl_Modeled_{model_version}_{year}{month:02d}.nc'))
    Chl_model = ds.variables['Chl'][:, :, :]
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
        fig = plt.figure(figsize=(13,9))

        gs2 = GridSpec(17, 29, left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.05)

        # map
        ax1 = fig.add_subplot(gs2[:-2, :14])
        # model_mask = ~np.isnan(Chl_model[day - 1, :, :])
        # model_mask = np.ma.masked_where(~model_mask, model_mask)
        # plt.pcolormesh(X, Y, model_mask, cmap='gray', zorder=1, vmin=0, vmax=1)
        # C = plt.pcolormesh(X, Y, Chl_model[day - 1, :, :], vmin=0, vmax=5, cmap='turbo', zorder=2)
        C = plt.pcolormesh(X, Y, Chl_obs[day - 1, :, :], vmin=0, vmax=5, cmap='turbo',zorder=3)
        land_mask = (bed == 0.0).astype(int)
        land_mask = np.ma.masked_where(land_mask==0, land_mask)
        plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
        ax1.set_xticks([])
        ax1.set_yticks([])
        # cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
        # cbar.set_label('Chlorophyll-a Concentration', rotation=90)
        ax1.set_title('Observations')

        # map
        ax1 = fig.add_subplot(gs2[:-2, 14:28])
        # model_mask = ~np.isnan(Chl_model[day - 1, :, :])
        # model_mask = np.ma.masked_where(~model_mask, model_mask)
        # plt.pcolormesh(X, Y, model_mask, cmap='gray', zorder=1, vmin=0, vmax=1)
        C = plt.pcolormesh(X, Y, Chl_model[day - 1, :, :], vmin=0, vmax=5, cmap='turbo', zorder=2)
        # C = plt.pcolormesh(X, Y, Chl_obs[day - 1, :, :], vmin=0, vmax=5, cmap='turbo',zorder=3)
        land_mask = (bed == 0.0).astype(int)
        land_mask = np.ma.masked_where(land_mask == 0, land_mask)
        plt.pcolormesh(bm_X, bm_Y, land_mask, cmap='gray', zorder=10, vmin=0, vmax=2)
        plt.contour(bm_X, bm_Y, bed, levels=[0.01], colors='k', zorder=11)
        ax1.set_xticks([])
        ax1.set_yticks([])
        cbar = plt.colorbar(C, fraction=0.026, pad=0.04)
        cbar.set_label('Chlorophyll-a Concentration', rotation=90)
        ax1.set_title('Model')

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

        output_file = os.path.join(project_folder, 'Figures', 'Maps','Chlorophyll Model vs Obs',model_version, 'panels',
                                   f'Chlorophyll_Model_{year}_{month:02d}_{day:02d}.png')
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

def compile_panels_to_movie(project_folder,year, model_version):
    pwd = os.getcwd()

    panels_dir = os.path.join(project_folder, 'Figures', 'Maps','Chlorophyll Model vs Obs',model_version, 'panels')

    # get a list of the file names
    all_file_names = []
    for file_name in os.listdir(panels_dir):
        if file_name[0]!='.' and file_name[-4:]=='.png':
            all_file_names.append(file_name)
    all_file_names = sorted(all_file_names)

    # make a temporary dir where we will link all available images and go there
    panels_dir_tmp = os.path.join(project_folder, 'Figures', 'Maps','Chlorophyll Model vs Obs',model_version, 'panels_tmp')
    os.mkdir(panels_dir_tmp)
    os.chdir(panels_dir_tmp)

    # link all of the images
    for ff in range(len(all_file_names)):
        os.system('ln -s ' + '../panels/'+all_file_names[ff]+' panel_'+'{:05d}'.format(ff)+'.png')

    output_name = 'Chlorophyll_Model_vs_Obs_'+str(year)+'.mp4'

    os.system("ffmpeg -r 5 -i panel_%05d.png -vcodec mpeg4 -b 3M -y " + output_name)
    os.rename(output_name, os.path.join('..', output_name))

    os.chdir(pwd)
    os.system('rm -rf ' + panels_dir_tmp)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

home_folder = os.path.expanduser('~')
bedmachine_folder = os.path.join(home_folder, 'Documents', 'Research', 'Data Repository')

model_version = 'Unet2'

resolution = 15

check_existing = True

year = 2019

print(' - Reading in the bedmachine data')
bm_X, bm_Y, bed = read_bedmachine_to_grid(bedmachine_folder)

# for month in range(5,6):
#     create_monthly_panels(project_folder, year, month, check_existing, bm_X, bm_Y, bed, model_version)

print(' - Compiling panels into movie')
compile_panels_to_movie(project_folder,year, model_version)
