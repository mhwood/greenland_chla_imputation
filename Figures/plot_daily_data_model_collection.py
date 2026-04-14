
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from matplotlib.gridspec import GridSpec
import cmocean.cm as cm

def read_all_data_subsets(project_folder, year, month):

    var_grids = {}

    for data_subset in ['SST','SSS','Sea_Ice','Surface_Wind','Velocity','Chlorophyll']:

        file_path = os.path.join(project_folder, 'Data', '15km Interpolated',
                                 data_subset, data_subset +'_' + str(year) +'{:02d}'.format(month) +'.nc')
        ds = nc4.Dataset(file_path)
        if data_subset=='SST':
            var_grids['SST'] = ds.variables['SST'][:,:,:]
            # print('SST',np.nanmin(var_grids['SST']), np.nanmax(var_grids['SST']))
        if data_subset=='SSS':
            var_grids['SSS'] = ds.variables['SSS'][:,:,:]
            # print('SSS',np.nanmin(var_grids['SSS']), np.nanmax(var_grids['SSS']))
        if data_subset=='Sea_Ice':
            var_grids['Sea_Ice'] = ds.variables['seaice_fraction'][:,:,:]
        if data_subset=='Surface_Wind':
            var_grids['U_wind'] = ds.variables['U_wind'][:,:,:]
            var_grids['V_wind'] = ds.variables['V_wind'][:, :, :]
            # print('U_wind',np.nanmin(var_grids['U_wind']), np.nanmax(var_grids['U_wind']))
        if data_subset=='Velocity':
            var_grids['U'] = ds.variables['U'][:,:,:]
            var_grids['V'] = ds.variables['V'][:, :, :]
            # print('V', np.nanmin(var_grids['V']), np.nanmax(var_grids['V']))
        if data_subset=='Chlorophyll':
            var_grids['Chl'] = ds.variables['Chl'][:,:,:]

    ds.close()

    return(var_grids)

def read_model_Chl(project_folder, model_version, year, month):

    file_path = os.path.join(project_folder, 'Data', '15km Interpolated',
                             'Chlorophyll_Modeled_'+model_version,'Chl_Modeled_'+model_version+'_' + str(year) +'{:02d}'.format(month) +'.nc')
    ds = nc4.Dataset(file_path)
    Chl_modeled = ds.variables['Chl_Modeled'][:,:,:]
    ds.close()

    return(Chl_modeled)

def add_colorbar(ax, units, cmap, vmin, vmax):
    cx = np.linspace(vmin,vmax,100)
    cy = np.array([0,1])
    CX, CY = np.meshgrid(cx, cy)
    ax.pcolormesh(CX,CY,CX, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    ax.set_xlabel(units)

def plot_data_model_grids(project_folder, model_version, var_grids, year, month, day):

    fig = plt.figure(figsize=(10,12))
    plt.style.use('dark_background')

    plot_rows = 6
    cb_rows = 1
    plot_cols = 3
    v_spacing = 2
    h_spacing = 1
    

    gs = GridSpec(3*(plot_rows+cb_rows)+2*v_spacing, 3*plot_cols+2*h_spacing)

    #################################################################
    # SST
    cmap = cm.thermal
    vmin = 271.15
    vmax = 290

    ax = fig.add_subplot(gs[:plot_rows, :plot_cols])
    ax.pcolormesh(var_grids['SST'][day-1,:,:], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('SST')
    ax = fig.add_subplot(gs[plot_rows:plot_rows+cb_rows, :plot_cols])
    add_colorbar(ax, 'C', cmap, vmin, vmax)

    if model_version!='XGB1':
        #################################################################
        # SSS
        cmap = 'viridis'
        vmin = 32
        vmax = 37

        ax = fig.add_subplot(gs[:plot_rows,
                             h_spacing+plot_cols:h_spacing+plot_cols*2])
        ax.pcolormesh(var_grids['SSS'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('SSS')
        ax = fig.add_subplot(gs[plot_rows:plot_rows + cb_rows,
                             h_spacing+plot_cols:h_spacing+plot_cols*2])
        add_colorbar(ax, 'psu', cmap, vmin, vmax)

    #################################################################
    # Seaice
    cmap = cm.ice
    vmin = 0
    vmax = 1

    ax = fig.add_subplot(gs[:plot_rows,
                         2*h_spacing + 2*plot_cols:2*h_spacing + plot_cols * 3])
    ax.pcolormesh(var_grids['Sea_Ice'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Sea Ice Conc.')
    ax = fig.add_subplot(gs[plot_rows:plot_rows + cb_rows,
                         2*h_spacing + 2*plot_cols:2*h_spacing + plot_cols * 3])
    add_colorbar(ax, 'm$^2$/m$^2$', cmap, vmin, vmax)

    #################################################################
    # Wind
    cmap = cm.balance
    vmin = -10
    vmax = 10

    ax = fig.add_subplot(gs[v_spacing+cb_rows+plot_rows:v_spacing+cb_rows+2*plot_rows,
                         :plot_cols])
    ax.pcolormesh(var_grids['U_wind'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Zonal Wind')
    ax = fig.add_subplot(gs[v_spacing+cb_rows+2*plot_rows:v_spacing+2*cb_rows+2*plot_rows,
                         :plot_cols])
    add_colorbar(ax, 'm/s', cmap, vmin, vmax)

    ax = fig.add_subplot(gs[v_spacing + cb_rows + plot_rows:v_spacing + cb_rows + 2 * plot_rows,
                         h_spacing+plot_cols:h_spacing+plot_cols*2])
    ax.pcolormesh(var_grids['V_wind'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Meridional Wind')
    ax = fig.add_subplot(gs[v_spacing + cb_rows + 2 * plot_rows:v_spacing + 2 * cb_rows + 2 * plot_rows,
                         h_spacing+plot_cols:h_spacing+plot_cols*2])
    add_colorbar(ax, 'm/s', cmap, vmin, vmax)

    #################################################################
    # Currents
    cmap = cm.balance
    vmin = -1
    vmax = 1

    ax = fig.add_subplot(gs[2*v_spacing + 2*cb_rows + 2*plot_rows:2*v_spacing + 2*cb_rows + 3*plot_rows, :plot_cols])
    ax.pcolormesh(var_grids['U'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Zonal Velocity')
    ax = fig.add_subplot(gs[2*v_spacing + 2*cb_rows + 3*plot_rows:2*v_spacing + 3*cb_rows + 3*plot_rows,
                         :plot_cols])
    add_colorbar(ax, 'm/s', cmap, vmin, vmax)

    ax = fig.add_subplot(gs[2*v_spacing + 2*cb_rows + 2*plot_rows:2*v_spacing + 2*cb_rows + 3*plot_rows,
                         h_spacing + plot_cols:h_spacing + plot_cols * 2])
    ax.pcolormesh(var_grids['V'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Meridional Velocity')
    ax = fig.add_subplot(gs[2*v_spacing + 2*cb_rows + 3*plot_rows:2*v_spacing + 3*cb_rows + 3*plot_rows,
                         h_spacing + plot_cols:h_spacing + plot_cols * 2])
    add_colorbar(ax, 'm/s', cmap, vmin, vmax)

    #################################################################
    # Chlorophyll
    cmap = 'turbo'
    vmin = 0
    vmax = 5

    ax = fig.add_subplot(gs[v_spacing+cb_rows+plot_rows:v_spacing+cb_rows+2*plot_rows,
                         2 * h_spacing + 2 * plot_cols:2 * h_spacing + plot_cols * 3])
    ax.pcolormesh(var_grids['Chl'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Chl Obs.')
    ax = fig.add_subplot(gs[v_spacing + cb_rows + 2 * plot_rows:v_spacing + 2 * cb_rows + 2 * plot_rows,
                         2 * h_spacing + 2 * plot_cols:2 * h_spacing + plot_cols * 3])
    add_colorbar(ax, 'mg/m$^3$', cmap, vmin, vmax)

    #################################################################
    # Chlorophyll Modeled
    cmap = 'turbo'
    vmin = 0
    vmax = 5

    ax = fig.add_subplot(gs[2*v_spacing + 2*cb_rows + 2*plot_rows:2*v_spacing + 2*cb_rows + 3*plot_rows,
                         2*h_spacing + 2*plot_cols:2*h_spacing + plot_cols * 3])
    ax.pcolormesh(var_grids['Chl_Modeled'][day - 1, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Chl Modeled.')
    ax = fig.add_subplot(gs[2*v_spacing + 2*cb_rows + 3*plot_rows:2*v_spacing + 3*cb_rows + 3*plot_rows,
                         2*h_spacing + 2*plot_cols:2*h_spacing + plot_cols * 3])
    add_colorbar(ax, 'mg/m$^3$', cmap, vmin, vmax)


    if 'Panels Model '+model_version not in os.listdir(os.path.join(project_folder, 'Figures', 'Remote Sensing Model')):
        os.mkdir(os.path.join(project_folder, 'Figures', 'Remote Sensing Model','Panels Model '+model_version))

    plt.savefig(os.path.join(project_folder, 'Figures', 'Remote Sensing Model', 'Panels Model '+model_version,
                             'Chl_Model_Data_' + str(year) +'{:02d}'.format(month) +'{:02d}'.format(day) +'.png'))
    plt.close(fig)


project_folder = '/Users/mike/Documents/Research/Projects/Greenland Biology'

model_version = 'XGB1'

year = 2025
month = 8
days = np.arange(1,32).tolist()

var_grids = read_all_data_subsets(project_folder, year, month)

Chl_modeled = read_model_Chl(project_folder, model_version, year, month)
var_grids['Chl_Modeled'] = Chl_modeled

for day in days:
    print(' - Plotting '+str(year)+'/'+str(month)+'/'+str(day))
    plot_data_model_grids(project_folder, model_version, var_grids, year, month, day)




