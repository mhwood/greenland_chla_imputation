
import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc4
from sklearn.metrics import r2_score
from matplotlib.gridspec import GridSpec

def generate_year_months(start_year, start_month, end_year, end_month):
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            year_months.append((year, month))
    return year_months

def read_training_testing_set_comparisons(project_folder, resolution, model_version, year_months):

    first_file = True

    for year_month in year_months:
        year = year_month[0]
        month = year_month[1]

        ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Interpolated', 'Chlorophyll',
                                      str(year), f'Chlorophyll_{year}{month:02d}.nc'))
        Chl_obs = ds.variables['Chl'][:, :, :]
        ds.close()

        ds = nc4.Dataset(os.path.join(project_folder, 'Data', '15km Model', model_version,
                                      str(year), f'Chl_Modeled_{model_version}_{year}{month:02d}.nc'))
        Chl_model = ds.variables['Chl'][:, :, :]
        ds.close()

        points = np.column_stack([Chl_obs.ravel(), Chl_model.ravel()])
        # remove nans
        non_nans = np.logical_and(~np.isnan(points[:,0]), ~np.isnan(points[:,1]))
        points = points[non_nans,:]
        points = np.log(points)

        if first_file:
            all_points = points
            first_file = False
        else:
            all_points = np.vstack((all_points, points))

    return(all_points)

def plot_monthly_R2_results(project_dir, model_version, subset, y_train_obs_sets, y_test_obs_sets, y_train_pred_sets, y_test_pred_sets):

    if subset == 'train':
        y_obs_sets = y_train_obs_sets
        y_pred_sets = y_train_pred_sets
    elif subset == 'test':
        y_obs_sets = y_test_obs_sets
        y_pred_sets = y_test_pred_sets

    y_min = -2
    y_max = 2

    fig = plt.figure(figsize=(10,9))

    panel_width = 1
    panel_height = 1

    gs = GridSpec(nrows=3*panel_height, ncols=panel_width*4, figure=fig, wspace=0.1, hspace=0.2,
                  left=0.05, right=0.95, top=0.95, bottom=0.05)

    for month in range(1,13):

        y_obs = y_obs_sets[month]
        y_pred = y_pred_sets[month]

        if month<=4:
            ax = fig.add_subplot(gs[:panel_height, (month-1)*panel_width:(month)*panel_width])
        elif month<=8 and month>4:
            ax = fig.add_subplot(gs[panel_height:2*panel_height, (month-5)*panel_width:(month-4)*panel_width])
        else:
            ax = fig.add_subplot(gs[2*panel_height:3*panel_height, (month-9)*panel_width:(month-8)*panel_width])

        ax.plot([y_min, y_max], [y_min, y_max], '--',color='silver')

        C = ax.plot(y_obs, y_pred, 'k.')
        ax.set_title('Month: ' + str(month))

        if month<9:
            ax.set_xticklabels([])
        if month not in [1,5,9]:
            ax.set_yticklabels([])

        ax.text(0.05, 0.95, 'N: ' + str(len(y_obs)), transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.90, 'RMSE: ' + str(round(np.sqrt(np.mean((y_obs - y_pred) ** 2)), 2)), transform=ax.transAxes, verticalalignment='top')
        ax.text(0.05, 0.85, 'R2: ' + str(round(r2_score(y_obs, y_pred), 2)), transform=ax.transAxes, verticalalignment='top')

        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)

    output_file = os.path.join(project_folder, 'Figures', model_version+'_monthly_' + subset + '_R2.png')
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

project_folder = '/Users/mike/Documents/Research/Projects/Greenland Chl Imputation'

model_version = 'XGB1'
resolution = 15

start_year = 2019
end_year = 2019

print(' - Reading the modeled and observed comparisons')
year_months = generate_year_months(start_year, 1, end_year, 12)

all_points = read_training_testing_set_comparisons(project_folder, resolution, model_version, year_months)

plt.plot(all_points[:,0], all_points[:,1], 'k.', alpha=0.1)
plt.plot([-2, 2], [-2, 2], '--', color='silver')
plt.title('Unet1 Modeled vs. Observed Chlorophyll (log scale)\nR2: ' + str(round(r2_score(all_points[:,0], all_points[:,1]), 2)) + ', RMSE: ' + str(round(np.sqrt(np.mean((all_points[:,0] - all_points[:,1]) ** 2)), 2)))
plt.show()



