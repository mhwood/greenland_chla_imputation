
import numpy as np
import torch
from torch.utils.data import Dataset
import netCDF4 as nc4
import os


def generate_year_months(start_year, start_month, end_year, end_month):
    year_months = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if (year == start_year and month < start_month) or (year == end_year and month > end_month):
                continue
            year_months.append((year, month))
    return year_months


class ChlDatasetLazy(Dataset):
    def __init__(self, project_folder, resolution, model_version,
                 start_year, end_year,
                 normalize_stats, device,
                 time_window=3, hide_prob=0.3, epsilon=1e-6,
                 fill_value=0.0, training=True, printing=True, verbose=False):

        self.project_folder = project_folder
        self.model_version = model_version
        self.start_year = start_year
        self.end_year = end_year
        self.resolution = resolution
        self.time_window = time_window
        self.hide_prob = hide_prob
        self.epsilon = epsilon
        self.fill_value = fill_value
        self.training = training
        self.printing = printing
        self.device = device
        self.verbose = verbose

        # make the time arrays
        self.year_months = generate_year_months(start_year, 1, end_year, 12)
        self.generate_time_index_indices()

        # load the domain
        self.var_dict = {}
        self.read_model_domain()

        # store the variable information
        self.normalize_stats = normalize_stats
        if model_version=='Unet1':
            self.var_names = ['Chl', 'SST', 'U_wind', 'V_wind', 'U', 'V', 'Lon', 'Lat', 'DOY_sin','DOY_cos']
        if model_version=='Unet2':
            self.var_names = ['Chl', 'SST', 'U_wind', 'V_wind', 'Lon', 'Lat', 'DOY_sin','DOY_cos']

    def generate_time_index_indices(self):

        self.index_to_date = {}
        self.date_to_index = {}

        for year_month in self.year_months:
            year = year_month[0]
            month = year_month[1]
            n_days_month = 31
            if month in [4, 6, 9, 11]:
                n_days_month = 30
            elif month == 2:
                n_days_month = 28
                if year % 4 == 0:
                    n_days_month = 29

            for day in range(1, n_days_month + 1):
                date_tuple = (year, month, day)
                index = len(self.index_to_date)
                self.index_to_date[index] = date_tuple
                self.date_to_index[date_tuple] = index

        self.n_days = len(self.index_to_date)

    def read_indexed_netcdf_stack(self, indices, varname, vector=False):

        if self.verbose:
            print('         - Global indicies:', indices)

        if varname=='Chl':
            read_varname = 'Chlorophyll'
        else:
            read_varname = varname

        last_month_tuple = (-1, -1)
        for idx in indices:
            date_tuple = self.index_to_date[idx]
            year_month = (date_tuple[0], date_tuple[1])
            year = date_tuple[0]
            month = date_tuple[1]
            day = date_tuple[2]

            if year_month!= last_month_tuple:
                # there is no chl data between 1986/7 and 1997/8 so we just
                # generate nans for these months - let check to see if this is the case
                nan_chl = False
                if read_varname == 'Chlorophyll':
                    if year==1986 and month>=7:
                        nan_chl = True
                    if year>1986 and year<1997:
                        nan_chl = True
                    if year==1997 and month<=8:
                        nan_chl = True
                if nan_chl:
                    print('    - No Chlorophyll data for', year, month, day, '- filling with NaNs')
                    # var_grid= np.full((1, 184, 103), np.nan, dtype=np.float32)
                    var_grid = np.full((1, np.shape(self.ocean_mask)[0], np.shape(self.ocean_mask)[1]), np.nan, dtype=np.float32)
                else:
                    file_path = os.path.join(self.project_folder, 'Data', str(self.resolution)+'km Interpolated',
                                             read_varname, str(year), read_varname +'_' + str(year) +'{:02d}'.format(month) +'.nc')
                    ds = nc4.Dataset(file_path)
                    if vector:
                        if varname== 'Wind':
                            u_grid = ds.variables['U_wind'][:,:,:]
                            v_grid = ds.variables['V_wind'][:,:,:]
                        if varname== 'Velocity':
                            u_grid = ds.variables['U'][:,:,:]
                            v_grid = ds.variables['V'][:,:,:]
                    else:
                        if varname=='Sea_Ice':
                            nc_readname = 'seaice_fraction'
                        elif varname=='Chl':
                            nc_readname = 'Chl'
                        else:
                            nc_readname = varname
                        var_grid = ds.variables[nc_readname][:,:,:]
                    ds.close()
                last_month_tuple = year_month

            first_month_index = self.date_to_index[(year, month, 1)]
            idx_in_month = idx - first_month_index
            if self.verbose:
                print('    - Reading data for index:', idx, 'which corresponds to date:', date_tuple, 'from file:',
                  file_path.split('/')[-1], 'with idx_in_month:', idx_in_month)

            if idx == indices[0]:
                if vector:
                    u_stacked = u_grid[idx_in_month:idx_in_month+1,:,:]
                    v_stacked = v_grid[idx_in_month:idx_in_month+1,:,:]
                else:
                    stacked = var_grid[idx_in_month:idx_in_month+1,:,:]
            else:
                if vector:
                    u_stacked = np.concatenate((u_stacked, u_grid[idx_in_month:idx_in_month+1,:,:]), axis=0)
                    v_stacked = np.concatenate((v_stacked, v_grid[idx_in_month:idx_in_month+1,:,:]), axis=0)
                else:
                    stacked = np.concatenate((stacked, var_grid[idx_in_month:idx_in_month+1,:,:]), axis=0)

        if self.printing and self.verbose:
            if vector:
                print('    - Finished reading and stacking netCDF files. Final shapes: U:', u_stacked.shape, 'V:', v_stacked.shape)
            else:
                print('    - Finished reading and stacking netCDF files. Final shape:', stacked.shape)

        if vector:
            if varname== 'Wind':
                self.var_dict['U_wind'] = u_stacked
                self.var_dict['V_wind'] = v_stacked
            if varname== 'Velocity':
                self.var_dict['U'] = u_stacked
                self.var_dict['V'] = v_stacked
        else:
            if varname=='Sea_Ice':
                self.var_dict['Sea_Ice'] = stacked
            elif varname=='Chlorophyll':
                self.var_dict['Chl'] = stacked
            else:
                self.var_dict[varname] = stacked

    def read_model_domain(self):
        file_path = os.path.join(self.project_folder, 'Data', str(self.resolution)+'km Interpolated',
                                 'Greenland_domain_'+str(self.resolution)+'km.nc')
        ds = nc4.Dataset(file_path)
        ocean_mask = ds.variables['mask'][:, :]
        Lon = ds.variables['Lon'][:, :]
        Lat = ds.variables['Lat'][:, :]
        ds.close()

        self.ocean_mask = ocean_mask.astype(bool)

        stack_size = self.time_window*2 +1

        # repeat X and Y to be the full size of the other variables (one per day)
        self.Lon = np.repeat(Lon[None, :, :], stack_size, axis=0)
        self.Lat = np.repeat(Lat[None, :, :], stack_size, axis=0)

        # add to the var dict
        self.var_dict['Lon'] = self.Lon
        self.var_dict['Lat'] = self.Lat

    def generate_DOY_fields(self, indices):

        doys_frac = np.zeros((self.time_window*2+1, ), dtype=np.float32)
        days_counted = 0
        for idx in indices:
            date_tuple = self.index_to_date[idx]
            year = date_tuple[0]

            if year%4==0:
                n_days_year = 366
            else:
                n_days_year = 365

            first_year_index = self.date_to_index[(year, 1, 1)]
            doy = idx - first_year_index + 1
            # print(date_tuple, idx, doy)
            doys_frac[days_counted] = doy / n_days_year
            days_counted+=1

        # print(doys_frac)
        doy_sin = np.sin(2 * np.pi * doys_frac)
        doy_cos = np.cos(2 * np.pi * doys_frac)

        self.DOY_sin = np.repeat(doy_sin[:, None, None], self.var_dict['SST'].shape[1], axis=1)
        self.DOY_sin = np.repeat(self.DOY_sin, self.var_dict['SST'].shape[2], axis=2)
        self.DOY_cos = np.repeat(doy_cos[:, None, None], self.var_dict['SST'].shape[1], axis=1)
        self.DOY_cos = np.repeat(self.DOY_cos, self.var_dict['SST'].shape[2], axis=2)

        # add to the var grid
        self.var_dict['DOY_sin'] = self.DOY_sin
        self.var_dict['DOY_cos'] = self.DOY_cos

    def __len__(self):
        return self.n_days

    def normalize(self, arr, varname):
        mean = self.normalize_stats[varname]["mean"]
        std = self.normalize_stats[varname]["std"]
        return (arr - mean) / (std + 1e-6)

    def make_chl_training_input(self, chl_log, observed_mask):
        random_hide = np.random.rand(*chl_log.shape) < self.hide_prob

        target_mask = observed_mask.astype(bool) & random_hide
        input_mask = observed_mask.astype(bool) & ~random_hide

        chl_input = np.full_like(chl_log, self.fill_value, dtype=np.float32)
        chl_input[input_mask] = chl_log[input_mask]

        return (
            chl_input.astype(np.float32),
            input_mask.astype(np.float32),
            target_mask.astype(np.float32),
        )

    def __getitem__(self, global_idx):

        # if idx is too close to the start or end, adjust it to ensure we have a full temporal window
        # this will cause a few repeats but maybe its ok given the big data set
        if global_idx<self.time_window:
            global_idx = self.time_window
        if global_idx>=len(self)-self.time_window:
            global_idx = len(self)-self.time_window-1

        center_local_idx = self.time_window

        ########################################################################################
        # read in the arrays from the files
        global_indices = np.arange(global_idx - self.time_window, global_idx + self.time_window + 1)
        for var_name in ['Chl','SST','Sea_Ice','Wind']:
            if var_name=='Wind' or var_name=='Velocity':
                vector = True
            else:
                vector = False
            self.read_indexed_netcdf_stack(global_indices,var_name,vector=vector)

        # recreate the DOY fields for this time step
        self.generate_DOY_fields(global_indices)

        seaice_mask_out = (self.var_dict['Sea_Ice'][center_local_idx, :, :] < 0.15).astype(int) * self.ocean_mask

        ########################################################################################
        # make output array with chl on the target time step (log chl) and mask of observed vs hidden chl
        chl_log = np.log(self.var_dict['Chl'][center_local_idx, :, :] + self.epsilon)
        observed_mask = np.isfinite(chl_log) & self.ocean_mask

        chl_log = np.where(np.isfinite(chl_log), chl_log, np.nan)
        chl_log = self.normalize(chl_log, 'Chl')
        chl_log = np.nan_to_num(
            chl_log,
            nan=self.fill_value,
            posinf=self.fill_value,
            neginf=self.fill_value,
        ).astype(np.float32)

        chl_log_input, input_mask, target_mask = self.make_chl_training_input(chl_log, observed_mask)

        ########################################################################################
        # make big input array with all channels and time steps (big X)
        channels = []

        # temporal window
        for local_idx in range(2*self.time_window + 1):

            seaice = self.var_dict['Sea_Ice'][local_idx, :, :]
            seaice_mask = (seaice<0.15).astype(int)
            seaice_mask = seaice_mask * self.ocean_mask

            # predictor channels
            for var_name in self.var_names:
                input_arr = self.var_dict[var_name]
                arr = input_arr[local_idx, :, :]

                arr = np.where(np.isfinite(arr), arr, np.nan)
                if var_name not in ['DOY_cos', 'DOY_sin']:
                    arr = self.normalize(arr, var_name)
                arr = np.nan_to_num(
                    arr,
                    nan=self.fill_value,
                    posinf=self.fill_value,
                    neginf=self.fill_value,
                ).astype(np.float32)

                arr = arr * self.ocean_mask * seaice_mask

                if var_name == 'Chl' and local_idx == center_local_idx:
                    arr = np.where(input_mask, arr, self.fill_value).astype(np.float32)
                channels.append(arr)

        # ocean mask channel
        channels.append(self.ocean_mask.astype(np.float32))

        # put it all together
        x = np.stack(channels, axis=0).astype(np.float32)

        y = chl_log[None, :, :].astype(np.float32)
        target_mask = target_mask[None, :, :].astype(np.float32)
        ocean_mask = self.ocean_mask[None, :, :].astype(np.float32)

        sample = {
            "x": torch.from_numpy(x).to(self.device),                     # [C, 184, 103]
            "target_log_chl": torch.from_numpy(y).to(self.device),        # [1, 184, 103]
            "target_mask": torch.from_numpy(target_mask).to(self.device), # [1, 184, 103]
            "ocean_mask": torch.from_numpy(ocean_mask).to(self.device),   # [1, 184, 103]
            "seaice_mask": torch.from_numpy(seaice_mask_out[None, :, :].astype(np.float32)).to(self.device)   # [1, 184, 103]
        }

        return sample