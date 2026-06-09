
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


class ChlDatasetSector(Dataset):
    def __init__(self, project_folder, resolution, model_version,
                 start_year, end_year,
                 normalize_stats, device, data_folder=None,
                 sector_ll_row=None, sector_ll_col=None, subset_size=None,
                 start_month=1, end_month=12,
                 time_window=3, hide_prob=0.3, epsilon=1e-6,
                 fill_value=0.0, training=True, printing=True):

        self.project_folder = project_folder
        self.model_version = model_version
        self.start_year = start_year
        self.end_year = end_year
        self.start_month = start_month
        self.end_month = end_month
        self.resolution = resolution
        self.time_window = time_window
        self.hide_prob = hide_prob
        self.epsilon = epsilon
        self.fill_value = fill_value
        self.training = training
        self.printing = printing
        self.device = device
        # self.check_target_nans = check_target_nans
        self.sector_ll_row = sector_ll_row
        self.sector_ll_col = sector_ll_col
        self.subset_size = subset_size

        if data_folder:
            self.data_folder=data_folder
        else:
            self.data_folder=project_folder

        self.year_months = generate_year_months(start_year, start_month, end_year, end_month)

        self.var_dict = {}
        self.normalize_stats = normalize_stats
        if len(normalize_stats)==0:
            self.compute_normalize_stats = True
        else:
            self.compute_normalize_stats = False
            print("Using provided normalization stats:")
            for var_name, stats in self.normalize_stats.items():
                print(f" - {var_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        self.read_netcdf_stack(self.data_folder, self.resolution, 'SST')
        self.read_netcdf_stack(self.data_folder, self.resolution, 'Sea_Ice')
        self.read_netcdf_stack(self.data_folder, self.resolution, 'Wind', vector=True)
        if model_version == 'Unet1':
            self.read_netcdf_stack(self.data_folder, self.resolution, 'Velocity', vector=True)
        self.read_netcdf_stack(self.data_folder, self.resolution, 'Chlorophyll')
        self.read_model_domain(self.project_folder, self.resolution)
        self.generate_DOY_fields()

        if model_version=='Unet1':
            self.var_names = ['Chl', 'SST', 'U_wind', 'V_wind', 'U', 'V', 'Lon', 'Lat', 'DOY_sin','DOY_cos']
        if model_version=='Unet2':
            self.var_names = ['Chl', 'SST', 'U_wind', 'V_wind', 'Lon', 'Lat', 'DOY_sin','DOY_cos']

    def read_netcdf_stack(self, project_folder, resolution, varname, vector=False):

        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        if self.printing:
            print(f' - Reading {varname} data from netCDF files...')

        first_file = True
        for year_month in self.year_months:
            year = year_month[0]
            month = year_month[1]

            # there is no chl data between 1986/7 and 1997/8 so we just
            # generate nans for these months
            nan_chl = False
            if varname == 'Chlorophyll':
                if year==1986 and month>=7:
                    nan_chl = True
                if year>1986 and year<1997:
                    nan_chl = True
                if year==1997 and month<=8:
                    nan_chl = True
            if nan_chl:
                print('    - No Chlorophyll data for', year, month, '- filling with NaNs')
                n_days_month = days_in_month[month-1]
                if year%4==0 and month==2:
                    n_days_month = 29
                var_grid= np.full((n_days_month, np.shape(self.ocean_mask)[0], np.shape(self.ocean_mask)[1]),
                                  np.nan, dtype=np.float32)
            else:
                file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                         varname, str(year), varname +'_' + str(year) +'{:02d}'.format(month) +'.nc')
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
                        readname = 'seaice_fraction'
                    elif varname=='Chlorophyll':
                        readname = 'Chl'
                    else:
                        readname = varname
                    var_grid = ds.variables[readname][:,:,:]
                ds.close()

                if vector:
                        u_grid = u_grid[:, self.sector_ll_row:self.sector_ll_row+self.subset_size[0],
                                        self.sector_ll_col:self.sector_ll_col+self.subset_size[1]]
                        v_grid = v_grid[:, self.sector_ll_row:self.sector_ll_row+self.subset_size[0],
                                        self.sector_ll_col:self.sector_ll_col+self.subset_size[1]]
                else:
                    var_grid = var_grid[:, self.sector_ll_row:self.sector_ll_row+self.subset_size[0],
                                        self.sector_ll_col:self.sector_ll_col+self.subset_size[1]]

            if first_file:
                if vector:
                    u_stacked = u_grid
                    v_stacked = v_grid
                else:
                    stacked = var_grid
                first_file = False
            else:
                if vector:
                    u_stacked = np.concatenate((u_stacked, u_grid), axis=0)
                    v_stacked = np.concatenate((v_stacked, v_grid), axis=0)
                else:
                    stacked = np.concatenate((stacked, var_grid), axis=0)

        if self.printing:
            if vector:
                print('    - Finished reading and stacking netCDF files. Final shapes: U:', u_stacked.shape, 'V:', v_stacked.shape)
            else:
                print('    - Finished reading and stacking netCDF files. Final shape:', stacked.shape)

        if vector:
            u_var_mean = np.mean(u_stacked[np.isfinite(u_stacked)])
            u_var_std = np.std(u_stacked[np.isfinite(u_stacked)])
            v_var_mean = np.mean(v_stacked[np.isfinite(v_stacked)])
            v_var_std = np.std(v_stacked[np.isfinite(v_stacked)])
        else:
            var_mean = np.mean(stacked[np.isfinite(stacked)])
            var_std = np.std(stacked[np.isfinite(stacked)])

        if vector:
            if varname== 'Wind':
                self.var_dict['U_wind'] = u_stacked
                self.var_dict['V_wind'] = v_stacked
                if self.compute_normalize_stats:
                    self.normalize_stats['U_wind'] = {"mean": u_var_mean, "std": u_var_std}
                    self.normalize_stats['V_wind'] = {"mean": v_var_mean, "std": v_var_std}
            if varname== 'Velocity':
                self.var_dict['U'] = u_stacked
                self.var_dict['V'] = v_stacked
                if self.compute_normalize_stats:
                    self.normalize_stats['U'] = {"mean": u_var_mean, "std": u_var_std}
                    self.normalize_stats['V'] = {"mean": v_var_mean, "std": v_var_std}
        else:
            if varname=='Sea_Ice':
                self.var_dict['Sea_Ice'] = stacked
                if self.compute_normalize_stats:
                    self.normalize_stats['Sea_Ice'] = {"mean": var_mean, "std": var_std}
            elif varname=='Chlorophyll':
                self.var_dict['Chl'] = stacked
                if self.compute_normalize_stats:
                    self.normalize_stats['Chl'] = {"mean": var_mean, "std": var_std}
            else:
                self.var_dict[varname] = stacked
                if self.compute_normalize_stats:
                    self.normalize_stats[varname] = {"mean": var_mean, "std": var_std}

        if varname=='SST':
            self.n_days = self.var_dict['SST'].shape[0]

    def read_model_domain(self, project_folder, resolution):
        file_path = os.path.join(project_folder, 'Data', str(resolution)+'km Interpolated',
                                 'Greenland_domain_'+str(resolution)+'km.nc')
        ds = nc4.Dataset(file_path)
        ocean_mask = ds.variables['mask'][:, :]
        Lon = ds.variables['Lon'][:, :]
        Lat = ds.variables['Lat'][:, :]
        ds.close()

        Lon = Lon[self.sector_ll_row:self.sector_ll_row+self.subset_size[0],
                                        self.sector_ll_col:self.sector_ll_col+self.subset_size[1]]
        Lat = Lat[self.sector_ll_row:self.sector_ll_row+self.subset_size[0],
                                        self.sector_ll_col:self.sector_ll_col+self.subset_size[1]]
        ocean_mask = ocean_mask[self.sector_ll_row:self.sector_ll_row+self.subset_size[0],
                                        self.sector_ll_col:self.sector_ll_col+self.subset_size[1]]

        self.ocean_mask = ocean_mask.astype(bool)

        # repeat X and Y to be the full size of the other variables (one per day)
        self.Lon = np.repeat(Lon[None, :, :], self.n_days, axis=0)
        self.Lat = np.repeat(Lat[None, :, :], self.n_days, axis=0)

        # add to the var dict
        self.var_dict['Lon'] = self.Lon
        self.var_dict['Lat'] = self.Lat

        if self.compute_normalize_stats:
            self.normalize_stats['Lon'] = {"mean": np.mean(self.Lon), "std": np.std(self.Lon)}
            self.normalize_stats['Lat'] = {"mean": np.mean(self.Lat), "std": np.std(self.Lat)}

    def generate_DOY_fields(self):

        doys = np.zeros((self.n_days, ), dtype=np.float32)
        days_counted = 0
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        for i, year_month in enumerate(self.year_months):
            year = year_month[0]
            month = year_month[1]

            if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                num_days = 29
            else:
                num_days = days_in_month[month - 1]

            if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
                days_in_year = 366
            else:
                days_in_year = 365

            start_day = np.sum(days_in_month[:month - 1]) + 1
            end_day = start_day + num_days - 1
            doys[days_counted:days_counted + num_days] = np.arange(start_day, end_day + 1)/days_in_year
            days_counted += num_days

        doy_sin = np.sin(2 * np.pi * doys) # already divided by days in year to get 0-1 range above
        doy_cos = np.cos(2 * np.pi * doys)

        self.DOY_sin = np.repeat(doy_sin[:, None, None], self.var_dict['SST'].shape[1], axis=1)
        self.DOY_sin = np.repeat(self.DOY_sin, self.var_dict['SST'].shape[2], axis=2)
        self.DOY_cos = np.repeat(doy_cos[:, None, None], self.var_dict['SST'].shape[1], axis=1)
        self.DOY_cos = np.repeat(self.DOY_cos, self.var_dict['SST'].shape[2], axis=2)

        # add to the var grid
        self.var_dict['DOY_sin'] = self.DOY_sin
        self.var_dict['DOY_cos'] = self.DOY_cos

        if self.compute_normalize_stats:
            self.normalize_stats['DOY_sin'] = {"mean": np.mean(self.DOY_sin), "std": np.std(self.DOY_sin)}
            self.normalize_stats['DOY_cos'] = {"mean": np.mean(self.DOY_cos), "std": np.std(self.DOY_cos)}

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

    def __getitem__(self, idx):

        # if idx is too close to the start or end, adjust it to ensure we have a full temporal window
        # this will cause a few repeats but maybe its ok given the big data set
        if idx<self.time_window:
            idx = self.time_window
        if idx>=len(self)-self.time_window:
            idx = len(self)-self.time_window-1

        seaice_mask_out = (self.var_dict['Sea_Ice'][idx, :, :]<0.15).astype(int) * self.ocean_mask

        ########################################################################################
        # make output array with chl on the target time step (log chl) and mask of observed vs hidden chl
        chl_log = np.log(self.var_dict['Chl'][idx, :, :] + self.epsilon)
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
        for offset in range(-self.time_window, self.time_window + 1):
            global_i = idx + offset

            seaice = self.var_dict['Sea_Ice'][global_i, :, :]
            seaice_mask = (seaice<0.15).astype(int)
            seaice_mask = seaice_mask * self.ocean_mask

            # predictor channels
            for var_name in self.var_names:
                input_arr = self.var_dict[var_name]
                arr = input_arr[global_i, :, :]

                arr = np.where(np.isfinite(arr), arr, np.nan)
                if var_name not in ['DOY_cos','DOY_sin']:
                    arr = self.normalize(arr, var_name)
                arr = np.nan_to_num(
                    arr,
                    nan=self.fill_value,
                    posinf=self.fill_value,
                    neginf=self.fill_value,
                ).astype(np.float32)

                arr = arr * self.ocean_mask * seaice_mask

                if var_name == 'Chl' and offset == 0:
                    arr = np.where(input_mask, arr, self.fill_value).astype(np.float32)
                channels.append(arr)

        # ocean mask channel
        channels.append(self.ocean_mask.astype(np.float32))

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