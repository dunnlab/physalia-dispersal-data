import numpy as np
import netCDF4 as nc

def create_netcdf(file_path):
    # Create a new NetCDF file
    dataset = nc.Dataset(file_path, 'w', format='NETCDF4')

    # Create dimensions
    time_dim = dataset.createDimension('time', 366)
    lat_dim = dataset.createDimension('lat', 720)
    lon_dim = dataset.createDimension('lon', 1440)

    # Create variables with fill values
    time_var = dataset.createVariable('time', np.int32, ('time',))
    lat_var = dataset.createVariable('lat', np.float32, ('lat',), fill_value=np.nan)
    lon_var = dataset.createVariable('lon', np.float32, ('lon',), fill_value=np.nan)
    u_var = dataset.createVariable('U', np.float32, ('time', 'lat', 'lon'), fill_value=np.nan)
    v_var = dataset.createVariable('V', np.float32, ('time', 'lat', 'lon'), fill_value=np.nan)

    # Set variable attributes
    time_var.units = 'days since 2022-11-01 00:00:00'
    time_var.calendar = 'proleptic_gregorian'
    
    lat_var.units = 'degrees_north'
    
    lon_var.units = 'degrees_east'
    
    u_var.units = 'm/s'
    u_var.long_name = 'U'
    
    v_var.units = 'm/s'
    v_var.long_name = 'V'

    # Set variable data
    time_var[:] = np.arange(366)
    lat_var[:] = np.arange(720)
    lon_var[:] = np.arange(1440)
    u_var[:, :, :] = np.zeros((366, 720, 1440))
    v_var[:, :, :] = np.zeros((366, 720, 1440))

    # Close the dataset
    dataset.close()

if __name__ == "__main__":
    file_path = "base_data.nc"
    create_netcdf(file_path)
    print(f"NetCDF file '{file_path}' created successfully.")
