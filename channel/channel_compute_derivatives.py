import numpy as np
import xarray as xr

import xgcm

ds = xr.open_dataset("/nethome/4302001/local_data/channel/ACC_ridge_fine_2y_loca.nc").isel(time=slice(0,361))
ds = ds.drop_vars(['TRELAX', 'SALT', 'THETA', 'MXLDEPTH', 'ETAN'])

grid = xgcm.Grid(ds, 
                 coords={"X" : {"center" : "XC", "left": "XG"},
                         "Y" : {"center" : "YC", "left": "YG"},
                         "Z" : {"center" : "Z", "left": "Zl", "outer" : "Zp1"}},
                 periodic = ['Y'])

# dUdX = grid.diff(ds.UVEL, 'X')/5000
# dUdY = grid.diff(ds.UVEL, 'Y')/5000
# dVdX = grid.diff(ds.VVEL, 'X')/5000
# dVdY = grid.diff(ds.VVEL, 'Y')/5000

# speed = np.sqrt(grid.interp(ds.UVEL, axis='X')**2 + grid.interp(ds.VVEL, axis='Y')**2)

# ds_out = xr.Dataset({'dUdX' : dUdX,
#                      'dUdY' : dUdY,
#                      'dVdX' : dVdX,
#                      'dVdY' : dVdY,
#                      'dWdZ' : dWdZ,
#                      'speed' : speed})

# encoding = {var: {"zlib": True, "complevel": 2} for var in ['dUdX', 'dUdY', 'dVdX', 'dVdY', 'dWdZ', 'speed']}

# ds_out.to_netcdf("/nethome/4302001/local_data/channel/ACC_ridge_fine_2y_derivatives_1y.nc", encoding=encoding)

# ds_derivatives_tave_1y = ds_out.mean('time')
# ds_derivatives_tave_1y.to_netcdf("/nethome/4302001/local_data/channel/ACC_ridge_fine_2y_derivatives_tave_1y.nc", encoding=encoding)

UVEL_mean = grid.interp(ds.UVEL.mean('time'), axis='X')
VVEL_mean = grid.interp(ds.VVEL.mean('time'), axis='Y')
WVEL_mean = grid.interp(ds.WVEL.mean('time'), axis='Z')
MKE = 0.5 * (UVEL_mean**2 + VVEL_mean**2)
EKE = 0.5 * ((grid.interp(ds.UVEL, axis='X') - UVEL_mean)**2 + (grid.interp(ds.VVEL, axis='Y') - VVEL_mean)**2).mean('time')

ds_energy = xr.Dataset({'UVEL_mean' : UVEL_mean,
                        'VVEL_mean' : VVEL_mean,
                        'WVEL_mean' : WVEL_mean,
                        'MKE' : MKE,
                        'EKE' : EKE})
encoding = {var: {"zlib": True, "complevel": 2} for var in ['UVEL_mean', 'VVEL_mean', 'WVEL_mean', 'MKE', 'EKE']}

ds_energy.to_netcdf("/nethome/4302001/local_data/channel/ACC_ridge_fine_2y_energy_1y.nc", encoding=encoding)
