import numpy as np
import xarray as xr
import tqdm

import Lagrangian_model as Lm

output_dir = "/nethome/4302001/output_data/backtracking/channel/"

# Global setting
integration_days = 180

# Loading the data
input_dir = "/nethome/4302001/local_data/channel/"
ds = xr.open_dataset(input_dir + "ACC_ridge_fine_2y_loca.nc").drop(["THETA", "SALT", "TRELAX", "MXLDEPTH", "ETAN"]).load()
ds_surface = ds.isel(Z=0)
ds_surface = ds_surface.drop(["WVEL"] + [coord for coord in ds_surface.coords if coord not in ["XC", "XG", "YC", "YG", "time", "iter"]]).load()
ds_surface_shot = ds_surface.isel(time=0)
ds_3d_shot = ds.isel(time=0)

# Defining the grid
dx = dy = 5_000
refinement = 2
x_grid = np.arange(0 + dx/refinement/2, 1_000_000, dx/refinement)
y_grid = np.arange(50000 + dx/refinement/2, 2_000_000, dx/refinement)
XX_grid, YY_grid = np.meshgrid(x_grid, y_grid)
X_parts = XX_grid.flatten()
Y_parts = YY_grid.flatten()
Z_parts_10 = np.ones_like(X_parts) * -10
Z_parts_300 = np.ones_like(X_parts) * -300


# Sensitivity analysis for RK4 2D experiments, with small different timesteps
pset_rk4_2D_dyn_dt1h = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                                      pset=pset_rk4_2D_dyn_dt1h,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 60 * 60,
                                      output_dt=24*60*60,
                                      )
pset_rk4_2D_dyn_dt1h.to_netcdf(output_dir + "rk4_2D_dyn_dt1h")

pset_rk4_2D_dyn_dt1h_backw = Lm.pset.from_forward(pset_rk4_2D_dyn_dt1h)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                                      pset=pset_rk4_2D_dyn_dt1h_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -60 * 60,
                                      )
pset_rk4_2D_dyn_dt1h_backw.to_netcdf(output_dir + "rk4_2D_dyn_dt1h_backw") 

pset_rk4_2D_dyn_dt6h = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                                      pset=pset_rk4_2D_dyn_dt6h,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 6 * 60 * 60,
                                      output_dt=24*60*60,
                                      )
pset_rk4_2D_dyn_dt6h.to_netcdf(output_dir + "rk4_2D_dyn_dt6h")

pset_rk4_2D_dyn_dt6h_backw = Lm.pset.from_forward(pset_rk4_2D_dyn_dt6h)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                                      pset=pset_rk4_2D_dyn_dt6h_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -6 * 60 * 60,
                                      )
pset_rk4_2D_dyn_dt6h_backw.to_netcdf(output_dir + "rk4_2D_dyn_dt6h_backw")

pset_rk4_2D_dyn_dt1m = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                                      pset=pset_rk4_2D_dyn_dt1m,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 1 * 60,
                                      output_dt=24*60*60,
                                      )
pset_rk4_2D_dyn_dt1m.to_netcdf(output_dir + "rk4_2D_dyn_dt1m")

pset_rk4_2D_dyn_dt1m_backw = Lm.pset.from_forward(pset_rk4_2D_dyn_dt1m)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                                      pset=pset_rk4_2D_dyn_dt1m_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -1* 60,
                                      )
pset_rk4_2D_dyn_dt1m_backw.to_netcdf(output_dir + "rk4_2D_dyn_dt1m_backw") 


del pset_rk4_2D_dyn_dt1m
del pset_rk4_2D_dyn_dt1m_backw
del pset_rk4_2D_dyn_dt1h
del pset_rk4_2D_dyn_dt1h_backw
del pset_rk4_2D_dyn_dt6h
del pset_rk4_2D_dyn_dt6h_backw