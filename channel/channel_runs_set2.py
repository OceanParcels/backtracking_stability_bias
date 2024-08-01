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


# Runs
# RK4 with backward perturbation
pset_RK4_2D_dyn_dt10 = Lm.pset.from_netcdf(output_dir + "RK4_2D_dyn_dt10_180d.nc")

np.random.seed(42)
pset_RK4_2D_dyn_dt10_backw_perturbed_100m = Lm.pset.from_forward(pset_RK4_2D_dyn_dt10)
pset_RK4_2D_dyn_dt10_backw_perturbed_100m.x_init = pset_RK4_2D_dyn_dt10_backw_perturbed_100m.x_init + np.random.normal(0, 100, pset_RK4_2D_dyn_dt10_backw_perturbed_100m.size)
pset_RK4_2D_dyn_dt10_backw_perturbed_100m.y_init = pset_RK4_2D_dyn_dt10_backw_perturbed_100m.y_init + np.random.normal(0, 100, pset_RK4_2D_dyn_dt10_backw_perturbed_100m.size)
pset_RK4_2D_dyn_dt10_backw_perturbed_100m.y_init = np.clip(pset_RK4_2D_dyn_dt10_backw_perturbed_100m.y_init, 5_0000.001, 1_999_999.999)

Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10_backw_perturbed_100m,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -10 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10_backw_perturbed_100m.to_netcdf(output_dir + "RK4_2D_dyn_dt10_backw_perturbed_100m")

np.random.seed(42)
pset_RK4_2D_dyn_dt10_backw_perturbed_1m = Lm.pset.from_forward(pset_RK4_2D_dyn_dt10)
pset_RK4_2D_dyn_dt10_backw_perturbed_1m.x_init = pset_RK4_2D_dyn_dt10_backw_perturbed_1m.x_init + np.random.normal(0, 1, pset_RK4_2D_dyn_dt10_backw_perturbed_1m.size)
pset_RK4_2D_dyn_dt10_backw_perturbed_1m.y_init = pset_RK4_2D_dyn_dt10_backw_perturbed_1m.y_init + np.random.normal(0, 1, pset_RK4_2D_dyn_dt10_backw_perturbed_1m.size)
pset_RK4_2D_dyn_dt10_backw_perturbed_1m.y_init = np.clip(pset_RK4_2D_dyn_dt10_backw_perturbed_1m.y_init, 5_0000.001, 1_999_999.999)

Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10_backw_perturbed_1m,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -10 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10_backw_perturbed_1m.to_netcdf(output_dir + "RK4_2D_dyn_dt10_backw_perturbed_1m")

np.random.seed(42)
pset_RK4_2D_dyn_dt10_backw_perturbed_1cm = Lm.pset.from_forward(pset_RK4_2D_dyn_dt10)
pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.x_init = pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.x_init + np.random.normal(0, 0.01, pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.size)
pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.y_init = pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.y_init + np.random.normal(0, 0.01, pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.size)
pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.y_init = np.clip(pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.y_init, 5_0000.001, 1_999_999.999)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10_backw_perturbed_1cm,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -10 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10_backw_perturbed_1cm.to_netcdf(output_dir + "RK4_2D_dyn_dt10_backw_perturbed_1cm")


# Sensitivity of analytical timestep
pset_anal_2D_dynamic_dt1d = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt1d,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 24 * 60  * 60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_dynamic_dt1d.to_netcdf(output_dir + "anal_2D_dynamic_dt1d")

pset_anal_2D_dynamic_dt1d = Lm.pset.from_netcdf(output_dir + "anal_2D_dynamic_dt1d.nc")
pset_anal_2D_dynamic_dt1d_backw = Lm.pset.from_forward(pset_anal_2D_dynamic_dt1d)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt1d_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -24 * 60 * 60,
                                      )
pset_anal_2D_dynamic_dt1d_backw.to_netcdf(output_dir + "anal_2D_dynamic_dt1d_backw") 

pset_anal_2D_dynamic_dt1h = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt1h,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 60 * 60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_dynamic_dt1h.to_netcdf(output_dir + "anal_2D_dynamic_dt1h")

pset_anal_2D_dynamic_dt1h_backw = Lm.pset.from_forward(pset_anal_2D_dynamic_dt1h)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt1h_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -60 * 60,
                                      )
pset_anal_2D_dynamic_dt1h_backw.to_netcdf(output_dir + "anal_2D_dynamic_dt1h_backw") 

pset_anal_2D_dynamic_dt6h = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt6h,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 6 * 60 * 60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_dynamic_dt6h.to_netcdf(output_dir + "anal_2D_dynamic_dt6h")

pset_anal_2D_dynamic_dt6h_backw = Lm.pset.from_forward(pset_anal_2D_dynamic_dt6h)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt6h_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -6 * 60 * 60,
                                      )
pset_anal_2D_dynamic_dt6h_backw.to_netcdf(output_dir + "anal_2D_dynamic_dt6h_backw")

pset_anal_2D_dynamic_dt1m = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt1m,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 1 * 60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_dynamic_dt1m.to_netcdf(output_dir + "anal_2D_dynamic_dt1m")

pset_anal_2D_dynamic_dt1m = Lm.pset.from_netcdf(output_dir + "anal_2D_dynamic_dt1m.nc")
pset_anal_2D_dynamic_dt1m_backw = Lm.pset.from_forward(pset_anal_2D_dynamic_dt1m)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_dt1m_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -1* 60,
                                      )
pset_anal_2D_dynamic_dt1m_backw.to_netcdf(output_dir + "anal_2D_dynamic_dt1m_backw") 


del pset_anal_2D_dynamic_dt1m
del pset_anal_2D_dynamic_dt1m_backw
del pset_anal_2D_dynamic_dt1h
del pset_anal_2D_dynamic_dt1h_backw
del pset_anal_2D_dynamic_dt6h
del pset_anal_2D_dynamic_dt6h_backw
del pset_anal_2D_dynamic_dt1d
del pset_anal_2D_dynamic_dt1d_backw
