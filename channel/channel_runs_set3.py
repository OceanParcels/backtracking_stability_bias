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


# Sensitivity analysis for analytical 2D experiments, with small perturbations
pset_anal_2D_dynamic = Lm.pset.from_netcdf(output_dir + "anal_2D_dynamic.nc")

np.random.seed(42)
pset_anal_2D_dynamic_backw_perturbed_100m = Lm.pset.from_forward(pset_anal_2D_dynamic)
pset_anal_2D_dynamic_backw_perturbed_100m.x_init = pset_anal_2D_dynamic_backw_perturbed_100m.x_init + np.random.normal(0, 100, pset_anal_2D_dynamic_backw_perturbed_100m.size)
pset_anal_2D_dynamic_backw_perturbed_100m.y_init = pset_anal_2D_dynamic_backw_perturbed_100m.y_init + np.random.normal(0, 100, pset_anal_2D_dynamic_backw_perturbed_100m.size)
pset_anal_2D_dynamic_backw_perturbed_100m.y_init = np.clip(pset_anal_2D_dynamic_backw_perturbed_100m.y_init, 5_0000.001, 1_999_999.999)

Lm.integration_loop_analytical_2D_dynamic(
                             pset=pset_anal_2D_dynamic_backw_perturbed_100m,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -5 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_anal_2D_dynamic_backw_perturbed_100m.to_netcdf(output_dir + "anal_2D_dynamic_backw_perturbed_100m")

np.random.seed(42)
pset_anal_2D_dynamic_backw_perturbed_1m = Lm.pset.from_forward(pset_anal_2D_dynamic)
pset_anal_2D_dynamic_backw_perturbed_1m.x_init = pset_anal_2D_dynamic_backw_perturbed_1m.x_init + np.random.normal(0, 1, pset_anal_2D_dynamic_backw_perturbed_1m.size)
pset_anal_2D_dynamic_backw_perturbed_1m.y_init = pset_anal_2D_dynamic_backw_perturbed_1m.y_init + np.random.normal(0, 1, pset_anal_2D_dynamic_backw_perturbed_1m.size)
pset_anal_2D_dynamic_backw_perturbed_1m.y_init = np.clip(pset_anal_2D_dynamic_backw_perturbed_1m.y_init, 5_0000.001, 1_999_999.999)

Lm.integration_loop_analytical_2D_dynamic(
                             pset=pset_anal_2D_dynamic_backw_perturbed_1m,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -5 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_anal_2D_dynamic_backw_perturbed_1m.to_netcdf(output_dir + "anal_2D_dynamic_backw_perturbed_1m")

np.random.seed(42)
pset_anal_2D_dynamic_backw_perturbed_1cm = Lm.pset.from_forward(pset_anal_2D_dynamic)
pset_anal_2D_dynamic_backw_perturbed_1cm.x_init = pset_anal_2D_dynamic_backw_perturbed_1cm.x_init + np.random.normal(0, 0.01, pset_anal_2D_dynamic_backw_perturbed_1cm.size)
pset_anal_2D_dynamic_backw_perturbed_1cm.y_init = pset_anal_2D_dynamic_backw_perturbed_1cm.y_init + np.random.normal(0, 0.01, pset_anal_2D_dynamic_backw_perturbed_1cm.size)
pset_anal_2D_dynamic_backw_perturbed_1cm.y_init = np.clip(pset_anal_2D_dynamic_backw_perturbed_1cm.y_init, 5_0000.001, 1_999_999.999)
Lm.integration_loop_analytical_2D_dynamic(
                             pset=pset_anal_2D_dynamic_backw_perturbed_1cm,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -5 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_anal_2D_dynamic_backw_perturbed_1cm.to_netcdf(output_dir + "anal_2D_dynamic_backw_perturbed_1cm")