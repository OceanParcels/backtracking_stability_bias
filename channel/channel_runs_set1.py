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


# #  Runs
# RK4 Static 2D
pset_RK4_2D_static_dt10 = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_static_numpy,
                             pset=pset_RK4_2D_static_dt10,
                             U_field=ds_surface_shot.UVEL.values,
                             V_field=ds_surface_shot.VVEL.values,
                             dt=10 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_static_dt10.to_netcdf(output_dir + "RK4_2D_static_dt10_180d")

pset_RK4_2D_static_dt10_backw = Lm.pset.from_forward(pset_RK4_2D_static_dt10)
Lm.integration_loop_2D_numpy(Lm.rk4_static_numpy,
                             pset=pset_RK4_2D_static_dt10_backw,
                             U_field=ds_surface_shot.UVEL.values,
                             V_field=ds_surface_shot.VVEL.values,
                             dt= -10 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_static_dt10_backw.to_netcdf(output_dir + "RK4_2D_static_dt10_180d_backw")

pset_RK4_2D_static_dt10_backw_from_t0 = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_static_numpy,
                             pset=pset_RK4_2D_static_dt10_backw_from_t0,
                             U_field=ds_surface_shot.UVEL.values,
                             V_field=ds_surface_shot.VVEL.values,
                             dt= -10 * 60,
                             T_int=180 * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_static_dt10_backw_from_t0.to_netcdf(output_dir + "RK4_2D_static_dt10_180d_backw_from_t0")

pset_RK4_2D_static_dt10_forward_from_backw_from_t0 = Lm.pset.from_forward(pset_RK4_2D_static_dt10_backw_from_t0)
Lm.integration_loop_2D_numpy(Lm.rk4_static_numpy,
                             pset=pset_RK4_2D_static_dt10_forward_from_backw_from_t0,
                             U_field=ds_surface_shot.UVEL.values,
                             V_field=ds_surface_shot.VVEL.values,
                             dt= 10 * 60,
                             T_int=180 * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_static_dt10_forward_from_backw_from_t0.to_netcdf(output_dir + "RK4_2D_static_dt10_forward_from_backw_from_t0")

del pset_RK4_2D_static_dt10
del pset_RK4_2D_static_dt10_backw
del pset_RK4_2D_static_dt10_backw_from_t0
del pset_RK4_2D_static_dt10_forward_from_backw_from_t0


# RK4 Dynamic 2D
pset_RK4_2D_dyn_dt10 = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt=10 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10.to_netcdf(output_dir + "RK4_2D_dyn_dt10_180d")

pset_RK4_2D_dyn_dt10_backw = Lm.pset.from_forward(pset_RK4_2D_dyn_dt10)
# pset_RK4_2D_dyn_dt10 = Lm.pset.from_netcdf(output_dir + "RK4_2D_dyn_dt10_180d.nc")
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10_backw,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -10 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10_backw.to_netcdf(output_dir + "RK4_2D_dyn_dt10_180d_backw")

pset_RK4_2D_dyn_dt10_backw_from_t0 = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10_backw_from_t0,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= -10 * 60,
                             T0 = integration_days * 24 * 60 * 60,
                             T_int=180 * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10_backw_from_t0.to_netcdf(output_dir + "RK4_2D_dyn_dt10_180d_backw_from_t0")

pset_RK4_2D_dyn_dt10_forward_from_backw_from_t0 = Lm.pset.from_forward(pset_RK4_2D_dyn_dt10_backw_from_t0)
Lm.integration_loop_2D_numpy(Lm.rk4_dynamic_numpy,
                             pset=pset_RK4_2D_dyn_dt10_forward_from_backw_from_t0,
                             U_field=ds_surface.UVEL.values,
                             V_field=ds_surface.VVEL.values,
                             dt= 10 * 60,
                             T0 = 0,
                             T_int= integration_days * 24 * 60 * 60 ,
                             output_dt=24*60*60
                             )
pset_RK4_2D_dyn_dt10_forward_from_backw_from_t0.to_netcdf(output_dir + "RK4_2D_dyn_dt10_forward_from_backw_from_t0")

del pset_RK4_2D_dyn_dt10
del pset_RK4_2D_dyn_dt10_backw
del pset_RK4_2D_dyn_dt10_backw_from_t0
del pset_RK4_2D_dyn_dt10_forward_from_backw_from_t0

# RK4 Static 3D
pset_RK4_3D_static_dt10_z10 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_10)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_static_numpy,
                             pset=pset_RK4_3D_static_dt10_z10,
                             U_field=ds_3d_shot.UVEL.values,
                             V_field=ds_3d_shot.VVEL.values,
                             W_field=ds_3d_shot.WVEL.values,
                             dt=10 * 60,
                             T0=0*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_static_dt10_z10.to_netcdf(output_dir + "RK4_3D_static_dt10_z10")                    

pset_RK4_3D_static_dt10_z10_backw = Lm.pset.from_forward(pset_RK4_3D_static_dt10_z10)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_static_numpy,
                             pset=pset_RK4_3D_static_dt10_z10_backw,
                             U_field=ds_3d_shot.UVEL.values,
                             V_field=ds_3d_shot.VVEL.values,
                             W_field=ds_3d_shot.WVEL.values,
                             dt= - 10 * 60,
                             T0=integration_days*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_static_dt10_z10_backw.to_netcdf(output_dir + "RK4_3D_static_dt10_z10_backw")                    
                            
pset_RK4_3D_static_dt10_z300 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_300)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_static_numpy,
                             pset=pset_RK4_3D_static_dt10_z300,
                             U_field=ds_3d_shot.UVEL.values,
                             V_field=ds_3d_shot.VVEL.values,
                             W_field=ds_3d_shot.WVEL.values,
                             dt=10 * 60,
                             T0=0*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_static_dt10_z300.to_netcdf(output_dir + "RK4_3D_static_dt10_z300")                    

pset_RK4_3D_static_dt10_z300_backw = Lm.pset.from_forward(pset_RK4_3D_static_dt10_z300)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_static_numpy,
                             pset=pset_RK4_3D_static_dt10_z300_backw,
                             U_field=ds_3d_shot.UVEL.values,
                             V_field=ds_3d_shot.VVEL.values,
                             W_field=ds_3d_shot.WVEL.values,
                             dt= - 10 * 60,
                             T0=integration_days*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_static_dt10_z300_backw.to_netcdf(output_dir + "RK4_3D_static_dt10_z300_backw")

del pset_RK4_3D_static_dt10_z10
del pset_RK4_3D_static_dt10_z10_backw
del pset_RK4_3D_static_dt10_z300
del pset_RK4_3D_static_dt10_z300_backw

# RK4 Dynamic 3D
pset_RK4_3D_dynamic_dt10_z10 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_10)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_dynamic_numpy,
                             pset=pset_RK4_3D_dynamic_dt10_z10,
                             U_field=ds.UVEL.values,
                             V_field=ds.VVEL.values,
                             W_field=ds.WVEL.values,
                             dt=10 * 60,
                             T0=0*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_dynamic_dt10_z10.to_netcdf(output_dir + "RK4_3D_dynamic_dt10_z10")                    

pset_RK4_3D_dynamic_dt10_z10_backw = Lm.pset.from_forward(pset_RK4_3D_dynamic_dt10_z10)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_dynamic_numpy,
                             pset=pset_RK4_3D_dynamic_dt10_z10_backw,
                             U_field=ds.UVEL.values,
                             V_field=ds.VVEL.values,
                             W_field=ds.WVEL.values,
                             dt= - 10 * 60,
                             T0=integration_days*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_dynamic_dt10_z10_backw.to_netcdf(output_dir + "RK4_3D_dynamic_dt10_z10_backw")                    
                            
pset_RK4_3D_dynamic_dt10_z300 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_300)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_dynamic_numpy,
                             pset=pset_RK4_3D_dynamic_dt10_z300,
                             U_field=ds.UVEL.values,
                             V_field=ds.VVEL.values,
                             W_field=ds.WVEL.values,
                             dt=10 * 60,
                             T0=0*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_dynamic_dt10_z300.to_netcdf(output_dir + "RK4_3D_dynamic_dt10_z300")                    

pset_RK4_3D_dynamic_dt10_z300_backw = Lm.pset.from_forward(pset_RK4_3D_dynamic_dt10_z300)
Lm.integration_loop_3D_numpy(Lm.rk4_3D_dynamic_numpy,
                             pset=pset_RK4_3D_dynamic_dt10_z300_backw,
                             U_field=ds.UVEL.values,
                             V_field=ds.VVEL.values,
                             W_field=ds.WVEL.values,
                             dt= - 10 * 60,
                             T0=integration_days*24*60*60,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_RK4_3D_dynamic_dt10_z300_backw.to_netcdf(output_dir + "RK4_3D_dynamic_dt10_z300_backw")

del pset_RK4_3D_dynamic_dt10_z10
del pset_RK4_3D_dynamic_dt10_z10_backw
del pset_RK4_3D_dynamic_dt10_z300
del pset_RK4_3D_dynamic_dt10_z300_backw

# Analytical 2D Static
pset_anal_2D_static = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_static(pset=pset_anal_2D_static,
                                      U_field = ds_surface_shot.UVEL.values,
                                      V_field = ds_surface_shot.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_static.to_netcdf(output_dir + "anal_2D_static")

pset_anal_2D_static_backw = Lm.pset.from_forward(pset_anal_2D_static)
Lm.integration_loop_analytical_2D_static(pset=pset_anal_2D_static_backw,
                                      U_field = -ds_surface_shot.UVEL.values,
                                      V_field = -ds_surface_shot.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_static_backw.to_netcdf(output_dir + "anal_2D_static_backw")  

pset_anal_2D_static_using_dyn = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_static_using_dyn,
                               U_field=ds_surface_shot.UVEL.values, 
                               V_field=ds_surface_shot.VVEL.values, 
                                T_int=integration_days *24*60*60, 
                                output_dt=24*60*60,
                                dt = 24 * 60 * 60
                                        )
pset_anal_2D_static_using_dyn.to_netcdf(output_dir + "anal_2D_static_using_dyn")    

del pset_anal_2D_static
del pset_anal_2D_static_backw
del pset_anal_2D_static_using_dyn

pset_anal_2D_static_backw_from_t0 = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_static(pset=pset_anal_2D_static_backw_from_t0,
                             U_field=-ds_surface_shot.UVEL.values,
                             V_field=-ds_surface_shot.VVEL.values,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_anal_2D_static_backw_from_t0.to_netcdf(output_dir + "anal_2D_static_backw_from_t0")

pset_anal_static_forward_from_backw_from_t0 = Lm.pset.from_forward(pset_anal_2D_static_backw_from_t0)
Lm.integration_loop_analytical_2D_static(pset=pset_anal_static_forward_from_backw_from_t0,
                             U_field=ds_surface_shot.UVEL.values,
                             V_field=ds_surface_shot.VVEL.values,
                             T_int=integration_days * 24*60*60,
                             output_dt=24*60*60
                             )
pset_anal_static_forward_from_backw_from_t0.to_netcdf(output_dir + "anal_static_forward_from_backw_from_t0")


# Analytical 2D Dynamic
pset_anal_2D_dynamic = Lm.pset(x_init=X_parts, y_init=Y_parts)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=0,
                                      dt = 5 * 60,
                                      output_dt=24*60*60,
                                      )
pset_anal_2D_dynamic.to_netcdf(output_dir + "anal_2D_dynamic")

pset_anal_2D_dynamic_backw = Lm.pset.from_forward(pset_anal_2D_dynamic)
Lm.integration_loop_analytical_2D_dynamic(pset=pset_anal_2D_dynamic_backw,
                                      U_field = ds_surface.UVEL.values,
                                      V_field = ds_surface.VVEL.values,
                                      T_int=integration_days * 24*60*60,
                                      T0=integration_days * 24*60*60,
                                      output_dt=24*60*60,
                                      dt = -5 * 60,
                                      )
pset_anal_2D_dynamic_backw.to_netcdf(output_dir + "anal_2D_dynamic_backw") 

del pset_anal_2D_dynamic
del pset_anal_2D_dynamic_backw


# Analytical 3D Stationary
pset_anal_3D_static_z10 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_10)
Lm.integration_loop_analytical_3D_static(pset=pset_anal_3D_static_z10,
                                         U_field=ds_3d_shot.UVEL.values,
                                         V_field=ds_3d_shot.VVEL.values,
                                         W_field=ds_3d_shot.WVEL.values,
                                         T_int=integration_days * 24*60*60,
                                         output_dt=24*60*60
                                         )
pset_anal_3D_static_z10.to_netcdf(output_dir + "anal_3D_static_z10")                                        

pset_anal_3D_static_z10_backw = Lm.pset.from_forward(pset_anal_3D_static_z10)
Lm.integration_loop_analytical_3D_static(pset=pset_anal_3D_static_z10_backw,
                                         U_field=-ds_3d_shot.UVEL.values,
                                         V_field=-ds_3d_shot.VVEL.values,
                                         W_field=-ds_3d_shot.WVEL.values,
                                         T_int=integration_days * 24*60*60,
                                         output_dt=24*60*60
                                         )
pset_anal_3D_static_z10_backw.to_netcdf(output_dir + "anal_3D_static_z10_backw")                                        

pset_anal_3D_static_z300 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_300)
Lm.integration_loop_analytical_3D_static(pset=pset_anal_3D_static_z300,
                                         U_field=ds_3d_shot.UVEL.values,
                                         V_field=ds_3d_shot.VVEL.values,
                                         W_field=ds_3d_shot.WVEL.values,
                                         T_int=integration_days * 24*60*60,
                                         output_dt=24*60*60
                                         )
pset_anal_3D_static_z300.to_netcdf(output_dir + "anal_3D_static_z300")                                        

pset_anal_3D_static_z300_backw = Lm.pset.from_forward(pset_anal_3D_static_z300)
Lm.integration_loop_analytical_3D_static(pset=pset_anal_3D_static_z300_backw,
                                         U_field=-ds_3d_shot.UVEL.values,
                                         V_field=-ds_3d_shot.VVEL.values,
                                         W_field=-ds_3d_shot.WVEL.values,
                                         T_int=integration_days * 24*60*60,
                                         output_dt=24*60*60
                                         )
pset_anal_3D_static_z300_backw.to_netcdf(output_dir + "anal_3D_static_z300_backw") 

del pset_anal_3D_static_z10
del pset_anal_3D_static_z10_backw
del pset_anal_3D_static_z300
del pset_anal_3D_static_z300_backw

# Analytical 3D Dynamic
pset_anal_3D_dynamic_z10 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_10)
Lm.integration_loop_analytical_3D_dynamic(pset=pset_anal_3D_dynamic_z10,
                                U_field = ds.UVEL.values, 
                                V_field = ds.VVEL.values, 
                                W_field = ds.WVEL.values,
                                dt = 5 * 60,
                                T0 = 0,
                                T_int = integration_days*24*60*60
                                )
pset_anal_3D_dynamic_z10.to_netcdf(output_dir + "anal_3D_dynamic_z10")   

pset_anal_3D_dynamic_z10_backw = Lm.pset.from_forward(pset_anal_3D_dynamic_z10) 
# pset_anal_3D_dynamic_z10_backw = Lm.pset.from_netcdf(output_dir + "anal_3D_dynamic_z10.nc")                   

Lm.integration_loop_analytical_3D_dynamic(pset=pset_anal_3D_dynamic_z10_backw,
                                U_field = ds.UVEL.values, 
                                V_field = ds.VVEL.values, 
                                W_field = ds.WVEL.values,
                                dt = -5 * 60,
                                T0 = integration_days*24*60*60,
                                T_int = integration_days*24*60*60
                                )
pset_anal_3D_dynamic_z10_backw.to_netcdf(output_dir + "anal_3D_dynamic_z10_backw")  

pset_anal_3D_dynamic_z300 = Lm.pset(x_init=X_parts, y_init=Y_parts, z_init=Z_parts_300)
Lm.integration_loop_analytical_3D_dynamic(pset=pset_anal_3D_dynamic_z300,
                                U_field = ds.UVEL.values, 
                                V_field = ds.VVEL.values, 
                                W_field = ds.WVEL.values,
                                dt = 5 * 60,
                                T0 = 0,
                                T_int = integration_days*24*60*60
                                )
pset_anal_3D_dynamic_z300.to_netcdf(output_dir + "anal_3D_dynamic_z300")                            

pset_anal_3D_dynamic_z300_backw = Lm.pset.from_forward(pset_anal_3D_dynamic_z300) 
Lm.integration_loop_analytical_3D_dynamic(pset=pset_anal_3D_dynamic_z300_backw,
                                U_field = ds.UVEL.values, 
                                V_field = ds.VVEL.values, 
                                W_field = ds.WVEL.values,
                                dt = -5 * 60,
                                T0 = integration_days*24*60*60,
                                T_int = integration_days*24*60*60
                                )
pset_anal_3D_dynamic_z300_backw.to_netcdf(output_dir + "anal_3D_dynamic_z300_backw") 


del pset_anal_3D_dynamic_z10
del pset_anal_3D_dynamic_z10_backw
del pset_anal_3D_dynamic_z300
del pset_anal_3D_dynamic_z300_backw