import numpy as np
import xarray as xr
import xgcm
import xnemogcm as xn
import pandas as pd
import argparse
import tqdm

moi_dir = "/nethome/4302001/data/input_data/MOi/"
psy_dir = moi_dir + "psy4v3r1/"
domain_dir = moi_dir + "domain_ORCA0083-N006/"
output_dir = "/nethome/4302001/local_data/psy4v3r1_derivatives/"

aux_files = [domain_dir + "PSY4V3R1_mesh_hgr.nc",
             domain_dir + "PSY4V3R1_mesh_zgr.nc"]
domcfg = xn.open_domain_cfg(files=aux_files)


drop_viscosities = ["sotkeavmu1", "sotkeavmu15", "sotkeavmu30", "sotkeavmu50"]


def process_first_timestep(timestamp):
    template_U = xr.open_dataset(psy_dir + f"psy4v3r1-daily_U_{str(timestamp)[:10]}.nc").isel(deptht=0).drop(drop_viscosities)
    template_V = xr.open_dataset(psy_dir + f"psy4v3r1-daily_V_{str(timestamp)[:10]}.nc").isel(deptht=0)

    template = xn.process_nemo(domcfg=domcfg, positions=[(template_U, "U"),
                                                         (template_V, "V")])

    template_full = xn._merge_nemo_and_domain_cfg(
        nemo_ds=template, domcfg=domcfg)

    grid = xgcm.Grid(template_full, metrics=xn.get_metrics(
        template_full), periodic=False)

    time_counter = template_U.time_counter

    template_U.close()
    template_V.close()

    return template, grid, time_counter


def process_timestep(timestamp, grid):
    ds_U = xr.open_dataset(psy_dir + f"psy4v3r1-daily_U_{str(timestamp)[:10]}.nc").isel(deptht=0).drop(drop_viscosities)
    ds_V = xr.open_dataset(psy_dir + f"psy4v3r1-daily_V_{str(timestamp)[:10]}.nc").isel(deptht=0)

    ds = xn.process_nemo(domcfg=domcfg, positions=[(ds_U, "U"),
                                                   (ds_V, "V")])

    time_counter = ds_U.time_counter

    ds_U.close()
    ds_V.close()

    return ds, time_counter


def compute_derivatives(grid, ds):
    derivatives = {
        "dudx": grid.derivative(ds.vozocrtx, 'X', boundary='extend'),
        "dudy": grid.derivative(ds.vozocrtx, 'Y', boundary='extend'),
        "dvdx": grid.derivative(ds.vomecrty, 'X', boundary='extend'),
        "dvdy": grid.derivative(ds.vomecrty, 'Y', boundary='extend'),
    }

    return derivatives


def create_dataset(derivatives, time_counter):
    ds = xr.Dataset(derivatives)
    ds["time_counter"] = time_counter
    ds = ds.assign_coords({"time_counter": time_counter})
    ds.attrs["History"] = "Created by backtracking_play/moi/moi_divergence.py"
    ds.attrs["Creation date"] = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S")

    return ds


def export_dataset(ds, timestamp):
    ds.to_netcdf(f"{output_dir}psy4v3r1_velocity_derivatives_{str(timestamp)[:10]}.nc",
                 encoding={var: {"zlib": True, "complevel": 1} for var in ['dudx', 'dudy', 'dvdx', 'dvdy']})


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("date", help="First date to process (YYYYMMDD)")
    argparser.add_argument("ndays", type=int, help="Number of days to process")

    args = argparser.parse_args()
    
    timestamp_init = pd.to_datetime(f"{args.date[0:4]}-{args.date[4:6]}-{args.date[6:]}")

    print(f"Will process {args.ndays} days starting from {timestamp_init}")

    template, grid, init_time_counter = process_first_timestep(timestamp_init)

    print("Loaded grid and first timestep.")

    for i in tqdm.tqdm(range(args.ndays)):
        timestamp = timestamp_init + pd.Timedelta(i, 'D')

        if i == 0:
            derivatives = compute_derivatives(grid, template)
            ds_out = create_dataset(derivatives, init_time_counter)

        else:
            ds, time_counter = process_timestep(timestamp, grid)
            derivatives = compute_derivatives(grid, ds)
            ds_out = create_dataset(derivatives, time_counter)

        
        export_dataset(ds_out, timestamp)
        ds_out.close()
