import datetime
import numpy as np
import sys
import os
import pickle
import pathlib

import parcels
import argparse
from glob import glob
import xarray as xr

import sys
sys.path.append("/nethome/4302001/backtracking_play/moi/")

from moi_gradient_tools import create_fieldset, CheckOutOfBounds, gradient_sampling_particle_simple, gradient_sampling_particle_simple_3D, SampleGradients_simple, SampleGradients_simple_3D

trajectory_out = "/nethome/4302001/output_data/backtracking/moi/divergence/"


def is_valid_file(parser, arg):
    if not pathlib.Path(arg).exists():
        parser.error(f"The file {arg} does not exist.")
    return arg


def lonlat_from_pickle(path):
    with open(path, "rb") as f:
        dict_lonlat = pickle.load(f)

    lons = dict_lonlat['lon']
    lats = dict_lonlat['lat']
    return lons, lats


def to_datetime(datetime64):
    timestamp = ((datetime64 - np.datetime64('1970-01-01T00:00:00'))
                 / np.timedelta64(1, 's'))
    return datetime.datetime.fromtimestamp(timestamp)#, datetime.UTC)


def add_time_coord(ds):
    traj_nonan_till_end = int(np.isfinite(ds.isel(obs=-1).time).where(np.isfinite(ds.isel(obs=-1).time)).dropna('trajectory').trajectory[0])
    ds = ds.assign_coords(time=ds.time.sel(trajectory=traj_nonan_till_end).drop('trajectory'))
    return ds


if __name__ == '__main__':
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-init_time', type=str, help="YYYYMMDD", required=False)
    argparser.add_argument("-input", dest="particleFile", required=True,
                    help="input file to start particles with", metavar="FILE",
                    type=lambda x: is_valid_file(argparser, x))
    argparser.add_argument('--T_integration', type=int, default=180, nargs='+', help="in days")
    argparser.add_argument('--dt', type=int, default=600, help="in seconds")
    argparser.add_argument('--advection_mode', type=str, default='3D', help="Advect `2D` or `3D` particles.")
    argparser.add_argument('--output_dt', type=int, default=1, help="in days")
    argparser.add_argument('--start_depth', type=float, default=4.9403e-1, help="initialization depth in meters")
    args = argparser.parse_args()

    
    # Loading generating lonlat file
    initFilePath = pathlib.Path(args.particleFile)

    # Define the type of run (init versus conjugate)
    if initFilePath.suffix in ['.pkl', '.pickle']:
        run_type = 'original'
    elif initFilePath.suffix in ['.zarr', '.nc']:
        run_type = 'conjugate'
    else:
        raise ValueError("Generating file must be either a pickle file or a zarr/netcdf file.")

    # Checks
    if initFilePath.suffix in ['.pkl', '.pickle']:
        # If the generating file is a pickle file, we need to have an init_time. Otherwise, it is inferred from the particle output file.
        assert len(args.init_time) == 8, "init_time must be YYYYMMDD"

        start_year = int(args.init_time[:4])
        start_month = int(args.init_time[4:6])
        start_day = int(args.init_time[6:])
    
    if type(args.T_integration) == int:
        integration_times = [args.T_integration]
    else:
        integration_times = args.T_integration

    fieldset_days = max(integration_times)
    print("Multiple integration times detected. Using the longest for fieldset.")
    
    if initFilePath.suffix in ['.pkl', '.pickle']:
        # Handle pickle file
        init_time = datetime.datetime(start_year, start_month, start_day)
        lons, lats = lonlat_from_pickle(args.particleFile)
        print(f"Loaded lonlat from pickle {args.particleFile}")
        print(f"Getting depths from args.start_depth: {args.start_depth}")
        depths = np.ones_like(lons) * args.start_depth
        times = [init_time]*len(lons)
    elif initFilePath.suffix in ['.zarr', '.nc']:
        # Handle zarr or netcdf file
        if initFilePath.suffix == '.zarr':
            engine='zarr'
        else:
            engine='netcdf4'
        ds_generator = xr.open_dataset(initFilePath, engine=engine).isel(obs=slice(0, fieldset_days + 1)) # assume obs in days
        ds_generator = add_time_coord(ds_generator)
        ds_generator_maxtime = ds_generator.time.isel(obs=-1).values
        fieldsetOriginTime = to_datetime(ds_generator_maxtime)
        total_seed_file_time = (ds_generator.time.isel(obs=-1) - ds_generator.time.isel(obs=0)).values.astype('timedelta64[D]').astype(int)
        if fieldset_days < total_seed_file_time:
            fieldset_days = total_seed_file_time
        print(f"Loaded dataset {args.particleFile}")


    print(f"Initializing fieldset. Mode: {args.advection_mode}")
    if args.init_time is not None:
        fieldsetOriginTime = datetime.datetime(start_year, start_month, start_day)

    fieldset = create_fieldset(T0=fieldsetOriginTime,
                               T=fieldset_days,
                               dt=args.dt,
                               mode=args.advection_mode)
    print("Fieldset creation successful.")

    # Choose advection kernel  and particle class
    if args.advection_mode == '3D' :
        advectionKernel = parcels.AdvectionRK4_3D
        gradientKernel = SampleGradients_simple_3D
        pclass = gradient_sampling_particle_simple_3D
    else:
        advectionKernel = parcels.AdvectionRK4
        gradientKernel = SampleGradients_simple
        pclass = gradient_sampling_particle_simple

    for integration_time in integration_times:
        print(f"Processing integration time: {integration_time} days")
        
        # If the generating file is zarr or netcdf, we need to load lats/lons/depths/times
        if initFilePath.suffix in ['.zarr', '.nc']:
            if integration_time + 1 == ds_generator.obs.size:
                obs_idx = -1
            else:
                obs_idx = integration_time + 1
            ds_generator_nonan = ds_generator.isel(trajectory=~np.isnan(ds_generator.isel(obs=obs_idx).lon.drop(['obs', 'time'])))

            obs_day = (ds_generator_nonan.time.isel(obs=1) - ds_generator_nonan.time.isel(obs=0)).values.astype('timedelta64[D]').astype(float)

            ds_generator_step_nonan = ds_generator_nonan.isel(obs=abs(int(integration_time * obs_day)))
            time = ds_generator_step_nonan.time.values

            lons = []
            lats = []
            depths = []

            lons.append(ds_generator_step_nonan.lon.values)
            lats.append(ds_generator_step_nonan.lat.values)
            depths.append(ds_generator_step_nonan.z.values)

            lons = np.vstack(lons).T.flatten()
            lats = np.vstack(lats).T.flatten()
            depths = np.vstack(depths).T.flatten()
            times = [time]*len(lons)

        # Parse the contents of the generating file
        genFileStem = ""
        initFileName = initFilePath.name
        if "regular" in initFileName:
            genFileStem += "regular_"
        elif "h3" in initFileName:
            genFileStem += "h3_"
        else:
            raise ValueError("Unknown generating file type. Must be either `regular` or `h3`.")
        genFileStem += 'res' + initFileName.split("res")[1].split('_')[0] + "_"
        genFileStem += initFilePath.name.split("lonlat")[0].split('_')[-2]

        # Creating a particle set
        pset = parcels.ParticleSet.from_list(fieldset=fieldset,
                                            pclass=pclass,
                                            lon=lons,
                                            lat=lats,
                                            depth=depths,
                                            time=times)

        # Defining the output file
        if args.dt > 0:
            direction = 'forward'
            initterminus = 'init'
        else:
            direction = 'backward'
            initterminus = 'terminus'
        if args.init_time is not None:
            inittimestr = init_time.strftime('%Y-%m-%d')
        else:
            if '_init_' in initFileName:
                inittimestr = initFileName.split("_init_")[1].split("_T")[0]
            elif '_terminus_' in initFileName:
                inittimestr = initFileName.split("_terminus_")[1].split("_T")[0]
        outname = f"MOi_divergence_{args.advection_mode}_{genFileStem}_{run_type}_{direction}_{initterminus}_{inittimestr}_T{integration_time}_dt{args.dt}.zarr"

        print(f"Output file: {outname}")

        output_file = pset.ParticleFile(name=f"{trajectory_out}/{outname}",
                                        outputdt=datetime.timedelta(days=args.output_dt))

        # Write metadata
        output_file.add_metadata("2D or 3D", args.advection_mode)
        if direction == 'forward':
            output_file.add_metadata("Initialization time", inittimestr)
        else:
            output_file.add_metadata("Initialization time (original run)", inittimestr)
        output_file.add_metadata("Timestep", args.dt)
        output_file.add_metadata("Integration time", integration_time)
        output_file.add_metadata("Output timestep", args.output_dt)
        output_file.add_metadata("Particles initialized from", args.particleFile)
        if direction == 'forward':
            output_file.add_metadata("Start depth", args.start_depth)

        # Execute simulation
        pset.execute([advectionKernel, gradientKernel, CheckOutOfBounds],
                        runtime=datetime.timedelta(days=integration_time, seconds=abs(args.dt)+1),
                        dt=args.dt,
                        output_file=output_file,
                        )

        print(f"Finished. Output file: {outname}")
