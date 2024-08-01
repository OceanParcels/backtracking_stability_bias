import parcels
import numpy as np
import datetime
import math
from parcels import StatusCode
import os

data_path = "/storage2/shared/oceanparcels/input_data/MOi/psy4v3r1/"
divergence_path = "/nethome/4302001/local_data/psy4v3r1_derivatives/"
mesh_horiz = "/storage2/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/PSY4V3R1_mesh_hgr.nc"
# mesh_vert = "/storage2/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/PSY4V3R1_mesh_zgr.nc"
mesh_vert = "/nethome/4302001/local_data/mesh/PSY4V3R1_mesh_zgr_t0.nc"
coastal_mask = "/storage/shared/oceanparcels/output_data/data_Michael/PlasticParcels/data/output_data/masks/mask_coast_NEMO0083.nc"


def create_fieldset(T0, T, dt, mode='3D'):
    times = [T0 + datetime.timedelta(days=int(np.sign(dt) * i)) for i in range(-1, T + 2)]
    ufiles = sorted([data_path + f'psy4v3r1-daily_U_{time.year}-{time.month:02d}-{time.day:02d}.nc' for time in times])
    vfiles = sorted([data_path + f'psy4v3r1-daily_V_{time.year}-{time.month:02d}-{time.day:02d}.nc' for time in times])
    divergence_files = sorted([divergence_path + f'psy4v3r1_velocity_derivatives_{time.year}-{time.month:02d}-{time.day:02d}.nc' for time in times])
    surffiles = sorted([data_path + f'psy4v3r1-daily_2D_{time.year}-{time.month:02d}-{time.day:02d}.nc' for time in times])

    assert len(ufiles) == len(vfiles) == len(divergence_files) == len(surffiles), "Different number of U, V and divergence files"
    assert len(ufiles) > T, "Not enough files to cover T"

    for ufile, vfile, divergence_file, surffile in zip(ufiles, vfiles, divergence_files, surffiles):
        assert os.path.isfile(ufile), f"Can't find file {ufile}"
        assert os.path.isfile(vfile), f"Can't find file {vfile}"
        assert os.path.isfile(divergence_file), f"Can't find file {divergence_file}"
        assert os.path.isfile(surffile), f"Can't find file {surffile}"

    filenames = {'U': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': ufiles},
                 'V': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': vfiles},
                 'U_linear_invdist': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': ufiles},
                 'V_linear_invdist': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': vfiles},
                 'dudx_linear': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': divergence_files},
                 'dudy_linear': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': divergence_files},
                 'dvdx_linear': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': divergence_files},
                 'dvdy_linear': {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': divergence_files},
                 'coastal_mask': {'lon': coastal_mask, 'lat': coastal_mask, 'data': coastal_mask},
                 'bathy' : {'lon': mesh_vert, 'lat': mesh_vert, 'data': mesh_vert},
                 'ssh' : {'lon': mesh_horiz, 'lat': mesh_horiz, 'data': surffiles}
                }

    variables = {'U': 'vozocrtx',
                 'V': 'vomecrty',
                 'U_linear_invdist': 'vozocrtx',
                 'V_linear_invdist': 'vomecrty',
                 'dudx_linear': 'dudx',
                 'dudy_linear': 'dudy',
                 'dvdx_linear': 'dvdx',
                 'dvdy_linear': 'dvdy',
                 'coastal_mask': 'mask_coast',
                 'bathy': 'mbathy',
                 'ssh': 'sossheig',
                 }

    dimensions = {'U': {'lon': 'glamf', 'lat': 'gphif', 'depth' : 'gdepw_0', 'time': 'time_counter'},
                  'V': {'lon': 'glamf', 'lat': 'gphif', 'depth' : 'gdepw_0', 'time': 'time_counter'},
                  'U_linear_invdist': {'lon': 'glamu', 'lat': 'gphiu', 'depth' : 'gdept_0', 'time': 'time_counter'},
                  'V_linear_invdist': {'lon': 'glamv', 'lat': 'gphiv', 'depth' : 'gdept_0','time': 'time_counter'},
                  'dudx_linear': {'lon': 'glamt', 'lat': 'gphit', 'time': 'time_counter'},
                  'dudy_linear': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'dvdx_linear': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'},
                  'dvdy_linear': {'lon': 'glamt', 'lat': 'gphit', 'time': 'time_counter'},
                  'coastal_mask': {'lon': 'lon', 'lat': 'lat'},
                  'bathy': {'lon': 'nav_lon', 'lat': 'nav_lat'},
                  'ssh': {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
                  }
    
    if mode == '3D':
        wfiles = sorted([data_path + f'psy4v3r1-daily_W_{time.year}-{time.month:02d}-{time.day:02d}.nc' for time in times])
        for wfile in wfiles:
            assert os.path.isfile(wfile), f"Can't find file {wilfe}"
        assert len(ufiles) == len(wfiles), "Different number of U, V and W files"

        filenames['W'] = {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': wfiles}
        filenames['W_linear_invdist'] =  {'lon': mesh_horiz, 'lat': mesh_horiz, 'depth': mesh_vert, 'data': wfiles}
        variables['W'] = 'vovecrtz'
        variables['W_linear_invdist'] = 'vovecrtz'
        dimensions['W'] = {'lon': 'glamf', 'lat': 'gphif', 'depth' : 'gdepw_0', 'time': 'time_counter'}
        dimensions['W_linear_invdist'] = {'lon': 'glamf', 'lat': 'gphif', 'depth' : 'gdepw_0', 'time': 'time_counter'}


    fieldset = parcels.FieldSet.from_nemo(filenames,
                                          variables,
                                          dimensions,
                                          chunksize=False)

    fieldset.U_linear_invdist.interp_method = 'linear_invdist_land_tracer'
    fieldset.V_linear_invdist.interp_method = 'linear_invdist_land_tracer'
    fieldset.dudx_linear.interp_method = 'linear_invdist_land_tracer'
    fieldset.dudy_linear.interp_method = 'linear_invdist_land_tracer'
    fieldset.dvdx_linear.interp_method = 'linear_invdist_land_tracer'
    fieldset.dvdy_linear.interp_method = 'linear_invdist_land_tracer'
    fieldset.coastal_mask.interp_method = 'nearest'
    fieldset.bathy.interp_method = 'nearest'
    fieldset.ssh.interp_method = 'linear_invdist_land_tracer'

    if mode == '3D':
        fieldset.W_linear_invdist.interp_method = 'linear_invdist_land_tracer'

    return fieldset

def CheckOutOfBounds(particle, fieldset, time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        print("Particle out of bounds: out of bounds (id, lon, lat, depth, time)", particle.id, particle.lon, particle.lat, particle.depth, particle.time)
        particle.delete()


class gradient_sampling_particle_simple(parcels.JITParticle):
    U_save_linear = parcels.Variable('U_save_linear', dtype=np.float32)
    V_save_linear = parcels.Variable('V_save_linear', dtype=np.float32)

    dudx_linear = parcels.Variable('dudx_linear', dtype=np.float32)
    dudy_linear = parcels.Variable('dudy_linear', dtype=np.float32)
    dvdx_linear = parcels.Variable('dvdx_linear', dtype=np.float32)
    dvdy_linear = parcels.Variable('dvdy_linear', dtype=np.float32)

    coastal = parcels.Variable('coastal', dtype=np.float32)

    bathymetry = parcels.Variable('bathymetry', dtype=np.float32)
    ssh = parcels.Variable('ssh', dtype=np.float32)

class gradient_sampling_particle_simple_3D(parcels.JITParticle):
    U_save_linear = parcels.Variable('U_save_linear', dtype=np.float32)
    V_save_linear = parcels.Variable('V_save_linear', dtype=np.float32)
    W_save_linear = parcels.Variable('W_save_linear', dtype=np.float32)

    dudx_linear = parcels.Variable('dudx_linear', dtype=np.float32)
    dudy_linear = parcels.Variable('dudy_linear', dtype=np.float32)
    dvdx_linear = parcels.Variable('dvdx_linear', dtype=np.float32)
    dvdy_linear = parcels.Variable('dvdy_linear', dtype=np.float32)
    coastal = parcels.Variable('coastal', dtype=np.float32)
    bathymetry = parcels.Variable('bathymetry', dtype=np.float32)
    ssh = parcels.Variable('ssh', dtype=np.float32)

class gradient_sampling_particle_minimal(parcels.JITParticle):
    dudx_linear = parcels.Variable('dudx_linear', dtype=np.float32)
    dudy_linear = parcels.Variable('dudy_linear', dtype=np.float32)
    dvdx_linear = parcels.Variable('dvdx_linear', dtype=np.float32)
    dvdy_linear = parcels.Variable('dvdy_linear', dtype=np.float32)

    coastal = parcels.Variable('coastal', dtype=np.float32)
    bathymetry = parcels.Variable('bathymetry', dtype=np.float32)

class gradient_sampling_particle_minimal_rk45(parcels.JITParticle):
    dudx_linear = parcels.Variable('dudx_linear', dtype=np.float32)
    dudy_linear = parcels.Variable('dudy_linear', dtype=np.float32)
    dvdx_linear = parcels.Variable('dvdx_linear', dtype=np.float32)
    dvdy_linear = parcels.Variable('dvdy_linear', dtype=np.float32)

    coastal = parcels.Variable('coastal', dtype=np.float32)
    bathymetry = parcels.Variable('bathymetry', dtype=np.float32)
    next_dt = parcels.Variable('next_dt', dtype=np.float32, initial=60)

class gradient_sampling_particle_minimal_rk45_backward(parcels.JITParticle):
    dudx_linear = parcels.Variable('dudx_linear', dtype=np.float32)
    dudy_linear = parcels.Variable('dudy_linear', dtype=np.float32)
    dvdx_linear = parcels.Variable('dvdx_linear', dtype=np.float32)
    dvdy_linear = parcels.Variable('dvdy_linear', dtype=np.float32)

    coastal = parcels.Variable('coastal', dtype=np.float32)
    bathymetry = parcels.Variable('bathymetry', dtype=np.float32)
    next_dt = parcels.Variable('next_dt', dtype=np.float32, initial=-60)

class gradient_sampling_particle_full(parcels.JITParticle):
    U_save = parcels.Variable('U_save', dtype=np.float64)
    V_save = parcels.Variable('V_save', dtype=np.float64)

    U_save_linear = parcels.Variable('U_save_linear', dtype=np.float64)
    V_save_linear = parcels.Variable('V_save_linear', dtype=np.float64)

    dudx_linear = parcels.Variable('dudx_linear', dtype=np.float64)
    dudy_linear = parcels.Variable('dudy_linear', dtype=np.float64)
    dvdx_linear = parcels.Variable('dvdx_linear', dtype=np.float64)
    dvdy_linear = parcels.Variable('dvdy_linear', dtype=np.float64)

    dudx_nearest = parcels.Variable('dudx_nearest', dtype=np.float64)
    dudy_nearest = parcels.Variable('dudy_nearest', dtype=np.float64)
    dvdx_nearest = parcels.Variable('dvdx_nearest', dtype=np.float64)
    dvdy_nearest = parcels.Variable('dvdy_nearest', dtype=np.float64)

    u_ln_lon_p1_large = parcels.Variable('u_ln_lon_p1_large', dtype=np.float64)
    u_ln_lon_m1_large = parcels.Variable('u_ln_lon_m1_large', dtype=np.float64)
    u_ln_lat_p1_large = parcels.Variable('u_ln_lat_p1_large', dtype=np.float64)
    u_ln_lat_m1_large = parcels.Variable('u_ln_lat_m1_large', dtype=np.float64)
    v_ln_lon_p1_large = parcels.Variable('v_ln_lon_p1_large', dtype=np.float64)
    v_ln_lon_m1_large = parcels.Variable('v_ln_lon_m1_large', dtype=np.float64)
    v_ln_lat_p1_large = parcels.Variable('v_ln_lat_p1_large', dtype=np.float64)
    v_ln_lat_m1_large = parcels.Variable('v_ln_lat_m1_large', dtype=np.float64)

    u_ln_lon_p1_medium = parcels.Variable('u_ln_lon_p1_medium', dtype=np.float64)
    u_ln_lon_m1_medium = parcels.Variable('u_ln_lon_m1_medium', dtype=np.float64)
    u_ln_lat_p1_medium = parcels.Variable('u_ln_lat_p1_medium', dtype=np.float64)
    u_ln_lat_m1_medium = parcels.Variable('u_ln_lat_m1_medium', dtype=np.float64)
    v_ln_lon_p1_medium = parcels.Variable('v_ln_lon_p1_medium', dtype=np.float64)
    v_ln_lon_m1_medium = parcels.Variable('v_ln_lon_m1_medium', dtype=np.float64)
    v_ln_lat_p1_medium = parcels.Variable('v_ln_lat_p1_medium', dtype=np.float64)
    v_ln_lat_m1_medium = parcels.Variable('v_ln_lat_m1_medium', dtype=np.float64)

    u_ln_lon_p1_small = parcels.Variable('u_ln_lon_p1_small', dtype=np.float64)
    u_ln_lon_m1_small = parcels.Variable('u_ln_lon_m1_small', dtype=np.float64)
    u_ln_lat_p1_small = parcels.Variable('u_ln_lat_p1_small', dtype=np.float64)
    u_ln_lat_m1_small = parcels.Variable('u_ln_lat_m1_small', dtype=np.float64)
    v_ln_lon_p1_small = parcels.Variable('v_ln_lon_p1_small', dtype=np.float64)
    v_ln_lon_m1_small = parcels.Variable('v_ln_lon_m1_small', dtype=np.float64)
    v_ln_lat_p1_small = parcels.Variable('v_ln_lat_p1_small', dtype=np.float64)
    v_ln_lat_m1_small = parcels.Variable('v_ln_lat_m1_small', dtype=np.float64)

    u_cg_lon_p1_large = parcels.Variable('u_cg_lon_p1_large', dtype=np.float64)
    u_cg_lon_m1_large = parcels.Variable('u_cg_lon_m1_large', dtype=np.float64)
    u_cg_lat_p1_large = parcels.Variable('u_cg_lat_p1_large', dtype=np.float64)
    u_cg_lat_m1_large = parcels.Variable('u_cg_lat_m1_large', dtype=np.float64)
    v_cg_lon_p1_large = parcels.Variable('v_cg_lon_p1_large', dtype=np.float64)
    v_cg_lon_m1_large = parcels.Variable('v_cg_lon_m1_large', dtype=np.float64)
    v_cg_lat_p1_large = parcels.Variable('v_cg_lat_p1_large', dtype=np.float64)
    v_cg_lat_m1_large = parcels.Variable('v_cg_lat_m1_large', dtype=np.float64)

    u_cg_lon_p1_medium = parcels.Variable('u_cg_lon_p1_medium', dtype=np.float64)
    u_cg_lon_m1_medium = parcels.Variable('u_cg_lon_m1_medium', dtype=np.float64)
    u_cg_lat_p1_medium = parcels.Variable('u_cg_lat_p1_medium', dtype=np.float64)
    u_cg_lat_m1_medium = parcels.Variable('u_cg_lat_m1_medium', dtype=np.float64)
    v_cg_lon_p1_medium = parcels.Variable('v_cg_lon_p1_medium', dtype=np.float64)
    v_cg_lon_m1_medium = parcels.Variable('v_cg_lon_m1_medium', dtype=np.float64)
    v_cg_lat_p1_medium = parcels.Variable('v_cg_lat_p1_medium', dtype=np.float64)
    v_cg_lat_m1_medium = parcels.Variable('v_cg_lat_m1_medium', dtype=np.float64)

    u_cg_lon_p1_small = parcels.Variable('u_cg_lon_p1_small', dtype=np.float64)
    u_cg_lon_m1_small = parcels.Variable('u_cg_lon_m1_small', dtype=np.float64)
    u_cg_lat_p1_small = parcels.Variable('u_cg_lat_p1_small', dtype=np.float64)
    u_cg_lat_m1_small = parcels.Variable('u_cg_lat_m1_small', dtype=np.float64)
    v_cg_lon_p1_small = parcels.Variable('v_cg_lon_p1_small', dtype=np.float64)
    v_cg_lon_m1_small = parcels.Variable('v_cg_lon_m1_small', dtype=np.float64)
    v_cg_lat_p1_small = parcels.Variable('v_cg_lat_p1_small', dtype=np.float64)
    v_cg_lat_m1_small = parcels.Variable('v_cg_lat_m1_small', dtype=np.float64)


def SampleGradients_simple(particle, fieldset, time):
    particle.U_save_linear = fieldset.U_linear_invdist[time, particle.depth, particle.lat, particle.lon]
    particle.V_save_linear = fieldset.V_linear_invdist[time, particle.depth, particle.lat, particle.lon]

    particle.dudx_linear = fieldset.dudx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dudy_linear = fieldset.dudy_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdx_linear = fieldset.dvdx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdy_linear = fieldset.dvdy_linear[time, particle.depth, particle.lat, particle.lon]

    particle.coastal = fieldset.coastal_mask[time, particle.depth, particle.lat, particle.lon]
    particle.bathymetry = fieldset.bathy[time, particle.depth, particle.lat, particle.lon]
    particle.ssh = fieldset.ssh[time, particle.depth, particle.lat, particle.lon]


def SampleGradients_simple_3D(particle, fieldset, time):
    particle.U_save_linear = fieldset.U_linear_invdist[time, particle.depth, particle.lat, particle.lon]
    particle.V_save_linear = fieldset.V_linear_invdist[time, particle.depth, particle.lat, particle.lon]
    particle.W_save_linear = fieldset.W_linear_invdist[time, particle.depth, particle.lat, particle.lon]

    particle.dudx_linear = fieldset.dudx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dudy_linear = fieldset.dudy_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdx_linear = fieldset.dvdx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdy_linear = fieldset.dvdy_linear[time, particle.depth, particle.lat, particle.lon]

    particle.coastal = fieldset.coastal_mask[time, particle.depth, particle.lat, particle.lon]
    particle.bathymetry = fieldset.bathy[time, particle.depth, particle.lat, particle.lon]
    particle.ssh = fieldset.ssh[time, particle.depth, particle.lat, particle.lon]


def SampleGradients_minimal(particle, fieldset, time):
    particle.dudx_linear = fieldset.dudx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dudy_linear = fieldset.dudy_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdx_linear = fieldset.dvdx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdy_linear = fieldset.dvdy_linear[time, particle.depth, particle.lat, particle.lon]

    particle.coastal = fieldset.coastal_mask[time, particle.depth, particle.lat, particle.lon]
    particle.bathymetry = fieldset.bathy[time, particle.depth, particle.lat, particle.lon]

def SampleGradients_full(particle, fieldset, time):
    particle.U_save, particle.V_save = fieldset.UV[time, particle.depth, particle.lat, particle.lon]

    particle.U_save_linear = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon]
    particle.V_save_linear = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon]

    particle.dudx_linear = fieldset.dudx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dudy_linear = fieldset.dudy_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdx_linear = fieldset.dvdx_linear[time, particle.depth, particle.lat, particle.lon]
    particle.dvdy_linear = fieldset.dvdy_linear[time, particle.depth, particle.lat, particle.lon]

    particle.dudx_nearest = fieldset.dudx_nearest[time, particle.depth, particle.lat, particle.lon]
    particle.dudy_nearest = fieldset.dudy_nearest[time, particle.depth, particle.lat, particle.lon]
    particle.dvdx_nearest = fieldset.dvdx_nearest[time, particle.depth, particle.lat, particle.lon]
    particle.dvdy_nearest = fieldset.dvdy_nearest[time, particle.depth, particle.lat, particle.lon]


    particle.u_cg_lat_p1_large, particle.v_cg_lat_p1_large = fieldset.UV[time, particle.depth, particle.lat + 1./24, particle.lon]
    particle.u_cg_lat_m1_large, particle.v_cg_lat_m1_large = fieldset.UV[time, particle.depth, particle.lat - 1./24, particle.lon]
    particle.u_cg_lon_p1_large, particle.v_cg_lon_p1_large = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 1./24]
    particle.u_cg_lon_m1_large, particle.v_cg_lon_m1_large = fieldset.UV[time, particle.depth, particle.lat, particle.lon - 1./24]

    particle.u_cg_lat_p1_medium, particle.v_cg_lat_p1_medium = fieldset.UV[time, particle.depth, particle.lat + 1./96, particle.lon]
    particle.u_cg_lat_m1_medium, particle.v_cg_lat_m1_medium = fieldset.UV[time, particle.depth, particle.lat - 1./96, particle.lon]
    particle.u_cg_lon_p1_medium, particle.v_cg_lon_p1_medium = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 1./96]
    particle.u_cg_lon_m1_medium, particle.v_cg_lon_m1_medium = fieldset.UV[time, particle.depth, particle.lat, particle.lon - 1./96]

    particle.u_cg_lat_p1_small, particle.v_cg_lat_p1_small = fieldset.UV[time, particle.depth, particle.lat + 1./384, particle.lon]
    particle.u_cg_lat_m1_small, particle.v_cg_lat_m1_small = fieldset.UV[time, particle.depth, particle.lat - 1./384, particle.lon]
    particle.u_cg_lon_p1_small, particle.v_cg_lon_p1_small = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 1./384]
    particle.u_cg_lon_m1_small, particle.v_cg_lon_m1_small = fieldset.UV[time, particle.depth, particle.lat, particle.lon - 1./384]

    particle.u_ln_lat_p1_large = fieldset.U_linear[time, particle.depth, particle.lat + 1./24, particle.lon]
    particle.u_ln_lat_m1_large = fieldset.U_linear[time, particle.depth, particle.lat - 1./24, particle.lon]
    particle.u_ln_lon_p1_large = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon + 1./24]
    particle.u_ln_lon_m1_large = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon - 1./24]
    particle.v_ln_lat_p1_large = fieldset.V_linear[time, particle.depth, particle.lat + 1./24, particle.lon]
    particle.v_ln_lat_m1_large = fieldset.V_linear[time, particle.depth, particle.lat - 1./24, particle.lon]
    particle.v_ln_lon_p1_large = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon + 1./24]
    particle.v_ln_lon_m1_large = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon - 1./24]

    particle.u_ln_lat_p1_medium = fieldset.U_linear[time, particle.depth, particle.lat + 1./96, particle.lon]
    particle.u_ln_lat_m1_medium = fieldset.U_linear[time, particle.depth, particle.lat - 1./96, particle.lon]
    particle.u_ln_lon_p1_medium = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon + 1./96]
    particle.u_ln_lon_m1_medium = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon - 1./96]
    particle.v_ln_lat_p1_medium = fieldset.V_linear[time, particle.depth, particle.lat + 1./96, particle.lon]
    particle.v_ln_lat_m1_medium = fieldset.V_linear[time, particle.depth, particle.lat - 1./96, particle.lon]
    particle.v_ln_lon_p1_medium = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon + 1./96]
    particle.v_ln_lon_m1_medium = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon - 1./96]

    particle.u_ln_lat_p1_small = fieldset.U_linear[time, particle.depth, particle.lat + 1./384, particle.lon]
    particle.u_ln_lat_m1_small = fieldset.U_linear[time, particle.depth, particle.lat - 1./384, particle.lon]
    particle.u_ln_lon_p1_small = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon + 1./384]
    particle.u_ln_lon_m1_small = fieldset.U_linear[time, particle.depth, particle.lat, particle.lon - 1./384]
    particle.v_ln_lat_p1_small = fieldset.V_linear[time, particle.depth, particle.lat + 1./384, particle.lon]
    particle.v_ln_lat_m1_small = fieldset.V_linear[time, particle.depth, particle.lat - 1./384, particle.lon]
    particle.v_ln_lon_p1_small = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon + 1./384]
    particle.v_ln_lon_m1_small = fieldset.V_linear[time, particle.depth, particle.lat, particle.lon - 1./384]
