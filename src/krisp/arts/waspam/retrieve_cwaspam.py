from krisp.data.readers import WaspamReader, WaspamData
from krisp.arts.tools import set_griddedfield3
from pathlib import Path
import pyarts
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import xarray as xr


@dataclass
class Measurement:
    data: WaspamData
    lines_cat: str
    atm_cat: str
    source: Path


def set_defaults(ws, meas: Measurement):
    # initialise workspace variables for retrieval
    ws.x = np.array([])
    ws.yf = np.array([])
    ws.jacobian = np.array([[]])

    # initialise retrieval framework
    ws.jacobianInit()
    ws.retrievalDefInit()

    # %% agendas
    ws.ppath_agendaSet(option="FollowSensorLosPath")
    ws.iy_main_agendaSet(option="Emission")
    ws.surface_rtprop_agendaSet(
        option="Specular_NoPol_ReflFix_SurfTFromt_surface")
    ws.ppath_step_agendaSet(option="GeometricPath")
    ws.iy_space_agendaSet(option="CosmicBackground")
    ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
    ws.water_p_eq_agendaSet(option="MK05")

    # %% calculations
    ws.iy_unit = "RJBT"
    ws.ppath_lmax = 1e3  # max spacing along path between points
    ws.ppath_lraytrace = 1e3  # spacing for ray tracing
    ws.rt_integration_option = "default"
    ws.nlteOff()
    ws.refellipsoidEarth(model="Sphere")

    # %% species and line absorption
    ws.abs_speciesSet(species=["H2O-161"])  # water vapour common isotopologue
    ws.abs_lines_per_speciesReadSpeciesSplitCatalog(basename=meas.lines_cat)
    ws.abs_lines_per_speciesCutoff(option="ByLine", value=10e9)

    # %% sensor
    ws.stokes_dim = 1
    ws.atmosphere_dim = 1

    # %% grids and planet
    ws.AtmosphereSet1D()
    now = datetime.now()
    print(f"-- {now}: Successfully set defaults in arts")


def set_atmosphere(ws, meas: Measurement):
    ws.xa = meas.data.apriori

    ws.p_grid = meas.data.p
    ws.lat_grid = np.array([])
    ws.lon_grid = np.array([])
    ws.AtmRawRead(basename=meas.atm_cat)
    ws.AtmFieldsCalc()
    ws.z_surface = [[16000]]
    gf_t = set_griddedfield3(
        pressure=meas.data.p,
        latitude=0,
        longitude=0,
        data=meas.data.temperature,
    )
    gf_h2o = set_griddedfield3(
        pressure=meas.data.p,
        latitude=0,
        longitude=0,
        data=meas.data.h2o,
    )
    ws.t_field_raw = gf_t
    ws.vmr_field_raw.value[0] = gf_h2o
    ws.AtmFieldsCalc()
    now = datetime.now()
    print(f"-- {now}: Successfully set atmosphere in arts")


def set_measurement(ws, meas: Measurement):
    timestamp = meas.data.mid
    dt = datetime.fromtimestamp(timestamp)
    time = pyarts.arts.Time(dt)
    arr_of_time = pyarts.arts.ArrayOfTime(1, time)
    ws.sensor_time = arr_of_time

    ws.y = meas.data.y + 2.73
    ws.yf = []
    ws.y_baseline = ws.y.value * 0
    ws.f_backend = meas.data.f

    # --- NOTE: HARDCODED!!! ---
    yerr = np.full_like(meas.data.f, 0.8)
    ws.covmat_seSet(covmat=pyarts.arts.Sparse(np.diag(yerr)))
    now = datetime.now()
    print(f"-- {now}: Successfully set measurements in arts")


def set_sensor(ws, meas: Measurement):
    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def sensor_response_agenda(ws):
        ws.AntennaOff()
        ws.sensor_responseInit(sensor_norm=1)
        ws.sensor_responseBackend(sensor_norm=1)

    n = len(meas.data.f)
    fe = meas.data.f[-1]
    fs = meas.data.f[0]
    res = (fe - fs) / n
    ws.backend_channel_responseFlat(resolution=res)
    ws.sensor_response_pol_grid = np.array([0])
    ws.sensor_response_dlos_grid = [[90, 0]]

    # -- CHECK THIS: Might be to simple
    ws.f_grid = np.linspace(fs - res, fe + res, n)
    ws.sensor_pos = [[16000]]
    ws.sensor_los = [[75]]
    ws.sensor_response_agenda.value.execute(ws)
    now = datetime.now()
    print(f"-- {now}: Successfully set sensor in arts")


def set_retrieval_quantities(ws, meas: Measurement):
    clip_lo = 1e-8
    clip_hi = 2e-3

    @pyarts.workspace.arts_agenda(ws=ws, set_agenda=True)
    def inversion_iterate_agenda(ws):
        ws.Ignore(ws.inversion_iteration_counter)
        ws.xClip(ijq=0, limit_low=clip_lo, limit_high=clip_hi)
        ws.x2artsAtmAndSurf()
        ws.x2artsSensor()
        ws.yCalc(y=ws.yf)
        ws.VectorAddElementwise(ws.yf, ws.yf, ws.y_baseline)
        ws.jacobianAdjustAndTransform()

    # -- HARDCODED xa_sx
    sx = np.full_like(meas.data.p, 5e-7)
    ws.retrievalAddAbsSpecies(
        species="H2O-161",
        covmat_block=pyarts.arts.Sparse(np.diag(sx)),
        unit="vmr",
        g1=ws.p_grid,
        g2=ws.lat_grid.value,
        g3=ws.lon_grid.value,
    )

    # -- Polyfit
    poly_order = 4
    # poly_var = [1, 10, 25, 25, 50]
    poly_var = [1, 0.1, 0.1, 0.1, 0.1]
    ws.retrievalAddPolyfit(
        poly_order=poly_order,
        no_pol_variation=0,
        no_los_variation=0,
        sensor_response_pol_grid=ws.sensor_response_pol_grid.value,
        sensor_response_dlos_grid=ws.sensor_response_dlos_grid.value,
    )

    for v in poly_var:
        ws.covmat_sxAddBlock(block=np.diag([v]))

    # -- Frequency shift
    fs_df = 1e3
    fs_var = 250e6
    fs_block = fs_var
    fs_inv_block = 1.0 / fs_var
    ws.retrievalAddFreqShift(
        covmat_block=[[fs_block]],
        covmat_inv_block=[[fs_inv_block]],
        df=fs_df,
    )
    n_extra = (poly_order + 1) + 1
    ws.xa.value = np.append(ws.xa.value, np.zeros(n_extra))
    now = datetime.now()
    print(f"-- {now}: Successfully set retrieval quantities in arts")


def set_sinefit(ws, period_lengths, period_var):
    elements = 2
    sine_param = elements * len(period_lengths)
    covmat_sine = pyarts.arts.Sparse(
        np.diag([period_var] * elements),)

    ws.retrievalAddSinefit(
        period_lengths=[float(p) for p in period_lengths],
        covmat_block=covmat_sine,
        no_pol_variation=0,
        no_los_variation=0,
        no_mblock_variation=0,
        sensor_response_pol_grid=ws.sensor_response_pol_grid.value,
        sensor_response_dlos_grid=ws.sensor_response_dlos_grid.value
    )
    for p in period_lengths:
        now = datetime.now()
        print(f"-- {now}: Adding sine fit with period: {p / 1e6} MHz")

    n = 2 * len(period_lengths)
    ws.xa.value = np.append(ws.xa.value, np.zeros(n))


def compute_checks(ws):
    """touch unused fields, close the retrieval definition and run the checks."""

    ws.Touch(ws.rtp_mag)  # magnetic field
    ws.Touch(ws.rtp_los)  # line of sight
    ws.Touch(ws.rtp_pressure)  # pressure
    ws.Touch(ws.rtp_temperature)  # temperature
    ws.Touch(ws.rtp_nlte)  # NLTE temperature ratio
    ws.Touch(ws.rtp_vmr)  # volume mixing ratio

    ws.Touch(ws.wind_u_field)
    ws.Touch(ws.wind_v_field)
    ws.Touch(ws.wind_w_field)
    ws.Touch(ws.time)
    ws.cloudboxOff()

    # close the retrieval definition and prepare the related WSVs
    ws.retrievalDefClose()

    # compute checks
    ws.propmat_clearsky_agendaAuto()
    ws.atmgeom_checkedCalc()
    ws.lbl_checkedCalc()
    ws.atmfields_checkedCalc()
    ws.cloudbox_checkedCalc()
    ws.sensor_checkedCalc()
    ws.propmat_clearsky_agenda_checkedCalc()

    now = datetime.now()
    print(f"-- {now}: Successfully computed checks in arts")


def compute_post_oem(ws):
    ws.avkCalc()
    ws.covmat_ssCalc()
    ws.covmat_soCalc()
    ws.retrievalErrorsExtract()
    ws.x2artsSensor()


def run_OEM(ws, meas: Measurement):
    now = datetime.now()
    print(f"-- {now}: Starting retrieval on {meas.source}")
    ws.OEM(
        method="lm",
        lm_ga_settings=[2, 2, 2, 1024, 1, 1024],
        display_progress=1,
        max_iter=20,
        verbosity=10,
    )
    now = datetime.now()
    print(f"-- {now}: Done.")


def save_retrieval(ws, fn):
    n = len(ws.p_grid.value)
    avk = ws.avk.value[0:n, 0:n]
    q = ws.x.value[0:n]
    qa = ws.xa.value[0:n]
    residual = ws.y.value - ws.yf.value
    ds = xr.Dataset(
        data_vars={
            "y": (["f"], ws.y.value),
            "yf": (["f"], ws.yf.value),
            "yb": (["f"], ws.y_baseline.value),
            "residual": (["f"], residual),
            "avk": (["p"], avk),
            "q": (["p"], q),
            "qa": (["p"], qa),
        },
        coords={
            "f": ws.f_backend.value,
            "p": ws.p_grid.value,
        },
    )
    ds.to_netcdf(fn)
    now = datetime.now()
    print(f"-- {now}: Saved data in {fn}")
