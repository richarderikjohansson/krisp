from krisp.data.readers import WaspamReader, WaspamData, ConfigReader
from krisp.arts.tools import set_griddedfield3
from krisp.arts.logger import get_retrieval_logger
import pyarts
import numpy as np
from datetime import datetime
import xarray as xr
from pathlib import Path


class WaspamRetrieve:
    def __init__(self, mfile, cfile):
        self.mfile = mfile
        self.cfile = cfile
        self.logger = get_retrieval_logger()
        self.ws = pyarts.Workspace()

    def _load_measurement_and_config(self):
        self.meas = WaspamReader(fp=self.mfile).load_data()
        self.config = ConfigReader.load(self.cfile)

    def _set_defaults(self):
        # initialise workspace variables for retrieval
        self.ws.x = np.array([])
        self.ws.yf = np.array([])
        self.ws.jacobian = np.array([[]])

        # initialise retrieval framework
        self.ws.jacobianInit()
        self.ws.retrievalDefInit()

        # %% agendas
        self.ws.ppath_agendaSet(option="FollowSensorLosPath")
        self.ws.iy_main_agendaSet(option="Emission")
        self.ws.surface_rtprop_agendaSet(
            option="Specular_NoPol_ReflFix_SurfTFromt_surface")
        self.ws.ppath_step_agendaSet(option="GeometricPath")
        self.ws.iy_space_agendaSet(option="CosmicBackground")
        self.ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
        self.ws.water_p_eq_agendaSet(option="MK05")

        # %% calculations
        self.ws.iy_unit = "RJBT"
        self.ws.ppath_lmax = self.config["defaults"]["ppath_lmax"]
        self.ws.ppath_lraytrace = self.config["defaults"]["ppath_lraytrace"]
        self.ws.rt_integration_option = "default"
        self.ws.nlteOff()
        self.ws.refellipsoidEarth(model="Sphere")

        # %% species and line absorption
        # water vapour common isotopologue
        self.ws.abs_speciesSet(species=["H2O-161"])
        self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
            basename=self.lines)
        self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=10e9)

        # %% sensor
        self.ws.stokes_dim = 1
        self.ws.atmosphere_dim = 1

        # %% grids and planet
        self.ws.AtmosphereSet1D()
        self.logger.info("Successfully set defaults in arts")

    def _set_atmosphere(self):

        self.ws.xa = self.meas.apriori

        self.ws.p_grid = self.meas.p
        self.ws.lat_grid = np.array([])
        self.ws.lon_grid = np.array([])
        self.ws.AtmRawRead(basename=self.atm)
        self.ws.z_surface = [[self.config["atmosphere"]["z_surface"]]]
        gf_t = set_griddedfield3(
            pressure=self.meas.p,
            latitude=0,
            longitude=0,
            data=self.meas.temperature,
        )
        gf_h2o = set_griddedfield3(
            pressure=self.meas.p,
            latitude=0,
            longitude=0,
            data=self.meas.h2o,
        )
        gf_z = set_griddedfield3(self.meas.p,
                                 latitude=0,
                                 longitude=0,
                                 data=self.meas.z)
        # self.ws.t_field_raw = gf_t
        # self.ws.vmr_field_raw.value[0] = gf_h2o
        self.ws.t_field_raw = gf_t
        self.ws.z_field_raw = gf_z
        self.ws.vmr_field_raw.value[0] = gf_h2o
        self.ws.AtmFieldsCalc(vmr_zeropadding=1)
        self.logger.info("Successfully set atmosphere in arts")

    def _set_measurement(self):
        timestamp = self.meas.mid
        dt = datetime.fromtimestamp(timestamp)
        time = pyarts.arts.Time(dt)
        arr_of_time = pyarts.arts.ArrayOfTime(1, time)
        self.ws.sensor_time = arr_of_time

        self.ws.y = self.meas.y + self.config["measurement"]["y_offset"]
        self.ws.yf = []
        self.ws.y_baseline = self.ws.y.value * 0
        self.ws.f_backend = self.meas.f

        # --- NOTE: HARDCODED!!! ---
        yerr = np.full_like(
            self.meas.f, self.config["measurement"]["yerr_const"],)

        self.ws.covmat_seSet(covmat=pyarts.arts.Sparse(np.diag(yerr)))
        self.logger.info("Successfully set measurements in arts")

    def _set_sensor(self):
        @pyarts.workspace.arts_agenda(ws=self.ws, set_agenda=True)
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.sensor_responseInit(sensor_norm=1)
            ws.sensor_responseBackend(sensor_norm=1)

        n = len(self.meas.f)
        fe = self.meas.f[-1]
        fs = self.meas.f[0]
        res = (fe - fs) / n
        self.ws.backend_channel_responseFlat(resolution=res)
        self.ws.sensor_response_pol_grid = np.array(
            self.config["sensor"]["sensor_response_pol_grid"])
        self.ws.sensor_response_dlos_grid = [
            self.config["sensor"]["sensor_response_dlos_grid"]]

        # -- CHECK THIS: Might be to simple
        self.ws.f_grid = np.linspace(fs - res, fe + res, n)
        self.ws.sensor_pos = [[self.config["sensor"]["sensor_pos"]]]
        self.ws.sensor_los = [[self.config["sensor"]["sensor_los"]]]
        self.ws.sensor_response_agenda.value.execute(self.ws)
        self.logger.info("Successfully set sensor variables in arts")

    def _set_retrieval_quantities(self):
        clip_lo = 1e-8
        clip_hi = 2e-3

        @pyarts.workspace.arts_agenda(ws=self.ws, set_agenda=True)
        def inversion_iterate_agenda(ws):
            ws.Ignore(ws.inversion_iteration_counter)
            ws.xClip(ijq=0, limit_low=clip_lo, limit_high=clip_hi)
            ws.x2artsAtmAndSurf()
            ws.x2artsSensor()
            ws.yCalc(y=ws.yf)
            ws.VectorAddElementwise(ws.yf, ws.yf, ws.y_baseline)
            ws.jacobianAdjustAndTransform()

        # -- HARDCODED xa_sx
        sx = np.full_like(self.meas.p, self.config["retrieval"]["sx"])
        self.ws.retrievalAddAbsSpecies(
            species="H2O-161",
            covmat_block=pyarts.arts.Sparse(np.diag(sx)),
            unit="vmr",
            g1=self.ws.p_grid,
            g2=self.ws.lat_grid.value,
            g3=self.ws.lon_grid.value,
        )

        # -- Polyfit
        # poly_order = 4
        # poly_var = [1, 0.1, 0.1, 0.1, 0.1]
        self.ws.retrievalAddPolyfit(
            poly_order=self.config["retrieval"]["poly_order"],
            no_pol_variation=0,
            no_los_variation=0,
            sensor_response_pol_grid=self.ws.sensor_response_pol_grid.value,
            sensor_response_dlos_grid=self.ws.sensor_response_dlos_grid.value,
        )

        for v in self.config["retrieval"]["poly_var"]:
            self.ws.covmat_sxAddBlock(block=np.diag([v]))

        # -- Frequency shift
        fs_df = self.config["retrieval"]["fs_df"]
        fs_var = self.config["retrieval"]["fs_var"]
        fs_block = fs_var
        fs_inv_block = 1.0 / fs_var
        self.ws.retrievalAddFreqShift(
            covmat_block=[[fs_block]],
            covmat_inv_block=[[fs_inv_block]],
            df=fs_df,
        )
        n_extra = (self.config["retrieval"]["poly_order"] + 1) + 1
        self.ws.xa.value = np.append(self.ws.xa.value, np.zeros(n_extra))
        if "period_lengths" in self.config["retrieval"].keys():
            period_lengths = self.config["retrieval"]["period_lengths"]
            period_var = self.config["retrieval"]["period_var"]
            elements = 2
            covmat_sine = pyarts.arts.Sparse(
                np.diag([period_var] * elements),)

            self.ws.retrievalAddSinefit(
                period_lengths=[float(p) for p in period_lengths],
                covmat_block=covmat_sine,
                no_pol_variation=0,
                no_los_variation=0,
                no_mblock_variation=0,
                sensor_response_pol_grid=self.ws.sensor_response_pol_grid.value,
                sensor_response_dlos_grid=self.ws.sensor_response_dlos_grid.value
            )
            for p in period_lengths:
                self.logger.warning(
                    f"Adding sine fit with period: {p / 1e6} MHz")

            n = 2 * len(period_lengths)
            self.ws.xa.value = np.append(self.ws.xa.value, np.zeros(n))

        self.logger.info("Successfully set retrieval quantities in arts")

    def _compute_checks(self):
        """touch unused fields, close the retrieval definition and run the checks."""

        self.ws.Touch(self.ws.rtp_mag)  # magnetic field
        self.ws.Touch(self.ws.rtp_los)  # line of sight
        self.ws.Touch(self.ws.rtp_pressure)  # pressure
        self.ws.Touch(self.ws.rtp_temperature)  # temperature
        self.ws.Touch(self.ws.rtp_nlte)  # NLTE temperature ratio
        self.ws.Touch(self.ws.rtp_vmr)  # volume mixing ratio

        self.ws.Touch(self.ws.wind_u_field)
        self.ws.Touch(self.ws.wind_v_field)
        self.ws.Touch(self.ws.wind_w_field)
        self.ws.Touch(self.ws.time)
        self.ws.cloudboxOff()

        # close the retrieval definition and prepare the related self.wsVs
        self.ws.retrievalDefClose()

        # compute checks
        self.ws.propmat_clearsky_agendaAuto()
        self.ws.atmgeom_checkedCalc()
        self.ws.lbl_checkedCalc()
        self.ws.atmfields_checkedCalc()
        self.ws.cloudbox_checkedCalc()
        self.ws.sensor_checkedCalc()
        self.ws.propmat_clearsky_agenda_checkedCalc()

    def _compute_post_oem(self):
        self.ws.avkCalc()
        self.ws.covmat_ssCalc()
        self.ws.covmat_soCalc()
        self.ws.retrievalErrorsExtract()
        self.ws.x2artsSensor()

    def _run_OEM(self):
        self.logger.info(f"Starting retrieval on {self.mfile}")
        self.ws.OEM(
            method="lm",
            lm_ga_settings=self.config["retrieval"]["lm_ga_settings"],
            display_progress=1,
            max_iter=self.config["retrieval"]["max_iter"],
            verbosity=10,
        )

    def _get_catalogue_data(self):
        home = Path.home()
        arts_dir = home / ".cache/arts"
        self.lines = str(arts_dir / self.config["paths"]["lines"]) + "/"
        self.atm = str(arts_dir / self.config["paths"]["xml"])

    def run(self):
        self._load_measurement_and_config()
        self._get_catalogue_data()
        self._set_defaults()
        self._set_atmosphere()
        self._set_measurement()
        self._set_sensor()
        self._set_retrieval_quantities()
        self._compute_checks()
        self._run_OEM()
        self._compute_post_oem()
        self.outpath = self.config["paths"]["outdir"]
        self._save_retrieval()

    def _save_retrieval(self):
        dt = datetime.fromtimestamp(self.meas.mid)
        dt = dt.isoformat()
        outpath = f"{self.outpath}/{dt}.nc"
        n = len(self.ws.p_grid.value)
        avk = self.ws.avk.value[0:n, 0:n]
        q = self.ws.x.value[0:n]
        qa = self.ws.xa.value[0:n]
        jacobian = self.ws.jacobian.value[:, 0:n]
        residual = self.ws.y.value - self.ws.yf.value
        ss = self.ws.retrieval_eo.value[0:n]
        eo = self.ws.retrieval_ss.value[0:n]
        diagnostics = self.ws.oem_diagnostics.value
        ds = xr.Dataset(
            data_vars={
                "y": (["f"], self.ws.y.value),
                "yf": (["f"], self.ws.yf.value),
                "yb": (["f"], self.ws.y_baseline.value),
                "residual": (["f"], residual),
                "avk": (["p", "pk"], avk),
                "q": (["p"], q),
                "qa": (["p"], qa),
                "jacobian": (["f", "p"], jacobian),
                "ss": (["p"], ss),
                "eo": (["p"], eo),
            },
            coords={
                "f": self.ws.f_backend.value,
                "p": self.ws.p_grid.value,
            },
        )
        ds.attrs = {
            "convergence": diagnostics[0],
            "start_cost": diagnostics[1],
            "end_cost": diagnostics[2],
            "end_ycost": diagnostics[3],
            "iterations": diagnostics[4]

        }
        ds.to_netcdf(outpath)
        outpath = Path(outpath)
        self.logger.info(f"Saved retrieval in {outpath}")
