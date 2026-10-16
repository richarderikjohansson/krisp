from krisp.data.readers import ConfigReader, DataReader
from krisp.arts.tools import set_griddedfield3, make_sx_o3
from krisp.arts.logger import get_logger
from krisp.filesystem.paths import find_apriori
import pyarts
import numpy as np
from datetime import datetime
import xarray as xr
from pathlib import Path
from h5py import File
from importlib.resources import files


class KimraRetrieve:
    def __init__(self, mfile, cfile, lims=None):
        self.mfile = mfile
        self.cfile = cfile
        self.logger = get_logger()
        self.ws = pyarts.Workspace()
        self.lims = lims
        self.ddir = get_data_dir()

    def load_measurement_and_config(self):
        self.meas, self.attrs = DataReader.load(
            file=self.mfile, lims=self.lims)
        self.config = ConfigReader.load(self.cfile)
        self.month = datetime.fromtimestamp(self.attrs.id.values).month

    def _set_defaults(self):
        # initialise workspace variables for retrieval

        # initialise retrieval framework
        self.ws.jacobianInit()
        self.ws.retrievalDefInit()

        # %% agendas
        self.ws.ppath_agendaSet(option="FollowSensorLosPath")
        self.ws.iy_main_agendaSet(option="Clearsky")
        self.ws.surface_rtprop_agendaSet(
            option="Specular_NoPol_ReflFix_SurfTFromt_surface")
        self.ws.ppath_step_agendaSet(option="GeometricPath")
        self.ws.iy_space_agendaSet(option="CosmicBackground")
        self.ws.iy_surface_agendaSet(option="UseSurfaceRtprop")
        self.ws.water_p_eq_agendaSet(option="MK05")
        self.ws.gas_scattering_agendaSet()

        # %% calculations
        self.ws.iy_unit = "PlanckBT"
        self.ws.ppath_lmax = self.config["defaults"]["ppath_lmax"]
        self.ws.ppath_lraytrace = self.config["defaults"]["ppath_lraytrace"]
        self.ws.rt_integration_option = "default"
        self.ws.nlteOff()
        self.ws.refellipsoidEarth(model="Sphere")

        # %% species and line absorption
        # water vapour common isotopologue

        self.ws.abs_speciesSet(
            species=[
                "O3",
                "H2O",
                "O2",
                "HNO3",
            ]
        )
        self.ws.abs_lines_per_speciesReadSpeciesSplitCatalog(
            basename=self.lines)
        self.ws.abs_lines_per_speciesCutoff(option="ByLine", value=750e9)
        self.ws.abs_lines_per_speciesTurnOffLineMixing()

        # %% sensor
        self.ws.stokes_dim = 1
        self.ws.atmosphere_dim = 1

        # %% grids and planet
        self.ws.AtmosphereSet1D()
        self.logger.info("Successfully set defaults in arts")

    def _set_atmosphere(self):
        apriori_path = find_apriori(self.config)
        with File(apriori_path, "r") as h5:
            data = h5[str(self.month)]
            pret = data["pret"][()]
            o3a = data["o3a"][()]

        self.ws.p_grid = pret
        self.ws.lat_grid = np.array([])
        self.ws.lon_grid = np.array([])
        self.ws.AtmRawRead(basename=self.atm)
        self.ws.z_surface = [[self.config["atmosphere"]["z_surface"]]]
        gf_t = set_griddedfield3(
            pressure=self.meas.p.values,
            latitude=0,
            longitude=0,
            data=self.meas.temp.values,
        )
        gf_h2o = set_griddedfield3(
            pressure=self.meas.p.values,
            latitude=0,
            longitude=0,
            data=self.meas.h2o.values,
        )
        gf_z = set_griddedfield3(
            self.meas.p.values,
            latitude=0,
            longitude=0,
            data=self.meas.z.values,
        )
        gf_o3 = set_griddedfield3(pret, 0, 0, o3a)
        self.ws.t_field_raw = gf_t
        self.ws.z_field_raw = gf_z
        self.ws.vmr_field_raw.value[0] = gf_o3
        self.ws.vmr_field_raw.value[1] = gf_h2o
        self.ws.AtmFieldsCalc(vmr_zeropadding=1)
        self.logger.info("Successfully set atmosphere in arts")

    def _set_measurement(self):
        timestamp = self.attrs.id.values
        dt = datetime.fromtimestamp(timestamp)
        time = pyarts.arts.Time(dt)
        arr_of_time = pyarts.arts.ArrayOfTime(1, time)
        self.ws.sensor_time = arr_of_time

        self.ws.y = self.meas.y.values
        # self.ws.y_baseline = self.ws.y.value * 0
        self.ws.f_backend = self.meas.fb.values

        fill = np.var(self.meas.y.values[0:100])
        yerr = np.full_like(self.meas.fb, fill)

        self.ws.covmat_seSet(covmat=pyarts.arts.Sparse(np.diag(yerr)))
        self.logger.info("Successfully set measurements in arts")

    def _set_sensor(self):
        @pyarts.workspace.arts_agenda(ws=self.ws, set_agenda=True)
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.sensor_responseInit(sensor_norm=1)
            ws.sensor_responseBackend(sensor_norm=1)

        n = len(self.meas.fb.values)
        fe = self.meas.fb.values[-1]
        fs = self.meas.fb.values[0]
        res = (fe - fs) / n
        self.ws.backend_channel_responseFlat(resolution=res)
        self.ws.sensor_response_pol_grid = np.array(
            self.config["sensor"]["sensor_response_pol_grid"]
        )
        self.ws.sensor_response_dlos_grid = [
            self.config["sensor"]["sensor_response_dlos_grid"]]

        # -- CHECK THIS: Might be to simple
        self.ws.f_grid = np.linspace(fs - res, fe + res, 5000)
        self.ws.sensor_pos = [[self.config["sensor"]["sensor_pos"]]]
        self.ws.sensor_los = [[self.attrs.za.values]]
        self.ws.sensor_response_agenda.value.execute(self.ws)
        self.logger.info("Successfully set sensor variables in arts")

    def _set_retrieval_quantities(self):
        clip_lo = 1e-8
        clip_hi = 2e-3

        @pyarts.workspace.arts_agenda(ws=self.ws, set_agenda=True)
        def inversion_iterate_agenda(ws):
            ws.Ignore(ws.inversion_iteration_counter)
            # ws.xClip(ijq=0, limit_low=clip_lo, limit_high=clip_hi)
            ws.vmr_fieldClip(species="ALL", limit_low=0)
            ws.x2artsAtmAndSurf()
            ws.x2artsSensor()
            ws.yCalc(y=ws.yf)
            ws.VectorAddElementwise(ws.yf, ws.yf, ws.y_baseline)
            ws.jacobianAdjustAndTransform()

        # o3
        # sx = make_sx_o3(self.ws.p_grid.value / 1e2)
        sx = np.full_like(self.ws.p_grid.value, 0.5)
        self.ws.retrievalAddAbsSpecies(
            species="O3",
            covmat_block=pyarts.arts.Sparse(np.diag(sx)),
            unit="rel",
            g1=self.ws.p_grid,
            g2=self.ws.lat_grid.value,
            g3=self.ws.lon_grid.value,
        )
        # h2o
        sx = np.full_like(self.ws.p_grid.value, 0.8)
        self.ws.retrievalAddAbsSpecies(
            species="H2O",
            covmat_block=pyarts.arts.Sparse(np.diag(sx)),
            unit="rel",
            g1=self.ws.p_grid,
            g2=self.ws.lat_grid.value,
            g3=self.ws.lon_grid.value,
        )

        # o2
        sx = np.full_like(self.ws.p_grid.value, 0.8)
        self.ws.retrievalAddAbsSpecies(
            species="O2",
            covmat_block=pyarts.arts.Sparse(np.diag(sx)),
            unit="rel",
            g1=self.ws.p_grid,
            g2=self.ws.lat_grid.value,
            g3=self.ws.lon_grid.value,
        )

        # -- Polyfit
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
        if "period_lengths" in self.config["retrieval"].keys():
            period_lengths = self.config["retrieval"]["period_lengths"]
            period_var = self.config["retrieval"]["period_var"]
            elements = 2
            covmat_sine = pyarts.arts.Sparse(
                np.diag([period_var] * elements),
            )

            self.ws.retrievalAddSinefit(
                period_lengths=[float(p) for p in period_lengths],
                covmat_block=covmat_sine,
                no_pol_variation=0,
                no_los_variation=0,
                no_mblock_variation=0,
                sensor_response_pol_grid=self.ws.sensor_response_pol_grid.value,
                sensor_response_dlos_grid=self.ws.sensor_response_dlos_grid.value,
            )
            for p in period_lengths:
                self.logger.warning(
                    f"Adding sine fit with period: {p / 1e6} MHz")

        self.logger.info("Successfully set retrieval quantities in arts")

    def _compute_checks(self):
        """touch unused fields, close the retrieval definition and run the checks."""

        # self.ws.Touch(self.ws.rtp_mag)  # magnetic field
        # self.ws.Touch(self.ws.rtp_los)  # line of sight
        # self.ws.Touch(self.ws.rtp_pressure)  # pressure
        # self.ws.Touch(self.ws.rtp_temperature)  # temperature
        # self.ws.Touch(self.ws.rtp_nlte)  # NLTE temperature ratio
        # self.ws.Touch(self.ws.rtp_vmr)  # volume mixing ratio

        # self.ws.Touch(self.ws.wind_u_field)
        # self.ws.Touch(self.ws.wind_v_field)
        # self.ws.Touch(self.ws.wind_w_field)
        # self.ws.Touch(self.ws.time)
        self.ws.cloudboxOff()

        # compute checks and load abslookup
        it = self.ddir.rglob("*")
        found = False
        for file in it:
            name = file.name
            if name == f"{self.suffix}_abslookup.xml":
                found = True
                self.ws.ReadXML(self.ws.abs_lookup, str(file))
                self.logger.info(f"Read abslookup for {
                                 self.suffix} from {file}")
                self.ws.abs_lookupAdapt()
                self.ws.propmat_clearsky_agendaAuto(use_abs_lookup=1)
        if not found:
            self.logger.info(
                f"Could not find abslookup file for {self.suffix}")
            self.ws.propmat_clearsky_agendaAuto()
        self.ws.atmgeom_checkedCalc()
        self.ws.lbl_checkedCalc()
        self.ws.atmfields_checkedCalc()
        self.ws.cloudbox_checkedCalc()
        self.ws.sensor_checkedCalc()
        self.ws.propmat_clearsky_agenda_checkedCalc()

        # close the retrieval definition and prepare the related self.wsVs
        self.ws.retrievalDefClose()
        self.ws.xaStandard()

    def _run_OEM(self):
        self.ws.x = []
        self.ws.jacobian = []
        self.ws.abs_f_interp_order = 1
        self.ws.yf = []

        self.logger.info(f"Starting retrieval on {self.mfile}")
        self.ws.OEM(
            method="lm",
            lm_ga_settings=self.config["retrieval"]["lm_ga_settings"],
            display_progress=1,
            max_iter=self.config["retrieval"]["max_iter"],
            verbosity=10,
        )

    def _compute_post_oem(self):
        self.ws.avkCalc()
        self.ws.covmat_ssCalc()
        self.ws.covmat_soCalc()
        self.ws.retrievalErrorsExtract()
        self.ws.x2artsSensor()

    def _save_retrieval(self):

        # set path
        dt = datetime.fromtimestamp(self.attrs.id.values)
        dt = dt.isoformat()
        outdir = get_outdir()
        outpath = f"{outdir}/{dt}_{self.suffix}.nc"

        # get data from WSV
        n = len(self.ws.p_grid.value)
        avk = self.ws.avk.value[0:n, 0:n]
        covmat_so = self.ws.covmat_so.value[0:n, 0:n]
        covmat_ss = self.ws.covmat_ss.value[0:n, 0:n]
        x = self.ws.x.value
        o3a = self.ws.vmr_field.value[0, :, 0, 0]
        o3 = x[0:n] * o3a
        jacobian = self.ws.jacobian.value[:, 0:n]
        residual = self.ws.y.value - self.ws.yf.value
        diagnostics = self.ws.oem_diagnostics.value
        dxdy = self.ws.dxdy.value
        self.ws.MatrixCreate(name="se_matrix")
        self.ws.MatrixFromCovarianceMatrix(
            output=self.ws.se_matrix, input=self.ws.covmat_se)
        se = self.ws.se_matrix.value
        zgrid = self.ws.z_field.value.flatten()

        # make dataset
        ds = xr.Dataset(
            data_vars={
                "y": (["f"], self.ws.y.value),
                "yf": (["f"], self.ws.yf.value),
                "yb": (["f"], self.ws.y_baseline.value),
                "residual": (["f"], residual),
                "avk": (["p", "pk"], avk),
                "o3": (["p"], o3),
                "o3a": (["p"], o3a),
                "jacobian": (["f", "p"], jacobian),
                "covmat_ss": (["p", "pk"], covmat_ss),
                "covmat_so": (["p", "pk"], covmat_so),
                # "dxdy": (["x", "f"], dxdy),
                # "covmat_se": (["f", "fk"], se),
                "z": (["p"], zgrid),
            },
            coords={"f": self.ws.f_backend.value,
                    "p": self.ws.p_grid.value, "x": x},
        )
        ds.attrs = {
            "convergence": diagnostics[0],
            "start_cost": diagnostics[1],
            "end_cost": diagnostics[2],
            "end_ycost": diagnostics[3],
            "iterations": diagnostics[4],
            "timestamp": self.attrs.id.values,
        }
        ds.to_netcdf(outpath)
        outpath = Path(outpath)
        self.logger.info(f"Saved retrieval in {outpath}")

    def slice_spectra(self, start, end, suffix):
        self.meas = self.meas.sel(fb=slice(start, end))
        self.suffix = suffix
        self.logger.info(
            f"Loaded and sliced data for {self.suffix} (f: {
                self.meas.fb.values[0] / 1e9}-{self.meas.fb.values[-1] / 1e9} GHz )"
        )

    def _get_catalogue_data(self):
        home = Path.home()
        arts_dir = home / ".cache/arts"
        self.lines = str(arts_dir / self.config["paths"]["lines"]) + "/"
        self.atm = str(arts_dir / self.config["paths"]["xml"])

    def run(self):
        self._get_catalogue_data()
        self._set_defaults()
        self._set_atmosphere()
        self._set_measurement()
        self._set_sensor()
        self._set_retrieval_quantities()
        self._compute_checks()
        self._run_OEM()
        self._compute_post_oem()
        self._save_retrieval()
        print("---")


def get_outdir():
    home = Path.home()
    wp = home / "Data" / "kimra"
    l2 = wp / "L2"
    if not l2.exists():
        l2.mkdir(parents=True)
    return l2


def get_data_dir():
    ddir = files("krisp.arts").joinpath("data")
    return Path(ddir)
