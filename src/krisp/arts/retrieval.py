import pyarts
from pyarts.arts import GriddedField3
from krisp.data.readers import Configuration, Attributes, ConfigReader, DataReader
from krisp.data.tools import interp_from_ecmwf_to_pret, make_ds_for_arts
from krisp.physics.atmosphere import altitude_from_pressure
from krisp.filesystem.paths import find_default_configs
from krisp.arts.agendas import default_agendas
from krisp.filesystem.paths import find_arts_paths
import xarray as xr
from typing import Tuple
from datetime import datetime
from pathlib import Path
import numpy as np


class Retrieval:
    def __init__(self, file: Path, verbosity: int = 1):
        self.start_of_retrieval = datetime.now().timestamp()
        self.arts = pyarts.workspace.Workspace(verbosity=verbosity)
        self.fp = file

    def set_default_oem(self):
        self.set_data_and_config(config="default")
        self.set_default_agendas()
        self.set_retrieval_grids()
        self.set_atmosphere()
        self.perform_checks()

    def set_data_and_config(self, config: Path | str = "default"):
        self.data, self.attrs = DataReader.load(self.fp, attrs=True, group="measurement")
        if config == "default":
            config = find_default_configs(self.attrs)
        self.config = ConfigReader.load(config)

    def set_default_agendas(self):
        self.arts = default_agendas(self.arts)

    def set_retrieval_grids(self):
        fs, fe = self.config.f_start, self.config.f_end
        fmask = (fs <= self.data.fb.values) & (self.data.fb.values <= fe)
        self.arts.y = self.data.y.values[fmask][50:-50]
        self.arts.f_backend = self.data.fb.values[fmask][50:-50]
        self.arts.f_grid = np.arange(fs, fe, step=self.config.f_res)
        self.arts.p_grid = self.data.p_ret.values

    def perform_checks(self):

        self.arts.propmat_clearsky_agenda_checkedCalc()
        self.arts.atmfields_checkedCalc()
        self.arts.atmgeom_checkedCalc()
        self.arts.cloudbox_checkedCalc()
        self.arts.lbl_checkedCalc()
        self.arts.sensor_checkedCalc()

    def set_atmosphere(self):
        atm = AtmosphereAndRT(obj=self)
        atm.execute()

    def init_OEM(self):
        ret = RetrievalOEMInit(obj=self)
        ret.set_ret_quantities()
        ret.define_outputs()

    def run_OEM(self):
        self.init_OEM()

        @pyarts.workspace.arts_agenda
        def inversion_iterate_agenda(ws):
            """Custom inversion iterate agenda to ignore bad partition functions."""
            ws.Ignore(ws.inversion_iteration_counter)

            ws.xClip(ijq=0, limit_low=0.00000000001, limit_high=0.00002)

            # Map x to ARTS' variables
            ws.x2artsAtmAndSurf()
            ws.x2artsSensor()

            ws.atmfields_checkedCalc(negative_vmr_ok=True)
            ws.atmgeom_checkedCalc()

            # Calculate yf and Jacobian matching x
            ws.yCalc(y=ws.yf)  # (y=ws.yf)

            # Add baseline term
            ws.VectorAddElementwise(ws.yf, ws.y, ws.y_baseline)

        self.arts.OEM(
            method="lm",
            stop_dx=self.config.stop_dx,
            lm_ga_settings=self.config.lm_ga_settings,
            display_progress=1,
        )


class RetrievalOEMInit:
    """
    TODO: This is very much not a finished product. Have to oversee all retrieval species and such
    """

    def __init__(self, obj: Retrieval):
        self.retobj = obj
        self.retconf = obj.config
        self.p_ret = obj.data.p_ret.values
        self.lat = obj.arts.lat_grid
        self.lon = obj.arts.lon_grid

    def set_ret_quantities(self):
        self.retobj.arts.retrievalDefInit()

        self.add_species(species="O3")
        self.add_polyfit()
        self.add_frequency_shift()
        self.add_Se()

        self.retobj.arts.retrievalDefClose()

    def add_species(self, species):
        # will change
        match species:
            case "O3":
                vec = np.full_like(self.retobj.data.p_ret.values, 0.5)
                covmat = np.diag(vec)
                spec = str(self.retobj.arts.abs_species.value[0])
                print(type(spec))
        self.retobj.arts.retrievalAddAbsSpecies(
            g1=self.p_ret,
            g2=np.array([0]),
            g3=np.array([0]),
            species=spec,
        )
        self.retobj.arts.covmat_sxAddBlock(block=covmat)

    def add_polyfit(self):
        self.retobj.arts.retrievalAddPolyfit(poly_order=self.retconf.poly_order)

        for cov in self.retconf.poly_covs:
            self.retobj.arts.covmat_sxAddBlock(block=cov)

    def add_frequency_shift(self):
        self.retobj.arts.retrievalAddFreqShift(df=self.retconf.fshift_df)
        self.retobj.arts.covmat_sxAddBlock(block=self.retconf.fshift_cov)

    def add_Se(self):
        vec = np.full_like(self.retobj.data.fb.values, 0.5)
        sparse_block = pyarts.arts.Sparse()
        self.retobj.arts.DiagonalMatrix(sparse_block, vec)
        self.retobj.arts.covmat_seAddBlock(block=sparse_block)

    def define_outputs(self):
        self.retobj.arts.x = np.array([])
        self.retobj.arts.yf = np.array([])
        self.retobj.arts.jacobian = np.array([[]])
        self.retobj.arts.xaStandard()


# move later
class AtmosphereAndRT:
    def __init__(self, obj: Retrieval):
        self.retobj = obj
        self.retconf = obj.config

    def get_arts_paths(self):
        arts_paths = find_arts_paths(self.retobj.attrs)
        self.lines_path = arts_paths.lines
        self.cia_path = arts_paths.cia
        self.atm_base_path = arts_paths.atmosphere_base

    def set_absorption(self):
        fs = self.retconf.f_start
        fe = self.retconf.f_end
        exclude = ["CIA", "PWR"]
        abs_species = []
        for s in self.retconf.abs_species:
            if exclude[0] in s or exclude[1] in s:
                abs_species.append(s)
            else:
                fill = s + f"-*-{fs - 1e9}-{fe + 1e9}"
                abs_species.append(fill)
        self.retobj.arts.abs_speciesSet(species=np.array(abs_species))
        self.retobj.arts.abs_lines_per_speciesReadSpeciesSplitCatalog(basename=self.lines_path)
        self.retobj.arts.abs_cia_dataReadSpeciesSplitCatalog(basename=self.cia_path)

    def set_radiative_transfer(self):
        self.retobj.arts.jacobianOff()
        self.retobj.arts.cloudboxOff()
        self.retobj.arts.stokes_dim = self.retconf.stokes_dim

        @pyarts.workspace.arts_agenda(ws=self.retobj.arts, set_agenda=True)
        def gas_scattering_agenda(ws):
            ws.Ignore(ws.rtp_vmr)
            ws.gas_scattering_coefAirSimple()
            ws.gas_scattering_matRayleigh()

        self.retobj.arts.iy_unit = self.retconf.iy_unit
        self.retobj.arts.ppath_lmax = self.retconf.ppath_lmax
        self.retobj.arts.propmat_clearsky_agendaAuto()

    def set_atmosphere(self):
        self.retobj.arts.AtmosphereSet1D()
        self.retobj.arts.PlanetSet(option="Earth")
        self.retobj.arts.nlteOff()
        self.retobj.arts.AtmRawRead(basename=self.atm_base_path)

        z = altitude_from_pressure(self.retobj.data.p.values, self.retobj.data.temperature.values)
        self.z_ret = interp_from_ecmwf_to_pret(data=self.retobj.data, product=z)
        self.t_ret = interp_from_ecmwf_to_pret(data=self.retobj.data, product="temperature")
        self.h2o_vmr = interp_from_ecmwf_to_pret(data=self.retobj.data, product="h2o")
        apriori = interp_from_ecmwf_to_pret(
            data=self.retobj.data,
            product=self.retobj.data.o3.values,
        )

        z_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.z_ret,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        t_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.t_ret,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        h2o_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                self.h2o_vmr,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        o3_gf = GriddedField3.from_xarray(
            make_ds_for_arts(
                apriori,
                self.retobj.data.p_ret.values,
                config=self.retconf,
            )
        )

        self.retobj.arts.t_field_raw = t_gf
        self.retobj.arts.z_field_raw = z_gf
        self.retobj.arts.vmr_field_raw.value[0] = o3_gf
        self.retobj.arts.vmr_field_raw.value[1] = h2o_gf
        self.retobj.arts.lat_true = [self.retconf.lat]
        self.retobj.arts.lon_true = [self.retconf.lon]
        self.retobj.arts.z_surface = [[self.z_ret[0]]]
        self.retobj.arts.AtmFieldsCalc()

    def set_los(self):
        self.retobj.arts.sensor_pos = [[self.z_ret[0] + 20]]
        self.retobj.arts.sensor_los = [[self.retobj.attrs.zenith]]
        self.retobj.arts.AntennaOff()

        @pyarts.workspace.arts_agenda(ws=self.retobj.arts)
        def sensor_response_agenda(ws):
            ws.AntennaOff()
            ws.FlagOn(ws.sensor_norm)
            ws.sensor_responseInit()
            ws.backend_channel_responseGaussianConstant(fwhm=self.retconf.f_res)
            ws.sensor_responseBackend()

        self.retobj.arts.Copy(self.retobj.arts.sensor_response_agenda, sensor_response_agenda)
        self.retobj.arts.AgendaExecute(self.retobj.arts.sensor_response_agenda)

    def execute(self):
        self.get_arts_paths()
        self.set_absorption()
        self.set_radiative_transfer()
        self.set_atmosphere()
        self.set_los()
