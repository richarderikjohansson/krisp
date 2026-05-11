import pyarts
from datetime import datetime
from pathlib import Path
import numpy as np

from importlib.resources import files

from krisp.data.readers import ConfigReader, DataReader, read_apriori_file, xr
from krisp.filesystem.paths import find_default_configs
from krisp.arts.agendas import default_agendas
from krisp.arts.atmosphere import RTandAtmosphereRetrieval
from krisp.arts.oem_setup import RetrievalOEMInit
from krisp.data.classes import Configuration
from krisp.filesystem.paths import find_apriori
from krisp.physics.atmosphere import pressure2height


class Retrieval:
    def __init__(self, file: Path, verbosity: int = 1):
        self.start_of_retrieval = datetime.now().timestamp()
        self.arts = pyarts.workspace.Workspace(verbosity=verbosity)
        self.fp = file

    def set_defaults(self):
        self.set_data_and_config(config="default")
        self.set_default_agendas()
        self.set_retrieval_grids()
        self.set_atmosphere()
        self.perform_checks()

    def set_data_and_config(self, config: Path | str = "default"):
        self.data, self.attrs = DataReader.load(self.fp, group="measurement")
        if config == "default":
            config = find_default_configs(self.attrs)
        self.config = ConfigReader.load(config)

    def set_default_agendas(self):
        self.arts = default_agendas(self.arts)

    def set_retrieval_grids(self):
        """
        Method setting retrieval grids

        """

        f_clip = self.config.f_clip
        fs, fe = self.config.f_start, self.config.f_end
        fmask = (fs <= self.data.fb.values) & (self.data.fb.values <= fe)
        self.arts.y = self.data.y.values[fmask][f_clip:-f_clip]
        self.arts.f_backend = self.data.fb.values[fmask][f_clip:-f_clip]
        self.arts.f_grid = np.arange(fs, fe, step=self.config.f_res)

        apriori_file = find_apriori(self.config)
        apriori = read_apriori_file(apriori_file)
        id = self.attrs.id.values
        dt = datetime.fromtimestamp(id)
        month = str(dt.month)
        data = apriori[month]
        apriori = data["o3a"]
        pret = data["pret"]
        zret = pressure2height(pret)

        self.data["apriori"] = xr.DataArray(data=apriori, dims="pret")
        self.data["zret"] = xr.DataArray(data=zret, dims="pret")
        self.data["pret"] = xr.DataArray(data=pret, dims="pret")
        self.arts.p_grid = pret

    def perform_checks(self):

        self.arts.propmat_clearsky_agenda_checkedCalc()
        self.arts.atmfields_checkedCalc()
        self.arts.atmgeom_checkedCalc()
        self.arts.cloudbox_checkedCalc()
        self.arts.lbl_checkedCalc()
        self.arts.sensor_checkedCalc()

    def set_atmosphere(self):
        atm = RTandAtmosphereRetrieval(obj=self)
        atm.execute()

    def init_OEM(self):
        ret = RetrievalOEMInit(obj=self)
        ret.set_ret_quantities()
        ret.define_outputs()

    def run_OEM(self):

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
