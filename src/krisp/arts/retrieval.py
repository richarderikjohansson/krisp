import pyarts
from datetime import datetime
from pathlib import Path
import numpy as np

from krisp.data.readers import ConfigReader, DataReader
from krisp.filesystem.paths import find_default_configs
from krisp.arts.agendas import default_agendas
from krisp.arts.atmosphere import AtmosphereAndRT
from krisp.arts.oem_setup import RetrievalOEMInit


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
        self.data, self.attrs = DataReader.load(self.fp, attrs=True, group="measurement")
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
