import pyarts
import xarray as xr


from krisp.data.classes import Configuration
from krisp.arts.agendas import default_agendas


class Simulation:
    def __init__(self, data: xr.Dataset, config: Configuration, verbosity: bool = False):
        self.arts = pyarts.Workspace()
        self.data = data
        self.config = config

    def set_default_agendas(self):
        self.arts = default_agendas(self.arts)
