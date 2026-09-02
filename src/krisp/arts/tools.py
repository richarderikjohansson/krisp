import xarray as xr
from pyarts import Workspace


def opacity_from_arts(data: xr.Dataset) -> float:
    pass

def make_sx_o3(p):
    w_min = 0.1
    p0 = 100.0
    k = 4.0
    weights = w_min + (1 - w_min) / (1 + np.exp(-k * (np.log10(p0) - np.log10(p))))
    return weights

