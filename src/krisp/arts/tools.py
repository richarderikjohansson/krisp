import xarray as xr
from pyarts.arts import GriddedField3, Tensor3
import numpy as np


def opacity_from_arts(data: xr.Dataset) -> float:
    pass


def make_sx_o3(p):
    w_min = 0.1
    p0 = 100.0
    k = 4.0
    weights = w_min + (1 - w_min) / (1 + np.exp(-k * (np.log10(p0) - np.log10(p))))
    return weights


def set_griddedfield3(pressure, latitude, longitude, data):
    gf = GriddedField3()
    gf.set_grid_name(0, "Pressure")
    gf.set_grid_name(1, "Latitude")
    gf.set_grid_name(2, "Longitude")
    gf.set_grid(0, pressure)
    gf.set_grid(1, [latitude])
    gf.set_grid(2, [longitude])

    data = Tensor3(data[:, np.newaxis, np.newaxis])
    gf.data = data
    return gf
