from scipy.interpolate import interp1d
from krisp.data.classes import Attributes, Configuration
from xarray import Dataset
import xarray as xr
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


def interp_from_ecmwf_to_pret(data: Dataset, product: str | NDArray) -> NDArray:
    if isinstance(product, np.ndarray):
        p_src = data["p"].values
        prod_src = product
        p_tgt = data["p_ret"].values
        interp = PchipInterpolator(np.log(p_src[::-1]), prod_src[::-1])
        prod_tgt = interp(np.log(p_tgt[::-1]))
        return prod_tgt[::-1]

    assert hasattr(data, product)
    p_src = data["p"].values
    prod_src = data[product].values
    p_tgt = data["p_ret"].values

    interp = PchipInterpolator(np.log(p_src[::-1]), prod_src[::-1])
    prod_tgt = interp(np.log(p_tgt[::-1]))
    return prod_tgt[::-1]


def make_ds_for_arts(product: NDArray, pressure: NDArray, config: Configuration):
    name = ["Pressure", "Latitude", "Longitude"]
    data = product[:, np.newaxis, np.newaxis]
    lat = np.array([config.lat])
    lon = np.array([config.lon])
    da = xr.DataArray(data, coords=[pressure, lat, lon], dims=name, name=name)
    return da
