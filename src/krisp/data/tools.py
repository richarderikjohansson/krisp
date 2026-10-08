from krisp.data.classes import Configuration
from xarray import Dataset
import xarray as xr
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from pathlib import Path
from h5py import File
from rich.progress import track
from krisp.arts.logger import get_logger


def interp_from_ecmwf_to_pret(data: Dataset, product: str | NDArray) -> NDArray:
    if isinstance(product, np.ndarray):
        p_src = data["p"].values
        prod_src = product
        p_tgt = data["pret"].values
        interp = PchipInterpolator(np.log(p_src[::-1]), prod_src[::-1])
        prod_tgt = interp(np.log(p_tgt[::-1]))
        return prod_tgt[::-1]

    assert hasattr(data, product)
    p_src = data["p"].values
    prod_src = data[product].values
    p_tgt = data["pret"].values

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


def nc2h5(root: Path | str, fn: Path | str):

    # get .nc files and set outpath
    logger = get_logger()
    root = Path(root)
    it = root.rglob("*.nc")
    files = [f for f in it]
    home = Path.home()
    downloads_dir = home / "Downloads"
    if not downloads_dir.exists():
        downloads_dir = home
    outpath = downloads_dir / fn

    # save data from .nc files in one h5 file
    with File(outpath, "w") as h5:
        for file in track(files):
            ds = xr.load_dataset(file)
            ts = str(ds.attrs["timestamp"])
            grp = h5.create_group(name=ts)

            for name, data in ds.data_vars.items():
                grp.create_dataset(name=name, data=data)
            for name, data in ds.coords.items():
                grp.create_dataset(name=name, data=data)
            for name, data in ds.attrs.items():
                grp.create_dataset(name=name, data=data)

            grp.create_dataset(name="src", data=str(file))

    logger.info(f"Saved data from {root} in {outpath}")
