import h5py
import xarray as xr
from typing import Tuple
from pathlib import Path
from numpy.typing import NDArray
from krisp.filesystem.io import Path
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MeasurementAttributes:
    start: datetime
    end: datetime
    azimuth: float
    zenith: float
    integration_time: float
    integrations: int
    mode: str
    source: str
    spectrometer: str
    instrument: str


class MeasurementReader:
    """Utility class for reading measurement HDF5 files into xarray."""

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "MeasurementReader cannot be instantiated. "
            "Use MeasurementReader.read(path) instead."
        )

    @classmethod
    def read(
        cls,
        file_path: str | Path,
        attrs: bool = False,
    ) -> xr.Dataset | Tuple[xr.Dataset, MeasurementAttributes]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        with h5py.File(path) as fh:
            group = fh["measurement"]
            return cls._group_to_dataset(group, attrs)

    @staticmethod
    def _group_to_dataset(
        group: h5py.Group,
        attrs: bool,
    ) -> xr.Dataset | Tuple[xr.Dataset, MeasurementAttributes]:
        data_vars = {}

        for name, obj in group.items():
            if not isinstance(obj, h5py.Dataset):
                continue

            data_vars[name] = MeasurementReader._wrap(name, obj)
        if attrs:
            for k, v in group.attrs.items():
                match k:
                    case "azimuth":
                        aa = v
                    case "zenith":
                        za = v
                    case "start":
                        start = datetime.fromtimestamp(v)
                    case "end":
                        end = datetime.fromtimestamp(v)
                    case "integration_time":
                        int_time = v
                    case "integrations":
                        ints = v
                    case "mode":
                        mode = v
                    case "source":
                        source = v
                    case "spectrometer":
                        spectro = v
                    case "instrument":
                        instrument = v

            return (
                xr.Dataset(data_vars, attrs=dict(group.attrs)),
                MeasurementAttributes(
                    start=start,
                    end=end,
                    azimuth=aa,
                    zenith=za,
                    integration_time=int_time,
                    integrations=ints,
                    mode=mode,
                    source=source,
                    spectrometer=spectro,
                    instrument=instrument,
                ),
            )
        else:
            return xr.Dataset(data_vars, attrs=dict(group.attrs))

    @staticmethod
    def _wrap(name: str, ds: h5py.Dataset) -> xr.DataArray:
        data = ds[()]
        dims = MeasurementReader._infer_dims(name, data)

        return xr.DataArray(
            data,
            dims=dims,
            attrs=dict(ds.attrs),
        )

    @staticmethod
    def _infer_dims(name: str, data) -> Tuple[str, ...]:
        era5_group = {"h2o", "o3", "temperature", "pressure"}
        ret_group = {"pret", "apriori"}

        if name == "spectra":
            return ("frequency",)

        if name in era5_group:
            return ("pressure",)

        if name == "frequency":
            return ("frequency",)

        if name == "pressure":
            return ("pressure",)

        if name in ret_group:
            return ("pret",)

        return tuple(f"dim_{i}" for i in range(data.ndim))
