import h5py
import xarray as xr
from pathlib import Path, PosixPath
from numpy.typing import NDArray
from krisp.data.classes import Attributes, Configuration
from typing import Tuple
from datetime import datetime
import tomllib


class GroupNotFoundError(Exception):
    pass


class ConfigReader:
    """Class for parsing retrieval configuration files"""

    @classmethod
    def load(cls, path: str | PosixPath | Path) -> Configuration:

        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        with path.open("rb") as f:
            config = tomllib.load(f)
            return Configuration(data=config)


class DataReader:
    """Class for reading data from HDF5 files into xarray.

    This class only have one job, namely load and return measurement data
    from either KIMRA or MIRA2. This is done by calling the .read() method

    Example:
        data = MeasurementReader("path/to/file")                    # returning only the data as a xarray.Dataset
        data, attrs = MeasurementReader("path/to/file", attrs=True) # returning data (xarray.Dataset) and measurement attributes (MeasurementAttributes)
    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "MeasurementReader cannot be instantiated. Use MeasurementReader.read(path) instead."
        )

    @classmethod
    def load(
        cls: DataReader,
        file_path: str | Path,
        attrs: bool = False,
        group: str = "measurement",
    ) -> xr.Dataset | Tuple[xr.Dataset, Attributes]:
        """Class method to read .h5 file with measurement data

        :param cls: MeasurementReader
        :param file_path: path to the file with measurement data
        :param attrs: boolean to determine if measurement attributes also should be returned
        separatley
        :raises FileNotFoundError: is raised if the file cannot be found
        :return: either only the data in xarray.Dataset if the attrs argument is false,
        or a tuple with in the following format Tuple[xarray.Dataset, MeasurementAttributes]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        with h5py.File(path) as fh:
            grp_keys = fh.keys()
            if group not in grp_keys:
                raise GroupNotFoundError(f"{group}" for found in {file_path})
            group = fh[group]
            return cls._group_to_dataset(group, attrs)

    @staticmethod
    def _group_to_dataset(
        group: h5py.Group,
        attrs: bool,
    ) -> xr.Dataset | Tuple[xr.Dataset, Attributes]:
        data_vars = {}

        for name, obj in group.items():
            if not isinstance(obj, h5py.Dataset):
                continue

            data_vars[name] = DataReader._wrap(name, obj)
        if attrs:
            # check if this can be generalized
            attrs_dict = group.attrs

            return (
                xr.Dataset(data_vars, attrs=dict(group.attrs)),
                Attributes(attrs_dict),
            )
        else:
            return xr.Dataset(data_vars, attrs=dict(group.attrs))

    @staticmethod
    def _wrap(name: str, ds: h5py.Dataset) -> xr.DataArray:
        data = ds[()]
        dims = DataReader._infer_dims(name, data)

        return xr.DataArray(
            data,
            dims=dims,
            attrs=dict(ds.attrs),
        )

    @staticmethod
    def _infer_dims(name: str, data: NDArray) -> Tuple[str, ...]:
        dim_map = {
            "h2o": ("pressure",),
            "o3": ("pressure",),
            "temperature": ("pressure",),
            "pressure": ("pressure",),
            "frequency": ("frequency",),
            "spectra": ("frequency",),
            "apriori": ("pret",),
            "pret": ("pret",),
        }
        if name in dim_map:
            return dim_map[name]

        return tuple(f"dim_{i}" for i in range(data.ndim))

    @staticmethod
    def _get_mid_from_attrs(start: datetime, end: datetime) -> datetime:
        half_meas = (end - start) / 2
        return (start + half_meas).replace(microsecond=0)
