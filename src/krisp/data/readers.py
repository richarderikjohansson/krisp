import h5py
from h5py import Dataset
import xarray as xr
from pathlib import Path, PosixPath
from numpy.typing import NDArray
from krisp.data.classes import Attributes, Configuration, GroupNotFoundError
from typing import Tuple, Any, Dict
from datetime import datetime, timedelta
import tomllib


class ConfigReader:
    """
    Configuration reader class
    """

    @classmethod
    def load(cls, path: PosixPath | Path) -> Configuration:
        """
        Loader method for ConfigReader. Loads a config from a path


        Parameters
        ----------
        path
            Path to the configuration file

        Returns
        -------
        Configuration
            The configuration as a Configuration class

        Raises
        ------
        FileNotFoundError:
            Raises if the configuration file cannot be found
        """

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        with path.open("rb") as f:
            config: Dict[str, Any] = tomllib.load(f)
            return Configuration(data=config)


class DataReader:
    """
    Data reader class
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
        """
        Loader method for the DataReader


        Parameters
        ----------
        file_path
            Path to .h5 file with data
        attrs
            Boolean flag whether attributes should be returned
        group
            Group in .h5 file

        Returns
        -------
        xr.Dataset | Tuple[xr.Dataset, Attributes]
            Data in as a xarray Dataset and optionally also the attributes with type Attributes

        Raises
        ------
        FileNotFoundError:
            Is raised if the file with data can not be found
        GroupNotFoundError:
            Is raised if the group in .h5 file can not be found
        """
        path: Path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")

        with h5py.File(name=path, mode="r") as fh:
            grp_keys = fh.keys()
            if group not in grp_keys:
                raise GroupNotFoundError(f"{group}" for found in {file_path})
            group: h5py.Dataset = fh[group]
            return cls._group_to_dataset(group, attrs)

    @staticmethod
    def _group_to_dataset(
        group: h5py.Group,
        attrs: bool,
    ) -> xr.Dataset | Tuple[xr.Dataset, Attributes]:
        data_vars: Dict[str, Any] = {}

        for name, obj in group.items():
            if not isinstance(obj, h5py.Dataset):
                continue

            data_vars[name] = DataReader._wrap(name, obj)
        if attrs:
            attrs_dict: Dict = group.attrs

            return (
                xr.Dataset(data_vars, attrs=dict(group.attrs)),
                Attributes(data=attrs_dict),
            )
        else:
            return xr.Dataset(data_vars, attrs=dict(group.attrs))

    @staticmethod
    def _wrap(name: str, ds: h5py.Dataset) -> xr.DataArray:
        data: h5py.Dataset = ds[()]
        dims: Tuple[str, ...] = DataReader._infer_dims(name, data)

        return xr.DataArray(
            data,
            dims=dims,
            attrs=dict(ds.attrs),
        )

    @staticmethod
    def _infer_dims(name: str, data: NDArray) -> Tuple[str, ...]:
        dim_map: Dict[str, Tuple] = {
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
        half_meas: timedelta = (end - start) / 2
        return (start + half_meas).replace(microsecond=0)
