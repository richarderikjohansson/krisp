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
    Data reader class for reading .h5 files into xarray Datasets.
    Cannot be instantiated — use DataReader.load(path) instead.
    """

    def __init__(self, *args, **kwargs):
        raise TypeError("DataReader cannot be instantiated. Use DataReader.load(path) instead.")

    @classmethod
    def load(
        cls,
        file: str | Path,
        group: str = "measurement",
    ) -> Tuple[xr.Dataset, xr.Dataset]:
        """
        Load data and provenance groups from an .h5 file.

        Parameters
        ----------
        file_path : str | Path
            Path to the .h5 file.
        group : str
            Measurement group name. Defaults to "measurement".

        Returns
        -------
        Tuple[xr.Dataset, xr.Dataset]
            Measurement data as the first element, provenance data as the second.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        GroupNotFoundError
            If either group is not found in the file.
        """
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")

        with h5py.File(path, "r") as fh:
            for grp_name in (group, "provenance"):
                if grp_name not in fh:
                    raise GroupNotFoundError(f"Group '{grp_name}' not found in {path}.")

            data = cls._group_to_dataset(fh[group], read_dims=True)
            provenance = cls._group_to_dataset(fh["provenance"], read_dims=False)

        return data, provenance

    @classmethod
    def _group_to_dataset(cls, grp: h5py.Group, read_dims) -> xr.Dataset:
        data_vars = {}
        for name, obj in grp.items():
            if not isinstance(obj, h5py.Dataset):
                continue
            data = obj[()]
            if isinstance(data, bytes):
                data = data.decode()

            dims = cls._read_dims(obj, data.ndim) if read_dims else ()
            data_vars[name] = xr.DataArray(data, dims=dims, attrs=dict(obj.attrs))
        return xr.Dataset(data_vars, attrs=dict(grp.attrs))

    @staticmethod
    def _read_dims(ds: h5py.Dataset, ndim: int) -> Tuple[str, ...]:
        if "dims" not in ds.attrs:
            return tuple(f"dim_{i}" for i in range(ndim))
        raw = ds.attrs["dims"]
        if isinstance(raw, (bytes, str)):
            return (raw.decode() if isinstance(raw, bytes) else raw,)
        return tuple(d.decode() if isinstance(d, bytes) else str(d) for d in raw)


def read_apriori_file(path: Path) -> Dict:
    with h5py.File(path, "r") as fh:
        dct = {}
        for k, v in fh.items():
            dct[k] = {n: d[()] for n, d in v.items()}
    return dct
