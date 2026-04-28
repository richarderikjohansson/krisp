from typing import Dict, List
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
from typing import Any


class BaseDataClass:
    """Base data class"""

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def to_dict(self):
        return dict(self.__dict__)


class GroupNotFoundError(Exception):
    """
    Custom error for when group not found in .h5 file
    """

    pass


class Configuration(BaseDataClass):
    """
    Configuration data class associated with content in .toml configuration files
    """

    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, List):
                value: NDArray = np.array(object=value)
            setattr(self, key, value)


class Attributes(BaseDataClass):
    """
    Attributes data class associated with attributes in .h5 files

    """

    mode: str
    start: datetime
    middle: datetime
    end: datetime

    def __init__(self, data: Dict):

        for key, value in data.items():
            if key in ("start", "end", "middle"):
                value: datetime = datetime.fromtimestamp(timestamp=value)
            setattr(self, key, value)
