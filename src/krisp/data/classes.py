from typing import Dict, List
from datetime import datetime
import numpy as np


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


class Attributes(BaseDataClass):
    """Utility class to dynamically set attribute data from a dictionary"""

    def __init__(self, data: Dict):
        for key, value in data.items():
            if key == "start" or key == "end" or key == "middle":
                value = datetime.fromtimestamp(value)
            setattr(self, key, value)


class Configuration(BaseDataClass):
    """Utility class to dynamically set configuration data from a dictionary"""

    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, List):
                value = np.array(value)
            setattr(self, key, value)
