import pyarts
from krisp.data.readers import Configuration, Attributes
import xarray as xr
from typing import Tuple


class Retrieval:
    def __init__(
        self,
        data: Tuple[xr.Dataset, Attributes],
        config: Configuration | None = None,
    ):
        pass
