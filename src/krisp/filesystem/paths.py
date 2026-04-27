from pathlib import Path
from pathlib import PosixPath
import pyarts
from krisp._const import PYARTS_VERSION
from krisp.filesystem._const import ARTS_SUMMER_ATM
from krisp.filesystem._const import ARTS_WINTER_ATM
from krisp.filesystem._const import ARTS_LINES
from krisp.filesystem._const import ARTS_CIA
from krisp.data.readers import Attributes
from typing import Dict
from importlib.resources import files


class Paths:
    def __init__(self, paths):
        for key, value in paths.items():
            setattr(self, key, value)


def find_arts_paths(month: int) -> Dict:
    """Function to set ARTS related paths

    :param month: month of measurement
    :return: ARTS related paths
    """
    home = Path.home()
    artsdir = home / ".cache/arts"

    xml_path = artsdir / f"arts-xml-data-{PYARTS_VERSION}"
    cat_path = artsdir / f"arts-cat-data-{PYARTS_VERSION}"

    if not xml_path.exists() or cat_path.exists():
        pyarts.cat.download.retrieve()

    if month in range(5, 10):
        atmos = str(xml_path / ARTS_SUMMER_ATM)
    else:
        atmos = str(xml_path / ARTS_WINTER_ATM)

    out = {
        "atmosphere_base": atmos,
        "lines": str(cat_path / ARTS_LINES) + "/",
        "cia": str(cat_path / ARTS_CIA) + "/",
    }
    return Paths(paths=out)


def find_default_configs(attrs: Attributes) -> Paths:
    """Function to set the path to the default config file based on the mode

    :param attrs: measurement attributes
    :return: path to config file
    """
    configs_dir = files("krisp.arts").joinpath("configs")
    for config in configs_dir.iterdir():
        name = config.name
        if attrs.mode in name:
            return Paths(paths={"path": config})
