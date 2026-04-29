from pathlib import Path
import pyarts
from krisp._const import PYARTS_VERSION
from krisp.filesystem._const import ARTS_SUMMER_ATM
from krisp.filesystem._const import ARTS_WINTER_ATM
from krisp.filesystem._const import ARTS_LINES
from krisp.filesystem._const import ARTS_CIA
from krisp.data.classes import Attributes
from importlib.resources import files
from typing import Dict


class Paths:
    """
    Paths class
    """

    def __init__(self, paths: Dict):
        for key, value in paths.items():
            setattr(self, key, value)


def find_arts_paths(attrs: Attributes) -> Paths:
    """
    Function to locate xml and catalogue directories for PyARTS

    This function mainly acts as a helper function to locate various
    paths to directories used for pyarts retrievals and simulations

    Parameters
    ----------
    month
        This parameter decides with atmospheric base that is used. Between
        October and April (inclusive) will winter atmosphere be used and between
        between May and September will summer atmosphere be used

    Returns
    -------
    Paths
        Paths to atrospheric base: "atmosphere_base, lines: "lines" and cia: "cia:
    """
    month = attrs.middle.month
    home: Path = Path.home()
    artsdir: Path = home / ".cache/arts"

    xml_path: Path = artsdir / f"arts-xml-data-{PYARTS_VERSION}"
    cat_path: Path = artsdir / f"arts-cat-data-{PYARTS_VERSION}"

    if not xml_path.exists() or cat_path.exists():
        pyarts.cat.download.retrieve()

    if month in range(5, 10):
        atmos: str = str(xml_path / ARTS_SUMMER_ATM)
    else:
        atmos = str(xml_path / ARTS_WINTER_ATM)

    out: Dict[str, str] = {
        "atmosphere_base": atmos,
        "lines": str(cat_path / ARTS_LINES) + "/",
        "cia": str(cat_path / ARTS_CIA) + "/",
    }
    return Paths(paths=out)


def find_default_configs(attrs: Attributes) -> Path | None:
    """
    Function to locate default configuration for retrievals


    Parameters
    ----------
    attrs
        Attributes from measurement data

    Returns
    -------
    Paths | None
        Path to configuration file
    """
    configs_dir = files("krisp.arts").joinpath("configs")
    for config in configs_dir.iterdir():
        name: str = config.name

        if attrs.mode in name:
            return config
