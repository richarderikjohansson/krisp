from krisp.data.readers import DataReader
from krisp.data.readers import ConfigReader
from krisp.data.classes import Attributes
from xarray import Dataset
from pathlib import Path


def test_data_reader_measurement(attrs=True):
    fp = Path(__file__).parent
    ddir = fp / "data"
    files = [f for f in ddir.rglob("*.h5")]

    res = DataReader.load(file_path=files[0], attrs=attrs, group="measurement")

    assert type(res[0]) is Dataset and type(res[1]) is Attributes
