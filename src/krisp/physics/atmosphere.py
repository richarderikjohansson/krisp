from typhon.physics import pressure2height
from numpy.typing import NDArray


def altitude_from_pressure(pressure: NDArray, temperature: NDArray = None) -> NDArray:
    """Function to calculate altitude from pressure and temperature

    This function can also calculate the altitude without the pressure profile
    and will do this if the shape of pressure and temperature arrays differ or
    if only the pressure array is given.

    :param pressure: pressure array
    :param temperature: temperature array
    :return: altitude in m
    """
    flag = pressure.shape == temperature.shape
    if temperature is not None and not flag:
        temperature = None

    altitude = pressure2height(pressure, temperature)
    return altitude
