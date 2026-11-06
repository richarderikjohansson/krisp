from pyarts.arts import GriddedField3, Tensor3, Vector
import numpy as np
from typhon.physics import pressure2height


def get_z(pressure, temperature):
    z = pressure2height(pressure, temperature)
    return z


def make_sx_o3(p):
    w_min = 0.1
    p0 = 100.0
    k = 4.0
    weights = w_min + (1 - w_min) / \
        (1 + np.exp(-k * (np.log10(p0) - np.log10(p))))
    return weights


def set_griddedfield3(pressure, latitude, longitude, data):
    p = Vector(pressure)
    gf = GriddedField3()
    gf.set_grid_name(0, "Pressure")
    gf.set_grid_name(1, "Latitude")
    gf.set_grid_name(2, "Longitude")
    gf.set_grid(0, p)
    gf.set_grid(1, [latitude])
    gf.set_grid(2, [longitude])

    data = Tensor3(data[:, np.newaxis, np.newaxis])
    gf.data = data
    return gf


def make_se_diag(f, y):
    nu0 = f[np.argmax(y)]
    n = len(f)
    i = np.arange(n)

    # sigma min/max
    sigma_min = np.std(y[0:100])
    sigma_max = 10 * sigma_min

    # quadratic
    x = np.abs(f - nu0) / np.max(np.abs(f - nu0))
    quadratic = sigma_min**2 + (sigma_max**2 - sigma_min**2) * x**2

    # gaussian
    center = np.argmax(y)
    d = np.maximum(0, np.abs(i - center) - 50)
    w = 1300
    gaussian = sigma_max**2 - (sigma_max**2 - sigma_min**2) * np.exp(
        -(d**2) / (2 * w**2)
    )

    return quadratic, gaussian, nu0
