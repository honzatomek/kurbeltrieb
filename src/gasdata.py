import os
import sys
from typing import Union

import numpy as np
import scipy

import matplotlib.pyplot as plt


def read_gasdata(filename: str = "gasdata.asc", mina: float = -np.inf, maxa: float = np.inf) -> np.ndarray:
    angle = []
    gpres = []

    try:
        gdfile = open(filename, "rt")
        for line in gdfile:
            a, p = [float(v) for v in line.strip().split()]
            if a > maxa:
                break
            elif a < mina:
                continue
            else:
                angle.append(a)
                gpres.append(p)

    except IOError as e:
        gdfile.close()

    finally:
        gdfile.close()

    angle = np.array(angle, dtype=float)
    gpres = np.array(gpres, dtype=float)

    gasdata = np.vstack([angle, gpres]).T

    return gasdata


def plot_gasdata(gasdata: Union[np.ndarray, str] = "gasdata.asc"):
    if isinstance(gasdata, str):
        gasdata = read_gasdata(gasdata)

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.plot(gasdata[:,0], gasdata[:,1], color="red", label="$p_{zyl}$")
    ax.legend()

    plt.show()


def create_gasdata(maxp: float = 40.):
    #                           deg   bar
    control_points = np.array([[ 0.,  35.],
                               [ 10., 40.],
                               [ 80., 20.],
                               [180., 10.],
                               [220., 10.],
                               [320., 20.]], dtype=float) * (maxp / 40.)

    offset = np.array([360., 0.], dtype=float)
    periodic = np.vstack([control_points - offset, control_points, control_points + offset])

    f = scipy.interpolate.interp1d(periodic[:,0], periodic[:,1], kind="cubic")

    angle = np.arange(0., 360., 1.)
    gpres  = f(angle)

    # convert to MPa
    gpres *= 0.1

    # add a bit of randomness
    rnd = (np.random.rand(gpres.shape[0]) * 2 - 1) * (0.005 * maxp)
    gpres += rnd

    gasdata = np.vstack([angle, gpres]).T

    return gasdata


def write_gasdata(filename: str, gasdata: np.ndarray):
    with open(filename, "wt") as gpfile:
        for a, p in gasdata:
            gpfile.write(f"{a:12.4f} {p:12.5f}\n")



if __name__ == "__main__":
    gd = create_gasdata()
    plot_gasdata(gd)
    write_gasdata("gasdata_rnd.asc", gd)

