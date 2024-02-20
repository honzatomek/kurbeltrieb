#!/usr/bin/python3

import os
import sys
from typing import Union

import numpy as np
import scipy

import matplotlib.pyplot as plt


def read_gasdata(filename: str = "gasdata.asc", mina: float = -np.inf, maxa: float = np.inf) -> np.ndarray:
    angle = []
    gpres = []

    with open(filename, "rt") as gpfile:
        print(f"[+] Reading {gpfile.name:s}.")
        for line in gpfile:
            a, p = [float(v) for v in line.strip().split()]
            if a > maxa:
                break
            elif a < mina:
                continue
            else:
                angle.append(a)
                gpres.append(p)

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


def create_gasdata(minp: float = 2., maxp: float = 40.):
    #                           deg   bar
    control_points = np.array([[ 0.,  35.],
                               [ 10., 40.],
                               [ 70., 15.],
                               [120.,  5.],
                               [160.,  2.],
                               [180.,  2.],
                               [200.,  2.],
                               [240.,  4.],
                               [320., 20.]], dtype=float)

    offset = np.array([360., 0.], dtype=float)
    periodic = np.vstack([control_points - offset, control_points, control_points + offset])

    f = scipy.interpolate.interp1d(periodic[:,0], periodic[:,1], kind="cubic")

    angle = np.arange(0., 360., 1.)
    gpres  = f(angle)

    gpresmin, gpresmax = np.min(gpres), np.max(gpres)
    gpres = minp + (gpres - gpresmin) * (maxp - minp) / (gpresmax - gpresmin)

    # convert to MPa
    gpres *= 0.1

    # add a bit of randomness
    rnd = (np.random.rand(gpres.shape[0]) * 2 - 1) * (0.005 * maxp)
    gpres += rnd

    gasdata = np.vstack([angle, gpres]).T

    return gasdata


def write_gasdata(filename: str, gasdata: np.ndarray):
    with open(filename, "wt") as gpfile:
        print(f"[+] Writing {gpfile.name:s}.")
        for a, p in gasdata:
            gpfile.write(f"{a:12.4f} {p:12.5f}\n")



def harmonic_decomp(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Harmonic decomposition of a time signal to a combination of sines
    and cosines, such that:

    y(t) = Σ (A_i cos(2 π f_i t) + B_i sin(2 π f_i t))

    Args:
        t (np.ndarray): time vector
        y (np.ndarray): signal data, if the time step is not uniform, will be resampled

    Returns:
        (np.ndarray): [f, A, B]
    """
    N  = t.shape[0]
    T  = t[-1] - t[0]
    fs = 1 / T
    dt = t[1:] - t[:-1]

    # is timestep uniform?
    if np.isclose(np.min(dt), np.max(dt)):  # yes
        dt = np.mean(dt[0])
    else:                                   # no -> resample
        print(f"[i] Resampling data to uniform timestep.")
        dt = T / (N - 1)
        f = scipy.interpolate.interp1d(t, y)
        t = np.arange(t[0], t[-1], dt)
        y = f(t)

    midpoint = N // 2
    fft = scipy.fft.fft(y)[:midpoint] * (2.0 / N)
    fft[0] /= 2.
    frq = scipy.fft.fftfreq(N, dt)[:midpoint]

    res = []
    for f, A in zip(frq, fft):
        res.append([f, A.real, -A.imag])

    return np.array(res, dtype=float)


def harmonic_comp(t: np.ndarray, fft: np.ndarray):
    print(f"[+] Combining decomposed time curve")
    res = np.zeros(t.shape[0])
    for f, A, B in fft:
        res += A * np.cos(2. * np.pi * f * t) + B * np.sin(2. * np.pi * f * t)

    return res



if __name__ == "__main__":
    # gd = create_gasdata()
    # plot_gasdata(gd)
    # write_gasdata("gasdata_rnd.asc", gd)

    gd = create_gasdata()
    write_gasdata("gasdata_rnd.asc", gd)
    angle, pressure = gd[:, 0], gd[:, 1]
    rpm = 13000.
    omega = rpm / 60. * 2 * np.pi
    radians = np.radians(angle)
    time = radians / omega

    fft = harmonic_decomp(time, pressure)

    comp = harmonic_comp(time, fft[1:7,:])

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    ax.plot(time, pressure, label="orig")
    ax.plot(time,     comp, label="comp")
    ax.legend()

    plt.show()


