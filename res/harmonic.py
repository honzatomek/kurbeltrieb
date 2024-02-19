#!/usr/bin/python3

import os
import sys

import numpy as np
import scipy
import matplotlib.pyplot as plt


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



def test():
    fig = plt.figure()
    ax1 = fig.add_subplot(511)
    ax2 = fig.add_subplot(512)
    ax3 = fig.add_subplot(513)
    ax4 = fig.add_subplot(514)
    ax5 = fig.add_subplot(515)

    dt = 0.01

    a1 = 10.
    f1 = 1.
    p1 = np.pi / 3
    o1 = 3.

    a2 = 3.
    f2 = 5.
    p2 = np.pi / 17
    o2 = -1.

    t = np.arange(0., 3., dt)

    sin1 = a1 * np.sin(2 * np.pi * f1 * t + p1) + o1
    sin2 = a2 * np.sin(2 * np.pi * f2 * t + p2) + o2

    ln1 = ax1.plot(t, sin1, label=f"{f1:.2f} Hz")
    ln2 = ax1.plot(t, sin2, label=f"{f2:.2f} Hz")

    signal = sin1 + sin2 + np.random.rand(t.shape[0])

    harmonic_decomp(t, signal)

    ln3 = ax2.plot(t, signal, label=f"{f1:.2f} + {f2:.2f} Hz")

    fft = scipy.fft.fft(signal)
    freq = scipy.fft.fftfreq(t.shape[0], dt)

    # ax3.plot(freq[:freq.shape[0] // 2], np.abs(fft[:freq.shape[0] // 2]))
    ax3.plot(freq[:freq.shape[0] // 2], np.abs(fft[:freq.shape[0] // 2]))


    sinesum = np.zeros(t.shape[0])
    sines = [(1 / (t.shape[0])) * (fft[0].real * np.cos(2 * np.pi * freq[0] * t) - fft[0].imag * np.sin(2 * np.pi * freq[0] * t))]
    sinesum += sines[-1]
    ax4.plot(t, sines[-1])

    for i in range(1, freq.shape[0] // 2 + 1):
        sines.append((2 / (t.shape[0])) * (fft[i].real * np.cos(2 * np.pi * freq[i] * t) - fft[i].imag * np.sin(2 * np.pi * freq[i] * t)))
        ax4.plot(t, sines[-1])
        sinesum += sines[-1]

    recombine = np.zeros(t.shape[0])
    for f, A, B in harmonic_decomp(t, signal):
        recombine += A * np.cos(2 * np.pi * f * t) + B * np.sin(2 * np.pi * f * t)

    ax5.plot(t, signal, label="signal")
    ax5.plot(t, sinesum, label="sinesum")
    ax5.plot(t, recombine, label="recombine")
    ax5.legend()

    plt.show()


if __name__ == "__main__":
    test()


