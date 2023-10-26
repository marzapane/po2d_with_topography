#!/usr/bin/env python

import numpy as np
import po2d_classes as po

def gauss_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = None,  # mount width
):
    ctr = int(n-1)/2 * Dx
    if sigma is None:
        sigma = 2*np.pi / 10
    dist, _ = po.relative_pos(ctr, ctr, Dx, n)
    return np.exp(-dist**2 / (2 * sigma**2))

def diag_ridge_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = None,  # mount width
):
    sd_len = n * Dx
    pos = np.arange(n) * Dx
    xy_diff = np.abs(pos[:, None] - pos[None, :])
    dist2diag = np.where(xy_diff < sd_len/2, xy_diff, sd_len - xy_diff) / np.sqrt(2)
    if sigma is None:
        sigma = 2*np.pi / 10
    return np.exp(-dist2diag**2 / (2 * sigma**2))

def random_topography(
    peaks: int, # number of peaks
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    topography = np.zeros((n, n))
    for i in range(peaks):
        ctr = 2*np.pi * np.random.rand(2)
        height = np.random.rand()
        fatness = 2 * height * (1.5 * np.random.rand() + 0.25)
        sign = [-1, 1][np.random.randint(2)]
        dist, _ = po.relative_pos(*ctr, Dx, n)
        topography += sign * height * np.exp(-(dist/fatness)**2)
    max_heigth = np.max(topography)
    return topography/max_heigth

def square_wells_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = None,  # mount width
):
    ctr = np.array(((np.pi/2, np.pi/2), (np.pi/2, 3*np.pi/2), (3*np.pi/2, np.pi/2), (3*np.pi/2, 3*np.pi/2)))
    if sigma is None:
        sigma = 2*np.pi / 5
    sign = (+1, -1, -1, +1)
    topography = np.zeros((n, n))
    for i in range(4):
        dist, _ = po.relative_pos(*ctr[i], Dx, n)
        topography += sign[i] * np.exp(-dist**2 / (2*sigma**2))
    max_heigth = np.max(topography)
    return topography/max_heigth

def main():
    analize_vortex = False
    simul = po.FluidSimulator(po.random_forcing, analize_vortex)
    # simul = po.FluidSimulator(po.zero_forcing, analize_vortex)
    simul.set_diagnostics()
    # topography = square_wells_topography(simul.Dx, simul.N)
    # topography = gauss_topography(simul.Dx, simul.N)
    topography = random_topography(30, simul.Dx, simul.N)
    fluid = po.FluidStateTopography(topography, simul.reload_bak, simul.bak_dir, simul.bak_file)
    print(f'LDA measure of topography: {po.measure_valley_dist(fluid.topography - fluid.topography.max(), fluid.N)}')
    simul.set_physical_param(fluid)
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)
    simul.conclude()

if __name__ == "__main__":
    main()