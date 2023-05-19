#!/usr/bin/env python

import numpy as np
import po2d_classes as po

def gauss_topography(
    Dx: float,  # grid spacing
    n: int,     # grid size
    sigma = 1,  # mount width
):
    ctr = int(n-1)/2 * Dx
    dist, _ = po.relative_pos(ctr, ctr, Dx, n)
    return np.exp(-dist**2 / (2 * sigma**2))

def random_topography(
    peaks: int, # number of peaks
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    topography = np.zeros((n, n))
    for i in range(peaks):
        ctr = 2*np.pi * np.random.rand(2)
        fatness = 2 * np.random.rand()
        height = np.random.rand()
        sign = [-1, 1][np.random.randint(2)]
        dist, _ = po.relative_pos(*ctr, Dx, n)
        topography += sign * height * np.exp(-(dist/fatness)**2)
    max_heigth = np.max(topography)
    return topography/max_heigth

def main():
    analize_vortex = False
    # simul = po.FluidSimulator(po.random_forcing, analize_vortex)
    simul = po.FluidSimulator(po.zero_forcing, analize_vortex)
    simul.set_diagnostics()
    topography = gauss_topography(simul.Dx, simul.N)
    # topography = random_topography(20, simul.Dx, simul.N)
    fluid = po.FluidStateTopography(topography, simul.reload_bak, simul.bak_dir, simul.bak_file)
    simul.set_physical_param(fluid)
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)
    simul.conclude()

if __name__ == "__main__":
    main()