#!/usr/bin/env python

import numpy as np
import po2d_classes as po


def main():
    analize_vortex = False
    simul = po.FluidSimulator(po.random_forcing, analize_vortex, adaptive_Dt=True)
    # simul = po.FluidSimulator(po.zero_forcing, analize_vortex)
    simul.set_diagnostics()
    # topography = square_wells_topography(simul.Dx, simul.N)
    topography = gauss_topography(simul.Dx, simul.N)
    # topography = random_topography(30, simul.Dx, simul.N)
    fluid = po.FluidStateTopography(topography, simul.reload_bak, simul.bak_dir, simul.bak_file)
    print(f'LDA measure of topography: {po.measure_valley_dist(fluid.topography - fluid.topography.max(), fluid.N)}')
    simul.set_physical_param(fluid)
    q_avg = fluid.q
    count = 1
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)
        q_avg += fluid.q
        count += 1
    simul.conclude()
    np.save(simul.bak_dir / f'q_avg', q_avg/count)

if __name__ == "__main__":
    main()