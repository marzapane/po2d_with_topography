#!/usr/bin/env python

import po2d_classes as po

def main():
    analize_vortex = False
    simul = po.FluidSimulator(random_forcing, analize_vortex)
    fluid = po.FluidState(simul.reload_bak, simul.bak_dir, simul.bak_file)
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)
    simul.conclude()

if __name__ == "__main__":
    main()