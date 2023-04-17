import numpy as np

from po2d_config import *
from po2d import *

def gauss_topography():
    ctr = int(N-1)/2 * Dx
    dist, _ = relative_pos(ctr, ctr)
    topography = np.exp(-dist**2)
    return 1 - 0.9 * topography
    
def rand_topography(
    n: int
):
    topography = np.zeros((N, N))
    for i in range(n):
        ctr = 2*np.pi * np.random.rand(2)
        fatness = np.random.rand()
        height = np.random.rand()
        dist, _ = relative_pos(*ctr)
        topography += height * np.exp(-(dist/fatness)**2)
    max_heigth = np.max(topography)
    return 1 - 0.9 * topography/max_heigth

def main():
    _, bak_folder = open_folders() # looking for past simulations backup files
    bak_file, t0 = find_highest_numbered_file(bak_folder)
    if bak_file: # asking user confirmation to reload last simulation (default)
        print(f'A previous simulation has been found: {bak_file}.')
        print('Starting a different one will overwrite it.')
        user_input = input('  Reload last simulation? [Y/n] ')
        if user_input.lower() in ['no', 'n']:
            print('Launching a new simulation and overwriting the past one.')
            reload_bak = False
        else:
            print(f'Restarting past simulation from {t0=}.')
            reload_bak = True
    else:
        print('No previous simulation was found; launching a new one.')
        reload_bak = False

    if reload_bak:
        q = np.load(bak_folder / bak_file)
        h = np.load(bak_folder / 'h.dat')
        psi = inv_laplacian_topography(q, h)
        J = arakawa_jacobian(q, psi)
        time_iter(q, psi, J, random_forcing, t0, streamfunction=inv_laplacian_topography)
    else:
        q = psi = J = np.zeros((N, N))
        h = gauss_topography(20)
        time_iter(q, psi, J, random_forcing, streamfunction=inv_laplacian_topography)

if __name__ == "__main__":
    main()