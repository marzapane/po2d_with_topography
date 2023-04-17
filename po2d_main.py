import numpy as np

from po2d_config import *
from po2d import *

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
        psi = inv_laplacian2d(q)
        J = arakawa_jacobian(q, psi)
        time_iter(q, psi, J, random_forcing, t0)
    else:
        q = psi = J = np.zeros((N, N))
        time_iter(q, psi, J, random_forcing)

if __name__ == "__main__":
    main()