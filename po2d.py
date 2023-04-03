import numpy as np
import numpy.fft as ft
import numpy.linalg as la
import scipy.special as sp
import matplotlib.pyplot as plt
from functools import cache
from tqdm import tqdm
from matplotlib import cm
from matplotlib import colors
from scipy.stats import binned_statistic
from pathlib import Path
from po2d_config import *

@cache
def coordinates():
    x = np.arange(0, N) * Dx
    y = np.arange(0, N) * Dx
    return np.meshgrid(x, y, indexing='ij')

def open_folders():
    result_fold = Path().resolve().parent / 'Results'
    if not result_fold.exists():
        result_fold.mkdir()
    frames_fold = result_fold / f'frames_{cfg_name}{N}'
    if not frames_fold.exists():
        frames_fold.mkdir()
    else:
        if not reload_:
            for img in frames_fold.iterdir():
                img.unlink()
    bak_fold = result_fold / f'{cfg_name}{N}'
    if not bak_fold.exists():
        bak_fold.mkdir()
    return frames_fold, bak_fold

def find_highest_numbered_file(path):
    max_number = -1
    max_file = None
    for file in path.glob('q_*.npy'):
        try:
            file_number = int(file.stem[2:])
        except ValueError:
            # Filename doesn't match expected pattern, skip to next file
            print(f'<{filename}.npy> does not have a valid name.')
            pass
        else:
            if file_number > max_number:
                max_number = file_number
                max_file = file.name
    return max_file, max_number

def contour_plot(
    v,
    frames_folder,
    t = -1,
    label = '',
):
    lim = max(abs(np.min(v)), abs(np.max(v)))
    lin_thresh = np.power(10.,np.floor(np.log10(lim/100))) # closest power of 10
    log_norm = colors.SymLogNorm(linthresh=lin_thresh, vmin=-lim, vmax=lim)
    col_map = cm.PuOr_r
    plt.contourf(v.T, norm=log_norm, cmap=col_map, levels=200)
    plt.colorbar(cm.ScalarMappable(norm=log_norm, cmap=col_map), ax=plt.gca())
    theta = np.linspace(0, N-1, num=5)
    plt.xticks(theta, ['0', 'π/2', 'π', '3π/2', '2π'])
    plt.yticks(theta, ['0', 'π/2', 'π', '3π/2', '2π'])
    if label != '':
        if t >= 0:
            # plt.title(f'{label}  |  t={(t*Dt):.3f}')
            plt.title(f'{label}  |  t={(t*Dt + 5792.311):.3f}') # to correct different Dts
        else:
            plt.title(f'{label}')
    else:
        if t >= 0:
            plt.title(f't={(t*Dt):.3f}')
    # plt.show()
    plt.savefig(frames_folder / f'{t:05}.png', dpi=200)#
    plt.close()

def relative_pos(
    x_ctr,
    y_ctr,
):
    pos = np.arange(N) * Dx
    x_dist = (pos - x_ctr + np.pi) % (2*np.pi) - np.pi
    y_dist = (pos - y_ctr + np.pi) % (2*np.pi) - np.pi
    dist2ctr = np.sqrt(np.square(x_dist)[:, None] + np.square(y_dist)[None, :])
    angle = np.arctan2(y_dist[None, :], x_dist[:, None])
    return dist2ctr, angle

def lamb_dipole(
    orient = np.pi/2,
    U = 2.,
):
    radius = 0.73
    c_0 = 3.83170597020751231
    K = c_0 / radius
    C_L = - 2. * K * U / sp.j0(c_0)
    ctr = int(N/2) - 0.5
    dist2axis = np.arange(N) - ctr
    dist2ctr = np.sqrt(np.square(dist2axis)[:, None] + np.square(dist2axis)[None, :]) * Dx
    angle = np.arctan2(dist2axis[None, :], dist2axis[:, None])
    dipole = C_L * sp.j1(K*dist2ctr) * np.sin(angle - orient)
    dipole[dist2ctr > radius] = 0
    return dipole

def zero_forcing():
    return np.zeros((N, N))

@cache
def modulus_k(
    n: int,
):
    square_sum = np.square(np.arange(n))[:, None] + np.square(np.arange(int(n/2)+1))[None, :]
    mod_k = np.sqrt(square_sum)
    mod_k[int(n/2)+1:n, :] = mod_k[n-int(n/2)-1:0:-1, :]
    return mod_k

@cache
def forcing_spectrum(
    strength: float,
    k_F: float,
):
    k = modulus_k(N)
    f = strength * 10e6 * np.logical_and(k_F-1 < k, k < k_F+1)
    # f = strength * 10e6 * np.logical_and(k_F-1.5 < k, k < k_F+1.5)
    return np.sqrt(k * f / np.pi)

def random_forcing(
    strength = 0.1,
    k_F = N / 8,
    # k_F = N * 25/128,
):
    power_spectrum = forcing_spectrum(strength, k_F)
    random_phase = np.random.rand(N,int(N/2)+1)
    F_ft = power_spectrum * np.exp(2j*np.pi * random_phase)
    return ft.irfft2(F_ft)

def energy(
    q,
    psi,
):
    return np.sum(q * psi / 2)

def energy_input(
    F,
    psi,
):
    return np.sum(F * psi) * Dt

def derivative(
    f,
    axis: int,
):
    df = (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) / (2 * Dx)
    return df
    
def velocity(
    psi,
):
    v_x = derivative(psi, 1) # d(psi)/dy
    v_y = -derivative(psi, 0) # -d(psi)/dx
    return v_x, v_y

def pseudo_laplacian2d(
    f,
):
    return np.roll(f, -1, axis=0) + np.roll(f, +1, axis=0) + np.roll(f, +1, axis=1) + np.roll(f, -1, axis=1) - 4* f

def spectrum(
    f,
):
    f_ft = ft.rfft2(f)
    k = modulus_k(N)
    k_range = np.floor((int(N/2) + 1) * np.sqrt(2))
    spectrum, _, _ = binned_statistic(k.flatten(), np.abs(f_ft).flatten(), statistic='mean', bins=np.arange(k_range+1))
    return 2*np.pi * np.arange(k_range) * spectrum

def linear_sys_inv(
    c,
    d,
):
    M = (1 + c + d/2) * np.eye(N) - np.diag(c/2 * np.ones(N-1), 1) - np.diag(c/2 * np.ones(N-1), -1)
    M[0, N-1] = M[N-1, 0] = -c/2
    return la.inv(M)

def arakawa_jacobian(
    q,
    p,
):
    # the neighbours of the i-th point are named by numbers
    # according to the following order:
    #    y
    #    ^ 8 1 2
    #    | 7 i 3
    #    | 6 5 4
    #   -|------> x
    # only the needed ones will be saved in memory
    q1 = np.roll(q,  -1, axis=1)
    q2 = np.roll(q1, -1, axis=0)
    q3 = np.roll(q,  -1, axis=0)
    q4 = np.roll(q3, +1, axis=1)
    q5 = np.roll(q,  +1, axis=1)
    q6 = np.roll(q5, +1, axis=0)
    q7 = np.roll(q,  +1, axis=0)
    q8 = np.roll(q7, -1, axis=1)
    p1 = np.roll(p,  -1, axis=1)
    p2 = np.roll(p1, -1, axis=0)
    p3 = np.roll(p,  -1, axis=0)
    p4 = np.roll(p3, +1, axis=1)
    p5 = np.roll(p,  +1, axis=1)
    p6 = np.roll(p5, +1, axis=0)
    p7 = np.roll(p,  +1, axis=0)
    p8 = np.roll(p7, -1, axis=1)
    return - ((p5 + p4 - p1 - p2) * (q3 - q)
            + (p6 + p5 - p8 - p1) * (q - q7)
            + (p3 + p2 - p7 - p8) * (q1 - q)
            + (p4 + p3 - p6 - p7) * (q - q5)
            + (p3 - p1) * (q2 - q)
            + (p5 - p7) * (q - q6)
            + (p1 - p7) * (q8 - q)
            + (p3 - p5) * (q - q4)) / (12 * Dx**2)

@cache
def laplacian_eig(
    shape,
):
    cos_sum = np.cos(np.arange(shape[0]) * Dx)[:, None] + np.cos(np.arange(shape[1]) * Dx)[None, :]
    lap_eig = 2* (2 - cos_sum) / Dx**2
    lap_eig[0, 0] = np.inf
    return lap_eig

def inv_laplacian2d(
    omega,
):
    omega_ft = ft.rfft2(omega)
    psi_ft = omega_ft / laplacian_eig(omega_ft.shape)
    return ft.irfft2(psi_ft)

def avg_coord( # averaging in the square domain [a, b]^2
    f, # weights
    a = 0,
    b = N,
):
    (x, y) = coordinates()
    avg_x = np.average(x[a:b, a:b], weights=f[a:b, a:b])
    avg_y = np.average(y[a:b, a:b], weights=f[a:b, a:b])
    return avg_x, avg_y

def find_vortex_center(
    q,
):
    ctr = int(N/2)
    hr = int(N/32) # half range
    (x_max, y_max) = np.unravel_index(q.argmax(), q.shape)
    max_ctr_q = np.roll(np.roll(q, ctr-x_max, axis=0), ctr-y_max, axis=1) # to bring the max in the position (N/2, N/2)
    avg_max_ctr_q = avg_coord(max_ctr_q, ctr-hr, ctr+hr)
    pos_ctr = avg_max_ctr_q + (np.array([x_max, y_max])-ctr) *Dx # revert reference frame change
    return pos_ctr

def avg_vorticity(
    v_x,
    v_y,
    r,
    angle,
):
    u = v_y * np.cos(angle) - v_x * np.sin(angle) # scalar product with tangential versor (-sin(a), cos(a))
    du_x = derivative(u, 0) # compute gradient of u
    du_y = derivative(u, 1)
    du_r = du_x * np.cos(angle) + du_y * np.sin(angle) # scalar product with radial versor (cos(a), sin(a))
    return u/r + du_r

def avg_centered_field(
    q,
    v_x,
    v_y,
    bins,
):
    dist, angle = relative_pos(*find_vortex_center(q))
    omega = avg_vorticity(v_x, v_y, dist, angle)
    avg_omega, _, _ = binned_statistic(dist.flatten(), omega.flatten(), statistic='mean', bins=bins)
    return avg_omega

def rungekutta_step(
    c,
    d,
    gamma,
    rho,
    L,
    q,
    psi,
    J,
    F,
):
    J_p = J
    J = arakawa_jacobian(q, psi)
    rhs = Dt*((gamma+rho)*F - (gamma*J + rho*J_p)) - d*q + c*pseudo_laplacian2d(q)
    middle_step = np.matmul(L, rhs)
    Dq = np.matmul(middle_step, L.transpose())
    q = q + Dq
    psi = inv_laplacian2d(q)
    return q, psi, J

def time_iter(
    q,
    psi,
    J,
    forcing: callable,
    t0 = -1,
):
    gamma = np.array((8/15, 5/12, 3/4))
    rho = np.array((0, -17/60, -5/12))
    alpha = gamma + rho
    c = Dt * alpha / (Re * Dx**2)
    d = Dt * alpha * ekman
    L = [0, 0, 0]
    for i in range(3):
        L[i] = linear_sys_inv(c[i], d[i])

    frames_folder, bak_folder = open_folders()
    stat_file = open(bak_folder / 'time_stat.dat', 'a')
    print('# time E eps avg_k', file=stat_file)
    max_dist = int(N/2 * np.sqrt(2)) # maximum distance from the domain center
    bkgnd_sum = np.zeros(max_dist)
    bins = np.arange(max_dist+1) * Dx
    F = forcing()
    eps = 0.02878
    revol_time = pow(sd_len**2/(2*eps), 1/3)
    T_print = min(int(revol_time/Dt), int(T/5))
    T_update = max(10, int(T/1000))
    print(f'{T_print=}\t{T/T_print} elements for statistics')
    time_exec = tqdm(np.arange(t0+1, T+t0+2))

    for t in time_exec:
        F = forcing()
        for i in range(3):
            q, psi, J = rungekutta_step(c[i], d[i], gamma[i], rho[i], L[i], q, psi, J, F)
        if t%T_update == 0:
            v_x, v_y = velocity(psi)
            E = energy(q, psi)
            eps = energy_input(F, psi)
            S = spectrum(q)
            avg_k = np.average(np.arange(S.size), weights=S)
            time_exec.set_description(f'E = {E:.5g}  |  <k> = {avg_k:.5g}  |  eps = {eps:.5g} ')
            print(t*Dt + 5792.311, E, eps, avg_k, sep='\t', file=stat_file)
        if t%T_print == 0:
            contour_plot(q, frames_folder, t, 'Vorticity')
            np.save(bak_folder / f'q_{t:06}', q)
        bkgnd_sum = bkgnd_sum + avg_centered_field(q, v_x, v_y, bins) + avg_centered_field(-q, -v_x, -v_y, bins)
        
    stat_file.close()
    bkgnd = bkgnd_sum / (2 * (T + 1))
    np.save(bak_folder / f'bkgnd', np.array((bins[:-1], bkgnd)))

def main():
    if reload_:
        _, bak_folder = open_folders()
        bak_file, t0 = find_highest_numbered_file(bak_folder)
        if bak_file:
            q = np.load(bak_folder / bak_file)
        else:
            raise Exception('Error: no valid backup file found.')
        psi = inv_laplacian2d(q)
        J = arakawa_jacobian(q, psi)
        time_iter(q, psi, J, random_forcing, t0)
    else:
        q = psi = J = np.zeros((N, N))
        time_iter(q, psi, J, random_forcing)

if __name__ == "__main__":
    main()