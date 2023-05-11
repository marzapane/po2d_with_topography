import pytest
import numpy as np
import matplotlib.pyplot as plt
import po2d_classes as po
from po2d_config import N, sd_len, Dx, Dt


def test_inv_laplacian():
    rand_k = 10 * np.random.rand(8)
    x = 2*np.pi * np.arange(N)/N
    data = np.sin(x * rand_k[0] + rand_k[1])[:, None] * np.sin(x * rand_k[2] + rand_k[3])[None, :] + np.sin(x * rand_k[4] + rand_k[5])[:, None] * np.sin(x * rand_k[6] + rand_k[7])[None, :]
    inv_lap = po.inv_laplacian2d(data, Dx)
    result = - po.pseudo_laplacian2d(inv_lap) / Dx**2
    if np.mean(np.abs(result - data)) > np.mean(np.abs(data))/10:
        print(f'Average difference ({np.mean(np.abs(result-data))}) >> average field ({np.mean(np.abs(data))})')
        plt.contourf(data.T, label='exp')
        plt.title('test_inv_laplacian')
        plt.show()
        plt.contourf(result.T, label='obt')
        plt.title('test_inv_laplacian')
        plt.show()
        raise Exception('Error: Laplacian not inverted correctly.')

def test_inverse():
    c = 0.01
    d = 0.003
    M = (1 + c + d/2) * np.eye(N) + np.diag(-c/2 * np.ones(N-1), 1) + np.diag(-c/2 * np.ones(N-1), -1)
    M[0, N-1] = M[N-1, 0] = -c/2
    L = po.FluidSimulator.linear_sys_inv(c, d, N)
    if np.mean(np.abs(np.matmul(L, M) - np.eye(N))) > 10e-3:
        raise Exception('Error: matrix not inverted properly.')

def test_forcing():
    F = po.random_forcing(N)
    if F.shape != (N, N):
        print(f'{F.shape=}')
        raise Exception('Error: something went wrong in the slicing.')
    s_F = po.spectrum(F, N)
    k_F = np.argmax(s_F)
    if (k_F > N *25/128 + 1.5) or (k_F < N *25/128 - 1.5):
        plt.loglog(s_F)
        plt.xlabel('|k|')
        plt.ylabel('Power Spectrum of sine_forcing')
        plt.title('test_forcing')
        plt.show()
        raise Exception('Error: something went wrong in the forcing frequency.')

def test_derivatives():
    x, y = po.coordinates(Dx, N)
    f = 5 * x - 3 * y
    f_x = po.derivative(f, 0, Dx)
    f_y = po.derivative(f, 1, Dx)
    avg_dx = np.mean(f_x[1:-1, :])
    avg_dy = np.mean(f_y[:, 1:-1])
    if avg_dx != 5 or avg_dy != -3:
        print('\nf(x,y) = 5x - 3y')
        print(f'<f_x> = {avg_dx}\n<f_y> = {avg_dy}')
        raise Exception('Error: wrong derivatives.')


@pytest.mark.slow
def test_lamb_dipole():
    simul = po.FluidSimulator(po.zero_forcing, False)
    simul.T = 200
    fluid = po.FluidState(vorticity = po.lamb_dipole(Dx, N))
    simul.set_physical_param(fluid)
    for t in simul.time_exec:
        simul.advance_dt(fluid, t)
    simul.conclude()

def test_find_center():
    (x, y) = po.coordinates(Dx, N)
    q = (np.exp(-(x-3)**2-(y-2)**2) - np.exp(-(x-1)**2-(y-3.5)**2)) * (10 + np.random.rand(N,N))
    fluid = po.FluidState(vorticity = q)
    fluid.psi = fluid.streamfunction()
    (fluid.v_x, fluid.v_y) = fluid.velocity()
    pos_vortex_ctr = fluid.find_vortex_center()
    nearest_pos_ctr = np.around(pos_vortex_ctr / (2*np.pi/N)).astype(int)
    q_pos_ctr = q[tuple(nearest_pos_ctr)]
    neg_vortex_ctr = (-fluid).find_vortex_center()
    nearest_neg_ctr = np.around(neg_vortex_ctr / (2*np.pi/N)).astype(int)
    q_neg_ctr = q[tuple(nearest_neg_ctr)]
    for i in range(N):
        rand_pos = np.random.randint(0, N, size=2)
        if(q[tuple(rand_pos)]>q_pos_ctr or q[tuple(rand_pos)]<q_neg_ctr):
            print(f'\n{pos_vortex_ctr=}, {neg_vortex_ctr=}')
            print(f'{nearest_pos_ctr=}, {nearest_neg_ctr=}')
            print(f'{q_pos_ctr=}, {q_neg_ctr=}')
            plt.streamplot(x.T, y.T, fluid.v_x.T, fluid.v_y.T, density=3)
            plt.scatter(*(pos_vortex_ctr), c='red')
            plt.scatter(*(neg_vortex_ctr), c='black')
            plt.scatter(*(nearest_pos_ctr*Dx), c='red')
            plt.scatter(*(nearest_neg_ctr*Dx), c='black')
            plt.title('test_find_center')
            plt.show()
            return
        
def test_avg_vorticity():
    (x, y) = po.coordinates(Dx, N)
    q = (np.exp(-(x-3.1)**2-(y-2.03)**2) - np.exp(-(x-1.06)**2-(y-3.5)**2)) * (10 + np.random.rand(N,N))
    fluid = po.FluidState(vorticity = q)
    fluid.psi = fluid.streamfunction()
    (fluid.v_x, fluid.v_y) = fluid.velocity()
    bins = np.arange(int(N/2 * np.sqrt(2))+1) * Dx
    omega_pos = fluid.avg_centered_field(bins)
    omega_neg = (-fluid).avg_centered_field(bins)
    plt.plot(bins[:-1], omega_pos, label='+')
    plt.plot(bins[:-1], omega_neg, label='-')
    plt.title('test_avg_vorticity')
    plt.show()