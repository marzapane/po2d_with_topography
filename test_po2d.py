import pytest
import po2d
import numpy as np
import matplotlib.pyplot as plt

N = po2d.N

def test_inv_laplacian():
    rand_k = 10 * np.random.rand(8)
    x = 2*np.pi * np.arange(N)/N
    data = np.sin(x * rand_k[0] + rand_k[1])[:, None] * np.sin(x * rand_k[2] + rand_k[3])[None, :] + np.sin(x * rand_k[4] + rand_k[5])[:, None] * np.sin(x * rand_k[6] + rand_k[7])[None, :]
    inv_lap = po2d.inv_laplacian2d(data)
    result = - po2d.pseudo_laplacian2d(inv_lap) / po2d.Dx**2
    if np.mean(np.abs(result - data)) > np.mean(np.abs(data))/10:
        print(f'Average difference ({np.mean(np.abs(result-data))}) >> average field ({np.mean(np.abs(data))})')
        plt.contourf(data.T, label='exp')
        plt.show()
        plt.contourf(result.T, label='obt')
        plt.sho()
        raise Exception('Error: Laplacian not inverted correctly.')

def test_inverse():
    c = 0.01
    d = 0.003
    M = (1 + c + d/2) * np.eye(N) + np.diag(-c/2 * np.ones(N-1), 1) + np.diag(-c/2 * np.ones(N-1), -1)
    M[0, N-1] = M[N-1, 0] = -c/2
    L = po2d.linear_sys_inv(c, d)
    if np.mean(np.abs(np.matmul(L, M) - np.eye(N))) > 10e-3:
        raise Exception('Error: matrix not inverted properly.')

def test_forcing():
    F = po2d.random_forcing(k_F=N/8)
    if F.shape != (N, N):
        print(f'{F.shape=}')
        raise Exception('Error: something went wrong in the slicing.')
    s_F = po2d.spectrum(F)
    k_F = np.argmax(s_F)
    if (k_F > N/8 + 1) or (k_F < N/8 - 1):
        plt.loglog(s_F)
        plt.xlabel('|k|')
        plt.ylabel('Power Spectrum of sine_forcing')
        plt.show()
        raise Exception('Error: something went wrong in the forcing frequency.')

def test_derivatives():
    x, y = po2d.coordinates()
    f = 5 * x - 3 * y
    f_x = po2d.derivative(f, 0)
    f_y = po2d.derivative(f, 1)
    avg_dx = np.mean(f_x[1:-1, :])
    avg_dy = np.mean(f_y[:, 1:-1])
    if avg_dx != 5 or avg_dy != -3:
        print('\nf(x,y) = 5x - 3y')
        print(f'<f_x> = {avg_dx}\n<f_y> = {avg_dy}')
        raise Exception('Error: wrong derivatives.')


@pytest.mark.slow
def test_lamb_dipole():
    q = po2d.lamb_dipole()
    psi = po2d.inv_laplacian2d(q)
    J = po2d.arakawa_jacobian(q, psi)
    po2d.T = 201
    po2d.time_iter(q, psi, J, po2d.zero_forcing)

def test_find_center():
    (x, y) = po2d.coordinates()
    q = (np.exp(-(x-3)**2-(y-2)**2) - np.exp(-(x-1)**2-(y-3.5)**2)) * (10 + np.random.rand(N,N))
    pos_vortex_ctr = po2d.find_vortex_center(q)
    nearest_pos_ctr = np.around(pos_vortex_ctr / (2*np.pi/N)).astype(int)
    q_pos_ctr = q[tuple(nearest_pos_ctr)]
    neg_vortex_ctr = po2d.find_vortex_center(-q)
    nearest_neg_ctr = np.around(neg_vortex_ctr / (2*np.pi/N)).astype(int)
    q_neg_ctr = q[tuple(nearest_neg_ctr)]
    for i in range(N):
        rand_pos = np.random.randint(0, N, size=2)
        if(q[tuple(rand_pos)]>q_pos_ctr or q[tuple(rand_pos)]<q_neg_ctr):
            print(f'\n{pos_vortex_ctr=}, {neg_vortex_ctr=}')
            print(f'{nearest_pos_ctr=}, {nearest_neg_ctr=}')
            print(f'{q_pos_ctr=}, {q_neg_ctr=}')
            plt.contourf(q.T, levels=50)
            plt.colorbar()
            plt.scatter(*(pos_vortex_ctr / (2*np.pi/N)), c='red')
            plt.scatter(*(neg_vortex_ctr / (2*np.pi/N)), c='black')
            plt.scatter(*(nearest_pos_ctr), c='red')
            plt.scatter(*(nearest_neg_ctr), c='black')
            plt.show()
            return
        
def test_avg_vorticity():
    (x, y) = po2d.coordinates()
    q = (np.exp(-(x-3)**2-(y-2)**2) - np.exp(-(x-1)**2-(y-3.5)**2)) * (10 + np.random.rand(N,N))
    psi = po2d.inv_laplacian2d(q)
    v_x, v_y = po2d.velocity(psi)
    bins = np.arange(int(N/2 * np.sqrt(2))+1) * po2d.Dx
    omega_pos = po2d.avg_centered_field(q, v_x, v_y, bins)
    omega_neg = po2d.avg_centered_field(-q, -v_x, -v_y, bins)
    plt.plot(bins[:-1], omega_pos, label='+')
    plt.plot(bins[:-1], omega_neg, label='-')
    plt.show()
