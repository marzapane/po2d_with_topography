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
from scipy.ndimage import gaussian_filter
from pathlib import Path

class FluidSimulator:
    from po2d_config import T, N, Re, ekman, eps, cfg_name, sd_len, Dx, Dt
    reload_bak = True
    
    gamma = np.array((8/15, 5/12, 3/4))
    rho = np.array((0, -17/60, -5/12))
    alpha = gamma + rho
    L = [None, None, None]

    def __init__(
        self,
        forcing: callable,
        analize_vortex = False,
        adaptive_Dt = False,
        plot_flag = True,
    ):
        self.forcing = forcing
        self.analize_vortex = analize_vortex
        self.adapt_Dt = adaptive_Dt
        self.diagnostics = False
        self.plot_flag = plot_flag

    def set_diagnostics(self):
        self.diagnostics = True
        print(f'Running simulation "{self.cfg_name}"')
        self.frames_dir, self.bak_dir = self.setup_folders() # looking for past simulations backup files
        self.bak_file, self.t0 = find_highest_numbered_npz(self.bak_dir)
        self.reload_bak = self.confirm_reload()
        open_folder(self.bak_dir, overwrite = not self.reload_bak)
        open_folder(self.frames_dir, overwrite = not self.reload_bak)
        if self.reload_bak:
            self.time = np.load(self.bak_dir / self.bak_file)['t']
            self.stat_file = open(self.bak_dir / 'time_stat.dat', 'a')
        else:
            self.t0 = 0
            self.time = 0.
            self.stat_file = open(self.bak_dir / 'time_stat.dat', 'w')
            print('# time E eps avg_k c(q)', file=self.stat_file)
        if self.analize_vortex:
            max_dist = int(self.N/2 * np.sqrt(2))    # furthest distance from the domain center
            self.bkgnd_sum = np.zeros(max_dist)
            self.bins = np.arange(max_dist+1) * self.Dx

    def setup_folders(self):
        result_dir = Path().resolve().parent / 'Results'
        if not result_dir.exists():
            result_dir.mkdir()
        fold_name = f'Ek{self.ekman:.1e},Re{self.Re:.0e},N{self.N}'
        if self.cfg_name != '':
            fold_name = f'{self.cfg_name},{fold_name}'
        bak_dir = result_dir / fold_name
        frames_dir = bak_dir / f'frames'
        return frames_dir, bak_dir

    def confirm_reload(self):   # returns the answer to "Restart past simulation?"
        if self.bak_file:   # asking user confirmation to reload last simulation (default)
            print(f'A previous simulation has been found: {self.bak_file}.')
            print('Starting a different one will overwrite it.')
            user_input = input('  Reload last simulation? [Y/n] ')
            if user_input.lower() in ['no', 'n']:
                print('Launching a new simulation and overwriting the past one.')
                return False
            else:
                print(f'Restarting past simulation from t0={self.t0}.')
                return True
        else:
            print('No previous simulation was found; launching a new one.')
            return False

    def set_physical_param(self, fluid):
        self.set_integration_const()
        if self.adapt_Dt:
            self.max_CFL = np.sqrt(3)  # for 3rd order Runge-Kutta
            self.cur_CFL = 1
        revol_time = pow(self.sd_len**2/(2*self.eps), 1/3)
        self.T_print = 0
        self.dT_print = min(int(revol_time/self.Dt), int(self.T/5))
        self.T_update = max(10, int(self.T/1000))
        # print(f'Simulation times are:\n  Dt = {self.Dt}\n  T_LE ~ {revol_time}')
        self.time_exec = tqdm(np.arange(self.T+1))
        self.F = self.forcing(self.N)
        fluid.psi = fluid.streamfunction(self.t0)
        fluid.J = fluid.arakawa_jacobian(self.t0)
        
    def set_integration_const(self):
        self.c = self.Dt * self.alpha / (self.Re * self.Dx**2)
        self.d = self.Dt * self.alpha * self.ekman
        for i in range(3):
            self.L[i] = self.linear_sys_inv(self.c[i], self.d[i], self.N)

    @staticmethod
    def linear_sys_inv(
        c: float,   # Runge-kutta coefficients
        d: float,
        n: int,     # grid size
    ):
        M = (1 + c + d/2) * np.eye(n) - np.diag(c/2 * np.ones(n-1), 1) - np.diag(c/2 * np.ones(n-1), -1)
        M[0, n-1] = M[n-1, 0] = -c/2
        return la.inv(M)
    
    def advance_dt(
        self,
        fluid,
        t: int,
    ):
        for step in range(3):
            self.rungekutta_step(fluid, step, t)
        self.time += self.Dt
        if (t % self.T_update) == 0:
            E = fluid.energy()
            eps = fluid.energy_dissipation(self.Re)
            revol_time = pow(self.sd_len**2/(2*eps), 1/3)
            self.dT_print = min(int(revol_time/self.Dt), int(self.T/5))
            # E_in = fluid.energy_input(self.F)
            S = energy_spectrum(*fluid.velocity(t))
            avg_k = np.average(np.arange(S.size), weights=S) if S.any() else 0.0
            self.time_exec.set_description(f't = {self.time:.1f} | E = {E:.2g} | <k> = {avg_k:.2g} | eps = {eps:.2g}')
            if self.diagnostics:
                print(self.time, E, eps, avg_k, measure_concentration(fluid.q, self.N), sep='\t', file=self.stat_file, flush=True)
            if self.adapt_Dt:
                fluid.velocity(t)
                max_velocity = np.hypot(fluid.v_x, fluid.v_y).max()
                line_Dt = (self.cur_CFL * self.Dx / max_velocity / 10) if (max_velocity > 0) else self.Dt
                turb_Dt = revol_time/np.sqrt(self.Re)
                Dt = min(line_Dt, turb_Dt, 50) / 4
                if abs(Dt - self.Dt) > self.Dt / 5:
                    self.Dt = Dt
                    print(f'    !!  v({t})<{max_velocity:.3g}  &  T_rev={revol_time:.1f}   ->   Dt = {self.Dt:.2g}')
                    self.set_integration_const()
        if t >= self.T_print + self.dT_print:
            self.T_print = t
            if self.diagnostics:
                np.savez(self.bak_dir / f'q_{(t+self.t0):06}', q=fluid.q, t=self.time)
            if self.plot_flag:
                fluid.plot_field(t, self, print_fig=self.diagnostics)
        if self.analize_vortex:
            fluid.velocity(t)
            self.bkgnd_sum = self.bkgnd_sum + fluid.avg_centered_field(self.bins) + (-fluid).avg_centered_field(self.bins)

    def rungekutta_step(
        self,
        fluid,
        step: int,
        t: int,
    ):
        F_p = self.F.copy()
        self.F = self.forcing(self.N)
        J_p = fluid.J.copy()
        fluid.arakawa_jacobian(t)
        rhs = self.Dt* (self.gamma[step]*(self.F - fluid.J) + self.rho[step]*(F_p - J_p)) - self.d[step]*fluid.q + self.c[step]*pseudo_laplacian2d(fluid.q*fluid.h)
        middle_step = np.matmul(self.L[step], rhs)
        Dq = np.matmul(middle_step, self.L[step].transpose())
        fluid.q = fluid.q + Dq
        fluid.streamfunction(t)

    def conclude(self):
        if self.diagnostics:
            self.stat_file.close()
        if self.analize_vortex:
            self.bkgnd = self.bkgnd_sum / (2 * (self.T + 1))
            np.save(self.bak_dir / f'bkgnd', np.array((self.bins[:-1], self.bkgnd)))
    

class FluidState:
    from po2d_config import N
    from po2d_config import Dx

    def __init__(
        self,
        reload_bak = False,
        bak_dir = None,
        bak_file = None,
        vorticity = None,
    ):
        if reload_bak:
            self.q = np.load(bak_dir / bak_file)['q']
        else:
            if vorticity is None:
                self.q = np.zeros((self.N, self.N))
            else:
                if isinstance(vorticity, np.ndarray) and vorticity.shape == (self.N, self.N):
                    self.q = vorticity
                else:
                    raise Exception("Error: wrong argument: q must be <numpy.ndarray> of shape (N, N).")
        self.psi = None
        self.t_psi = None
        self.J = None
        self.t_J = None
        (self.v_x, self.v_y) = (None, None)
        self.t_vel = None

    def streamfunction(
        self,
        t: int,
    ):
        if self.t_psi != t:
            self.t_psi = t
            self.psi = inv_laplacian2d(self.q, self.Dx)
        return self.psi

    def velocity(
        self,
        t: int,
    ):
        if self.t_vel != t:
            self.t_vel = t
            self.streamfunction(t)
            self.v_x = derivative(self.psi, 1, self.Dx)  # d(psi)/dy
            self.v_y = -derivative(self.psi, 0, self.Dx) # -d(psi)/dx
        return self.v_x, self.v_y

    def energy(self):
        return np.sum(self.q * self.psi / 2)

    def energy_dissipation(
        self,
        Re: float,
    ):
        velocity_gradient = self.q * self.h
        return np.square(velocity_gradient).sum() / Re

    def energy_input(
        self,
        F: np.ndarray,
    ):
        return np.sum(F * self.psi)

    def avg_centered_field(
        self,
        bins: np.ndarray,
    ):
        vortex_ctr = self.find_vortex_center()
        dist, angle = relative_pos(*vortex_ctr, self.Dx, self.N)
        omega = self.avg_vorticity(dist, angle)
        avg_omega, _, _ = binned_statistic(dist.flatten(), omega.flatten(), statistic='mean', bins=bins)
        return avg_omega

    def find_vortex_center(self):
        ctr = int(self.N/2)     # grid center
        hr = int(self.N/16)     # half range of the interval considered
        q_flt = gaussian_filter(self.q, sigma=256/25, truncate=2., mode='wrap') # gaussian convolution, to exclude large fluctuations
        (x_max, y_max) = np.unravel_index(q_flt.argmax(), q_flt.shape)
        max_ctr_v_x = np.roll(np.roll(self.v_x, ctr-x_max, axis=0), ctr-y_max, axis=1)  #bring the max in the position (N/2, N/2)
        max_ctr_v_y = np.roll(np.roll(self.v_y, ctr-x_max, axis=0), ctr-y_max, axis=1)
        zero_max_ctr_v = zero_coord(max_ctr_v_x, max_ctr_v_y, ctr-hr, ctr+hr)
        grid_ctr = zero_max_ctr_v + np.array([x_max, y_max])-ctr    # revert reference frame change
        return (grid_ctr % self.N) * self.Dx

    def avg_vorticity(
        self,
        r: np.ndarray,
        angle: np.ndarray,
    ):
        u = self.v_y * np.cos(angle) - self.v_x * np.sin(angle)   # scalar product with tangential versor (-sin(a), cos(a))
        du_x = derivative(u, 0, self.Dx)     # compute gradient of u
        du_y = derivative(u, 1, self.Dx)
        du_r = du_x * np.cos(angle) + du_y * np.sin(angle)  # scalar product with radial versor (cos(a), sin(a))
        return u/r + du_r

    def plot_field(
        self,
        t: int,
        simul = None,
        print_fig = True,
    ):
        lim = max(abs(np.min(self.q)), abs(np.max(self.q)))
        lin_thresh = np.power(10, np.floor(np.log10(lim/100)))  # power of 10 closest to lim/100
        log_norm = colors.SymLogNorm(linthresh=lin_thresh, vmin=-lim, vmax=lim)
        # col_map = cm.PuOr_r
        col_map = cm.BrBG
        streamplot_color='purple'
        (x, y) = coordinates(self.Dx, self.N)
        plt.contourf(x.T, y.T, self.q.T, norm=log_norm, cmap=col_map, levels=75)
        plt.colorbar(cm.ScalarMappable(norm=log_norm, cmap=col_map), ax=plt.gca())
        self.streamfunction(t)
        self.velocity(t)
        plt.streamplot(x.T, y.T, self.v_x.T, self.v_y.T, color=streamplot_color)
        theta = np.linspace(0, (self.N-1) * self.Dx, num=5)
        labels = ['$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        plt.xticks(theta, labels)
        plt.yticks(theta, labels)
        plt.xlim(0, (self.N-1) * self.Dx)
        plt.ylim(0, (self.N-1) * self.Dx)
        if simul:
            plt.title(f'Vorticity  $|$  t={simul.time:.3f}')
        plt.gca().set_aspect('equal')
        if print_fig and simul:
            plt.savefig(simul.frames_dir / f'{(t+simul.t0):05}.png', dpi=200)
        else:
            plt.show()
        plt.close()

    def arakawa_jacobian(
        self,
        t: int,
    ):
        if self.t_J != t:
            self.t_J = t
            # the neighbours of the i-th point are named by numbers according to the following order:
            #    y
            #    ^ 8 1 2
            #    | 7 i 3
            #    | 6 5 4
            #   -|------> x
            # only the needed ones will be saved in memory
            q = self.q
            q1 = np.roll(q,  -1, axis=1)
            q2 = np.roll(q1, -1, axis=0)
            q3 = np.roll(q,  -1, axis=0)
            q4 = np.roll(q3, +1, axis=1)
            q5 = np.roll(q,  +1, axis=1)
            q6 = np.roll(q5, +1, axis=0)
            q7 = np.roll(q,  +1, axis=0)
            q8 = np.roll(q7, -1, axis=1)
            p = self.psi
            p1 = np.roll(p,  -1, axis=1)
            p2 = np.roll(p1, -1, axis=0)
            p3 = np.roll(p,  -1, axis=0)
            p4 = np.roll(p3, +1, axis=1)
            p5 = np.roll(p,  +1, axis=1)
            p6 = np.roll(p5, +1, axis=0)
            p7 = np.roll(p,  +1, axis=0)
            p8 = np.roll(p7, -1, axis=1)
            self.J= -((p5 + p4 - p1 - p2) * (q3 - q)
                    + (p6 + p5 - p8 - p1) * (q - q7)
                    + (p3 + p2 - p7 - p8) * (q1 - q)
                    + (p4 + p3 - p6 - p7) * (q - q5)
                    + (p3 - p1) * (q2 - q)
                    + (p5 - p7) * (q - q6)
                    + (p1 - p7) * (q8 - q)
                    + (p3 - p5) * (q - q4)) / (12 * self.Dx**2)
        return self.J

    def __neg__(self):
        neg_self = self.__class__(vorticity = -self.q)
        if self.psi is not None:
            neg_self.psi = -self.psi
        if self.v_x is not None:
            (neg_self.v_x, neg_self.v_y) = (-self.v_x, -self.v_y)
        return neg_self
    
    def __add__(self, other):
        sum_ = self.__class__(vorticity = self.q + other.q)
        if (self.psi is not None) and (other.psi is not None):
            sum_.psi = self.psi + other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (sum_.v_x, sum_.v_y) = (self.v_x + other.v_x, self.v_y + other.v_y)
        return sum_

    def __sub__(self, other):
        diff = self.__class__(vorticity = self.q - other.q)
        if (self.psi is not None) and (other.psi is not None):
            diff.psi = self.psi - other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (diff.v_x, diff.v_y) = (self.v_x - other.v_x, self.v_y - other.v_y)
        return diff

    def __repr__(self):
        psi_exist = (self.psi is not None)
        J_exist = (self.J is not None)
        v_exist = (self.v_x is not None)
        return f'<{self.__class__.__name__}> instance of size {self.q.shape} with attributes [q{", psi" if psi_exist else ""}{", J" if J_exist else ""}{", v_x, v_y" if v_exist else ""}].'

class FluidStateTopography(FluidState):
    def __init__(
        self,
        topography: np.ndarray,
        reload_bak = False,
        bak_dir = None,
        bak_file = None,
        vorticity = None,
    ):
        if reload_bak:
            self.topography = np.load(bak_dir / 'topography.npy')
        else:
            self.topography = topography
            if bak_dir is not None:
                topography_file = bak_dir / 'topography.npy'
                if not topography_file.is_file():
                    np.save(topography_file, topography)
        self.h = 1 - 2/3 * self.topography/self.topography.max()
        super().__init__(reload_bak, bak_dir, bak_file, vorticity)

    def streamfunction(
        self,
        t: int,
    ):
        if self.t_psi != t:
            self.t_psi = t
            self.psi = inv_laplacian2d(self.q * self.h, self.Dx)
        return self.psi
    
    def plot_field(
        self,
        t: int,
        simul = None,
        print_fig = True,
    ):
        # plt.contourf(self.v_x); plt.colorbar(); plt.show(); plt.close()
        lim = max(abs(np.min(self.q)), abs(np.max(self.q)))
        lin_thresh = np.power(10.,np.floor(np.log10(lim/100)))  # closest power of 10
        log_norm = colors.SymLogNorm(linthresh=lin_thresh, vmin=-lim, vmax=lim)
        # col_map = cm.PuOr_r
        col_map = cm.BrBG
        streamplot_color='purple'
        (x, y) = coordinates(self.Dx, self.N)
        plt.contour(x.T, y.T, self.topography.T, colors='black', alpha=0.75)
        plt.contourf(x.T, y.T, self.q.T, norm=log_norm, cmap=col_map, levels=75)
        plt.colorbar(cm.ScalarMappable(norm=log_norm, cmap=col_map), ax=plt.gca())
        self.streamfunction(t)
        self.velocity(t)
        plt.streamplot(x.T, y.T, self.v_x.T, self.v_y.T, color=streamplot_color)
        theta = np.linspace(0, (self.N-1) * self.Dx, num=5)
        labels = ['$0$', '$\pi/2$', '$\pi$', '$3\pi/2$', '$2\pi$']
        plt.xticks(theta, labels)
        plt.yticks(theta, labels)
        plt.xlim(0, (self.N-1) * self.Dx)
        plt.ylim(0, (self.N-1) * self.Dx)
        if simul is not None:
            plt.title(f'Vorticity  $|$  t={simul.time:.3f}')
        plt.gca().set_aspect('equal')
        if print_fig and simul:
            plt.savefig(simul.frames_dir / f'{(t+simul.t0):05}.png', dpi=200)
        else:
            plt.show()
        plt.close()
    
    def __neg__(self):
        neg_self = self.__class__(vorticity = -self.q, topography=self.topography)
        if self.psi is not None:
            neg_self.psi = -self.psi
        if self.v_x is not None:
            (neg_self.v_x, neg_self.v_y) = (-self.v_x, -self.v_y)
        return neg_self
    
    def __add__(self, other):
        sum_ = self.__class__(vorticity = self.q + other.q, topography=self.topography)
        if (self.psi is not None) and (other.psi is not None):
            sum_.psi = self.psi + other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (sum_.v_x, sum_.v_y) = (self.v_x + other.v_x, self.v_y + other.v_y)
        return sum_

    def __sub__(self, other):
        diff = self.__class__(vorticity = self.q - other.q, topography=self.topography)
        if (self.psi is not None) and (other.psi is not None):
            diff.psi = self.psi - other.psi
        if (self.v_x is not None) and (other.v_x is not None):
            (diff.v_x, diff.v_y) = (self.v_x - other.v_x, self.v_y - other.v_y)
        return diff



def energy_spectrum(
    v_x: np.ndarray,
    v_y: np.ndarray,
):
    n = v_x.shape[0]
    vt_square = np.abs(ft.rfft2(v_x))**2 + np.abs(ft.rfft2(v_y))**2
    k = modulus_k(n)
    k_range = np.floor((int(n/2) + 1) * np.sqrt(2))
    spectrum, _, _ = binned_statistic(k.flatten(), vt_square.flatten(), statistic='mean', bins=np.arange(k_range+1))
    return 2*np.pi * np.arange(k_range) * spectrum


def random_forcing(
    n: int,     # grid size
):
    strength = 1e+7
    k_F = n * 25/128
    # k_F = n / 8
    power_spectrum = forcing_spectrum(strength, k_F, n)
    random_phase = np.random.rand(n, int(n/2)+1)
    F_ft = power_spectrum * np.exp(2j*np.pi * random_phase)
    return ft.irfft2(F_ft)


@cache
def forcing_spectrum(
    strength: float,
    k_F: float,
    n: int,     # grid size
):
    band_size = 3.  # band_size = 2
    band_max = k_F + band_size/2.
    band_min = k_F - band_size/2.
    k = modulus_k(n)
    f = strength * np.logical_and(band_min < k, k < band_max)
    return np.sqrt(k * f / np.pi)


@cache
def modulus_k(
    n: int,     # grid size
):
    square_sum = np.square(np.arange(n))[:, None] + np.square(np.arange(int(n/2)+1))[None, :]
    mod_k = np.sqrt(square_sum)
    mod_k[int(n/2)+1:n, :] = mod_k[n-int(n/2)-1:0:-1, :]
    return mod_k


@cache
def zero_forcing(
    n: int,     # grid size
):
    return np.zeros((n, n))


def eig_function(
    h: np.ndarray,  # topography
    n: int,         # grid size
):
    from scipy.sparse.linalg import eigs
    q_of_psi = (laplacian_matrix(n) / h.flatten()).T
    e, ev = eigs(q_of_psi, 64, which='SM', tol=0, maxiter=1e14)
    del(q_of_psi)
    return e, ev


# @cache
# def laplacian_matrix(
#     n: int, # grid size
# ):
#     laplacian_1d = 2 * np.eye(n) - np.diag(np.ones(n-1), 1) - np.diag(np.ones(n-1), -1) - np.diag(np.ones(1), n-1) - np.diag(np.ones(1), 1-n)
#     laplacian_2d = np.kron(laplacian_1d, np.eye(n)) + np.kron(np.eye(n), laplacian_1d) 
#     return laplacian_2d


def laplacian_matrix(
    n: int, # grid size
):
    import scipy.sparse as sp
    laplacian_1d = sp.diags_array([2] + [-1]*4, offsets=[0, 1, -1, n-1, 1-n], shape=(n, n))
    laplacian_2d = sp.kron(laplacian_1d, sp.eye(n)) + sp.kron(sp.eye(n), laplacian_1d)
    return laplacian_2d


def relative_pos(
    x_ctr: float,   # center coordinates
    y_ctr: float,
    Dx: float,      # grid spacing
    n: int,         # grid size
):
    sd_len = n * Dx
    pos = np.arange(n) * Dx
    ctr = sd_len / 2
    x_dist = (pos - x_ctr + ctr) % sd_len - ctr
    y_dist = (pos - y_ctr + ctr) % sd_len - ctr
    dist2ctr = np.sqrt(np.square(x_dist)[:, None] + np.square(y_dist)[None, :])
    angle = np.arctan2(y_dist[None, :], x_dist[:, None])
    return dist2ctr, angle


def inv_laplacian2d(
    omega: np.ndarray,
    Dx: float,  # grid spacing
):
    omega_ft = ft.rfft2(omega)
    psi_ft = omega_ft / laplacian_eig(omega_ft.shape, Dx)
    return ft.irfft2(psi_ft)


@cache
def laplacian_eig(
    shape: tuple,
    Dx: float,  # grid spacing
):
    cos_sum = np.cos(np.arange(shape[0]) * Dx)[:, None] + np.cos(np.arange(shape[1]) * Dx)[None, :]
    lap_eig = 2* (2 - cos_sum) / Dx**2
    lap_eig[0, 0] = np.inf
    return lap_eig


def derivative(
    f: np.ndarray,
    axis: int,
    Dx: float,  # grid spacing
):
    df = (np.roll(f, -1, axis=axis) - np.roll(f, +1, axis=axis)) / (2 * Dx)
    return df


def pseudo_laplacian2d(
    f: np.ndarray,
):
    return np.roll(f, -1, axis=0) + np.roll(f, +1, axis=0) + np.roll(f, +1, axis=1) + np.roll(f, -1, axis=1) - 4* f


def autocorrelation(
    f: np.ndarray,  # field
):
    f_T = ft.rfft2(f)
    return ft.irfft2(f_T * f_T.conjugate())
    

def pbc_dist(
    x: int, # position
    D: int, # domain size
):
    return min(x, D-x)


def measure_valley_dist(
    t: np.ndarray,  # topography
    n: int,         # domain size
):
    C_self = autocorrelation(t)
    max_dist = int(np.round(n / np.sqrt(2)))
    C = np.zeros(max_dist + 1)
    count = np.zeros(max_dist + 1)

    for x in range(n):
        for y in range(n):
            dist = np.hypot(pbc_dist(x, n), pbc_dist(y, n)).round().astype(int)
            count[dist] += 1
            C[dist] += C_self[x, y]
    C /= count * np.square(t).sum()
    dist = np.arange(max_dist + 1)
    return C.sum(where=(dist > n/2)) / (max_dist - n/2)


def measure_concentration(
    f: np.ndarray,  # field
    n: int,         # domain size
):
    f_flt = gaussian_filter(f, sigma=n/50, truncate=2., mode='wrap')
    L2_measure = np.square(f_flt).sum()
    L1_measure = np.abs(f_flt).sum()
    concentration =  L2_measure * n**2 / L1_measure**2
    return 1 - 1 / concentration


def zero_coord( # finding combined zeros in the square domain [a, b]^2
    f_x: np.ndarray,
    f_y: np.ndarray,
    a: int, # interval left endpoint
    b: int, # interval right endpoint
):
    coor = np.arange(a, b-1)
    (X, Y) = np.meshgrid(coor, coor, indexing='ij')
    is_zero_fx_y = f_x[a:b, a:b-1] * f_x[a:b, a+1:b] <= 0
    is_zero_fy_x = f_y[a:b-1, a:b] * f_y[a+1:b, a:b] <= 0
    is_zero_f = np.logical_and(np.logical_or(is_zero_fx_y[0:-1, :], is_zero_fx_y[1:, :]), np.logical_or(is_zero_fy_x[:, 0:-1], is_zero_fy_x[:, 1:]))
    (zero_x, zero_y) = (0.,0.)
    count = 0
    for x, y in zip(X[is_zero_f], Y[is_zero_f]):
        (zero_x, zero_y) = (zero_x + x+0.5, zero_y + y+0.5)
        count = count + 1
    if count != 0:
        return zero_x/count, zero_y/count
    else:
        return (a+b)/2, (a+b)/2


def avg_coord( # averaging in the square domain [a, b]^2
    f: np.ndarray,      # weights
    a: int, # interval left endpoint
    b: int, # interval right endpoint
):
    (x, y) = coordinates()
    avg_x = np.average(x[a:b, a:b], weights=f[a:b, a:b])
    avg_y = np.average(y[a:b, a:b], weights=f[a:b, a:b])
    return avg_x, avg_y


def neighboring_clusters(
    x: int,         # point coordinates
    y: int,
    f: np.ndarray,  # field
    n: int,         # domain size
):
    neighbors = []
    if x != 0:
        c = f[x-1, y]
        if c >= 0:
            neighbors.append(c)
    if y != 0:
        c = f[x, y-1]
        if c >= 0 and c not in neighbors:
            neighbors.append(c)
    if x == n-1:
        c = f[0, y]
        if c >= 0 and c not in neighbors:
            neighbors.append(c)
    if y == n-1:
        c = f[x, 0]
        if c >= 0 and c not in neighbors:
            neighbors.append(c)
    return sorted(neighbors)


def cluster_ones(
    f: np.ndarray,  # field
    n: int,         # domain size
):
    cluster = []
    f_c = -np.ones((n, n)).astype(int)
    for x in range(n):
        for y in range(n):
            if f[x, y] == 1:
                neighbors = neighboring_clusters(x, y, f_c, n)
                if len(neighbors) == 0:
                    f_c[x, y] = len(cluster)
                    cluster.append([[x,y]])
                else:
                    f_c[x, y] = neighbors[0]
                    cluster[neighbors[0]].append([x,y])
                if len(neighbors) > 1:
                    for i in range(1, len(neighbors)):
                        for node in cluster[neighbors[i]]:
                            f_c[tuple(node)] = neighbors[0]
                        cluster[neighbors[0]].extend(cluster[neighbors[i]])
                        del cluster[neighbors[i]]
                        f_c[f_c > neighbors[i]] -= 1
                        for j in range(i+1, len(neighbors)):
                            neighbors[j] -= 1
    return sorted(cluster, key=len, reverse=True)


def largest_valley(
    topography: np.ndarray,
    n: int,     # domain size
    threshold = 0.2,
):
    valley_thresh = threshold * topography.max() + (1 - threshold) * topography.min()
    valleys = np.where(topography < valley_thresh, np.ones((n, n)), np.zeros((n, n)))
    cluster = cluster_ones(valleys, n)
    rel_size = len(cluster[0]) / n**2
    x_avg = y_avg = count = 0
    for point in cluster[0]:
        x_avg += point[0]
        y_avg += point[1]
        count += 1
    x_avg /= count
    y_avg /= count
    # print(f'The largest valley is located at ({x_avg:.1f}, {y_avg:.1f}) and fills about {rel_size*100:.1f}% of the domain.')
    return rel_size, (x_avg, y_avg)


def optimal_rescale(f, g):
    from scipy.optimize import minimize
    def Q(alpha, f, g):
        return np.square(g - alpha*f).sum()
    def Q_Jac(alpha, f, g):
        return (f * (alpha*f - g)).sum()
    def Q_Hess(alpha, f, g):
        return np.square(f).sum()
    alpha0 = (g.max() / f.max() + g.min() / f.min()) / 2
    result = minimize(fun=Q, jac=Q_Jac, hess=Q_Hess, x0=alpha0, args=(f, g), method='Newton-CG')
    if result['success'] is True:
        return result['x']
    else:
        return None


def state_diff(f, g):
    if (f * g).sum() < 0:
        g = -g
    alpha = optimal_rescale(f, g)
    if alpha is not None:
        print(f'The relative difference between states is {abs(alpha*f - g).sum() / abs(g).sum()}.')
        plt.contourf(alpha*f.T - g.T, levels=50)
        plt.colorbar()
        plt.show()
    else:
        print('Error: could not compare states.')


def lamb_dipole(
    Dx: float,  # grid spacing
    n: int,     # grid size
    orient = np.pi/2,
    U = 2.,
):
    radius = 0.73
    c_0 = 3.83170597020751231
    K = c_0 / radius
    C_L = - 2. * K * U / sp.j0(c_0)
    ctr = int(n/2) - 0.5
    dist2axis = np.arange(n) - ctr
    dist2ctr = np.sqrt(np.square(dist2axis)[:, None] + np.square(dist2axis)[None, :]) * Dx
    angle = np.arctan2(dist2axis[None, :], dist2axis[:, None])
    dipole = C_L * sp.j1(K*dist2ctr) * np.sin(angle - orient)
    dipole[dist2ctr > radius] = 0
    return dipole


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


def find_highest_numbered_npz(
    directory,
):
    max_number = -1
    max_file = None
    for file in directory.glob('q_*.npz'):
        try:
            file_number = int(file.stem[2:])
        except ValueError:
            print(f'<{filename}.npz> does not have a valid name.')
            pass
        else:
            if file_number > max_number:
                max_number = file_number
                max_file = file.name
    return max_file, max_number


def open_folder(
    folder,
    overwrite = False,
):
    if folder.exists():
        if overwrite:
            for file in folder.iterdir():
                if not file.is_dir():
                    file.unlink()
    else:
        folder.mkdir()


@cache
def coordinates(
    Dx: float,  # grid spacing
    n: int,     # grid size
):
    x = np.arange(0, n) * Dx
    y = np.arange(0, n) * Dx
    return np.meshgrid(x, y, indexing='ij')