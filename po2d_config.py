import numpy as np

reload_ = False
cfg_name = 'Ek1e-4,N256'
N = 256
sd_len = 2*np.pi # domain size
Dx = sd_len / N
Dt = 0.2 * Dx
Re = 50000
ekman = 0.0001
T = 2000