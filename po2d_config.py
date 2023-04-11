import numpy as np

T = 200000
N = 256
Re = 80000
ekman = 0.0001
eps = 1.24 # actual dissipation
# eps = 0.00243
# eps = 0.02878
cfg_name = ''
sd_len = 2*np.pi # domain size
Dx = sd_len / N
Dt = 0.5 * Dx