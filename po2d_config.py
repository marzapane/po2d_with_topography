from numpy import pi

T = 1000000
N = 256
Re = 50000
ekman = 0.00002
# eps = 6.53 # actual dissipation
# eps = 0.00243
eps = 0.001
cfg_name = 'rand2'
sd_len = 2*pi # domain size
Dx = sd_len / N
Dt = 0.5 * Dx