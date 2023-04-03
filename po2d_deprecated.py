@cache
def forcing_theta(
    k_F,
):
    k_range = int(np.ceil(k_F + 1))
    k = np.sqrt(np.square(np.arange(k_range))[:, None] + np.square(np.arange(k_range))[None, :])
    accept_k = np.logical_and(k > k_F-1, k < k_F+1)
    theta = 2*np.pi/N * np.arange(k_range)[:, None] * np.arange(N)[None, :]
    theta_x = theta[:, None, :] + np.zeros(k_range)[None, :, None]
    theta_y = theta[None, :, :] + np.zeros(k_range)[:, None, None]
    return theta_x[accept_k], theta_y[accept_k]

def rand_sine_forcing(
    strength = 0.2,
    k_F = N / 8,
):
    theta_x, theta_y = forcing_theta(k_F)
    phase_x = 2*np.pi * (np.random.rand(theta_x.shape[0]) - 0.5)
    phase_y = 2*np.pi * (np.random.rand(theta_y.shape[0]) - 0.5)
    rand_f = np.sin(theta_x[:, :, None] + phase_x[:, None, None]) * np.sin(theta_y[:, None, :] + phase_y[:, None, None])
    F = strength * np.sum(rand_f, axis=0)
    return F