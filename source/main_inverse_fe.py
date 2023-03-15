import jax
import jax.numpy as jnp
import numpy as np
from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
from jax_plate.Optimizers import optimize_trust_region, optimize_gd
import matplotlib.pyplot as plt

# Steel plate
rho = 7920. # [kg/m^3]
E = 198*1e9 # [Pa]
# G = 77*1e9 # [Pa] General steel
G = 68.5*1e9 # our steel type
h = 2.0e-3 # [m]
# this value i don't know
beta = .003 # loss factor, [1]

nu = E/(2.*G) - 1.# [1]
D = E*h**3/(12.*(1. - nu**2))

accelerometer_params = {'radius': 3.7e-3, 'mass': 1.7e-3}

params = jnp.array([D, nu, beta])
start_params = params*(jnp.array([0.2, 0.05, 99.]) + 1.)
# start_params = params


with np.printoptions(formatter={'float': lambda x: f'{x:.6f}'}):
    print('Params before:', start_params)
    print('RE before local optimization:', (start_params - params)/params)


# p = Problem("./edps/real_shifted_strip_fe.edp", h, rho, accelerometer_params)
p = Problem("./edps/ideal_shifted_strip_fe.edp", h, rho, accelerometer_params)
getAFC = p.getAFCFunction(isotropic_to_full)

N_skip = 200
freqs = np.load('../data/processed_afcs/afc_x_fe.npy')[::N_skip]
freqs = jnp.array(freqs)
print(f'{freqs.size=}')
afc_mag = np.load('../data/processed_afcs/afc_mag_fe.npy')[::N_skip]
afc_ph = np.load('../data/processed_afcs/afc_ph_fe.npy')[::N_skip]
ref_afc = afc_mag*np.exp(1.j*afc_ph - beta)
ref_afc = jnp.array([np.real(ref_afc), np.imag(ref_afc)]).T

start_afc = getAFC(freqs, start_params)

from time import perf_counter

loss_function = p.getMSELossFunction(isotropic_to_full, freqs, ref_afc)
print('Starting optimization.')
t = perf_counter()
opt_result = optimize_trust_region(loss_function, jnp.array(start_params), N_steps=50, delta_max=0.9, eta=0.15)
# opt_result = optimize_trust_region(loss_function, jnp.array(start_params), N_steps=250, delta_max=0.1, eta=0.15)
dt = perf_counter() - t

print('Elapsed time:', dt/60, 'min')


with np.printoptions(formatter={'float': lambda x: f'{x:.6f}'}):
    print('Params after:', opt_result.x)
    print('RE after optimization (MSE)', (opt_result.x - params)/params)

optimized_afc = getAFC(freqs, opt_result.x)
fig, axs = plot_afc(freqs, start_afc, label='Initial')
plot_afc(freqs, optimized_afc, label='Optimized', fig=fig)
plot_afc(freqs, ref_afc, label='Reference (experimental)', linestyle='--', fig=fig)

# plt.plot(freqs, np.abs(ref_afc[:, 0] + 1.j*ref_afc[:, 1]))
# # plt.plot(freqs, afc_mag)
# ax = plt.gca()
# ax.set_yscale('log')
plt.show()

from time import gmtime, strftime
np.savetxt(f'./optimization/{strftime("%Y_%m_%d_%H_%M_%S", gmtime())}', opt_result.x)
