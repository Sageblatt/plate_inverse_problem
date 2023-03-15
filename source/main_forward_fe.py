import jax
import jax.numpy as jnp
import numpy as np
from numpy import pi, sqrt
from sys import exit, path
from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
from jax_plate.Optimizers import optimize_trust_region, optimize_gd
import matplotlib.pyplot as plt


# Steel plate
rho = 7920. # [kg/m^3]
E = 198*1e9 # [Pa]
# G = 77*1e9 # General steel
G = 68.5*1e9 # our steel type
h = 2.0e-3 # [m]
# this value i don't know
beta = .003 # loss factor, [1]

accelerometer_params = {'radius': 3.7e-3, 'mass': 1.7e-3}

nu = E/(2.*G) - 1.# [1]
D = E*h**3/(12.*(1. - nu**2))

if not 0 < nu < 0.5:
    print(f'Not real nu: {nu}!')
    exit(0)

params = jnp.array([D, nu, beta])

# p = Problem("./edps/real_shifted_strip_fe.edp", h, rho, accelerometer_params)
p = Problem("./edps/ideal_shifted_strip_fe.edp", h, rho, accelerometer_params)
getAFC = p.getAFCFunction(isotropic_to_full)

N_skip = 1
freqs = np.load('../data/processed_afcs/afc_x_fe.npy')[::N_skip]
print(f'{freqs.size=}')
freqs = jnp.array(freqs)
afc_mag = np.load('../data/processed_afcs/afc_mag_fe.npy')[::N_skip]
afc_ph = np.load('../data/processed_afcs/afc_ph_fe.npy')[::N_skip]
ref_afc = afc_mag*np.exp(1.j*afc_ph - beta)
ref_afc = jnp.array([np.real(ref_afc), np.imag(ref_afc)]).T

num_freqs = np.linspace(freqs[0], freqs[-1], 900)
num_afc = getAFC(num_freqs, params)

fig = plt.figure(figsize=(9.0, 6.0), dpi=400)
plt.plot(num_freqs, np.linalg.norm(num_afc, axis=1, ord=2), label='Computed')
plt.plot(freqs, np.linalg.norm(ref_afc, axis=1, ord=2), 'v-', ms=2, lw=0, label='Experimental')

plt.grid('on')
ax = plt.gca()
ax.set_yscale('log')
ax.legend()
ax.set_xlim([40, 1200])
ax.set_xlabel(r"$\nu,\ Hz$")
ax.set_ylabel(r"$\|u\|$")

# plt.savefig('fe.png', bbox_inches='tight')
plt.show()
