import jax
import jax.numpy as jnp
import numpy as np
from numpy import pi, sqrt
from jax_plate.Problem import Problem
from jax_plate.Utils import *
from jax_plate.ParamTransforms import *
from jax_plate.Optimizers import optimize_trust_region, optimize_gd
import matplotlib.pyplot as plt
from sys import exit

# Titanium plate
rho = 4550. # [kg/m^3] # General titanium type
E = 106.*1e9 # [Pa] # General titanium type
G = 40.*1e9 # [Pa] # this value i don't know
h = 3.0e-3 # [m]
beta = .007 # loss factor, [1] # this value i don't know

accelerometer_params = {'radius': 3.7e-3, 'mass': 1.7e-3}

nu = E/(2.*G) - 1.# [1]
D = E*h**3/(12.*(1. - nu**2))

if not 0 < nu < 0.5:
    print(f'Not real nu: {nu}!')
    exit(0)

params = jnp.array([D, nu, beta])

# p = Problem("./edps/real_shifted_strip_ti.edp", h, rho, accelerometer_params)
p = Problem("./edps/ideal_shifted_strip_ti.edp", h, rho, accelerometer_params)
getAFC = p.getAFCFunction(isotropic_to_full)

N_skip = 1
freqs = np.load('../data/processed_afcs/afc_x_ti.npy')[::N_skip]
freqs = jnp.array(freqs)
print(f'{freqs.size=}')
afc_mag = np.load('../data/processed_afcs/afc_mag_ti.npy')[::N_skip]
afc_ph = np.load('../data/processed_afcs/afc_ph_ti.npy')[::N_skip]
ref_afc = afc_mag*np.exp(1.j*afc_ph - beta)
ref_afc = jnp.array([np.real(ref_afc), np.imag(ref_afc)]).T

num_freqs = np.linspace(freqs[0], freqs[-1], 500)
num_afc = getAFC(num_freqs, params)

fig = plt.figure(figsize=(6.0, 6.0), dpi=400)
plt.plot(num_freqs, np.linalg.norm(num_afc, axis=1, ord=2), label='Current')
plt.plot(freqs, np.linalg.norm(ref_afc, axis=1, ord=2), label='Reference', linestyle='-')

plt.grid('on')
ax = plt.gca()
ax.set_yscale('log')

plt.show()
