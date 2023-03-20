import jax.numpy as jnp
import numpy as np
from jax_plate.Problem import Problem
from jax_plate.Utils import plot_fr
from jax_plate.ParamTransforms import isotropic
from jax_plate.Optimizers import optimize_trust_region
import matplotlib.pyplot as plt


correction = [0.0, 0.0, 0.0]
N_skip = 200
PLOT = False
SAVE = True
N_STEPS = 250
DELTA_MAX = 0.1

def main():
    from time import gmtime, strftime
    start_time = strftime("%Y_%m_%d_%H_%M_%S", gmtime())

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
    start_params = params*(jnp.array(correction) + 1.)


    with np.printoptions(formatter={'float': lambda x: f'{x:.6f}'}):
        print('Params before:', start_params)
        print('RE before local optimization:', (start_params - params)/params)


    # p = Problem("./edps/real_shifted_strip_fe.edp", h, rho, accelerometer_params)
    p = Problem("./edps/ideal_shifted_strip_fe.edp", h, rho, accelerometer_params)
    fr_func = p.getFRFunction(isotropic)

    freqs = np.load('../data/processed_afcs/afc_x_fe.npy')[::N_skip]
    freqs = jnp.array(freqs)
    print(f'{freqs.size=}')
    afc_mag = np.load('../data/processed_afcs/afc_mag_fe.npy')[::N_skip]
    afc_ph = np.load('../data/processed_afcs/afc_ph_fe.npy')[::N_skip]
    ref_afc = afc_mag*np.exp(1.j*afc_ph - beta)
    ref_afc = jnp.array([np.real(ref_afc), np.imag(ref_afc)]).T

    start_afc = fr_func(freqs, start_params)

    from time import perf_counter

    loss_function = p.getLossFunction(isotropic, freqs, ref_afc, 'MSE_AFC')
    print('Starting optimization.')
    t = perf_counter()
    opt_result = optimize_trust_region(loss_function, jnp.array(start_params), N_steps=N_STEPS, delta_max=DELTA_MAX, eta=0.15)
    dt = perf_counter() - t

    print('Elapsed time:', dt/60, 'min')

    with np.printoptions(formatter={'float': lambda x: f'{x:.6f}'}):
        print('Params after:', opt_result.x)
        print('RE after optimization (MSE)', (opt_result.x - params)/params)
        print("Status: ", opt_result.status)

    if PLOT:
        optimized_afc = fr_func(freqs, opt_result.x)
        fig, axs = plot_fr(freqs, start_afc, label='Initial')
        plot_fr(freqs, optimized_afc, label='Optimized', fig=fig)
        plot_fr(freqs, ref_afc, label='Reference (experimental)', linestyle='--', fig=fig)
        plt.show()

    if SAVE:
        np.savetxt(f'./optimization/{start_time}',
                   np.array([opt_result.x, opt_result.status]))

if __name__ == "__main__":
    main()