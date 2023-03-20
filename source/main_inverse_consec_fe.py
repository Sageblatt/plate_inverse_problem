import jax.numpy as jnp
import numpy as np
from jax_plate.Problem import Problem
from jax_plate.Utils import plot_fr
from jax_plate.ParamTransforms import isotropic, FixedParameterFunction
from jax_plate.Optimizers import optimize_trust_region
import matplotlib.pyplot as plt


correction = [0.0, 0.0, 0.0]
N_skip = 200
PLOT = False
SAVE = True
N_STEPS1 = 25
DELTA_MAX1 = 0.1
N_STEPS2 = 15
DELTA_MAX2 = 0.1

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

    params1 = jnp.array([D, nu])
    start_params = params1*(jnp.array([correction[0], correction[1]]) + 1.)
    beta_start = beta * (correction[2] + 1)


    with np.printoptions(formatter={'float': lambda x: f'{x:.6f}'}):
        print('Params before:', start_params, beta_start)
        print('RE before local optimization:', (start_params - params1)/params1, (beta_start-beta)/beta)


    # p = Problem("./edps/real_shifted_strip_fe.edp", h, rho, accelerometer_params)
    p = Problem("./edps/ideal_shifted_strip_fe.edp", h, rho, accelerometer_params)
    fixed_beta = FixedParameterFunction(isotropic, 3, 2, beta_start)

    freqs = np.load('../data/processed_afcs/afc_x_fe.npy')[::N_skip]
    freqs = jnp.array(freqs)
    print(f'{freqs.size=}')
    afc_mag = np.load('../data/processed_afcs/afc_mag_fe.npy')[::N_skip]
    afc_ph = np.load('../data/processed_afcs/afc_ph_fe.npy')[::N_skip]
    ref_afc = afc_mag*np.exp(1.j*afc_ph - beta)
    ref_afc = jnp.array([np.real(ref_afc), np.imag(ref_afc)]).T

    from time import perf_counter

    print('Starting optimization.')
    loss_function1 = p.getLossFunction(fixed_beta, freqs, ref_afc, 'MSE_AFC')
    t = perf_counter()
    opt_result1 = optimize_trust_region(loss_function1, jnp.array(start_params), N_steps=N_STEPS1, delta_max=DELTA_MAX1, eta=0.15)

    params2 = jnp.array([beta])
    fixed_Dnu = FixedParameterFunction(isotropic, 3, (0, 1), opt_result1.x)
    loss_function2 = p.getLossFunction(fixed_Dnu, freqs, ref_afc, 'MSE_AFC')
    start_params2 = jnp.array([beta_start])
    opt_result2 = optimize_trust_region(loss_function2, start_params2, N_steps=N_STEPS2, delta_max=DELTA_MAX2, eta=0.15)

    dt = perf_counter() - t

    print('Elapsed time:', dt/60, 'min')

    with np.printoptions(formatter={'float': lambda x: f'{x:.6f}'}):
        print('Params after:', opt_result1.x, opt_result2.x)
        print('RE after optimization (MSE)', (opt_result1.x - params1)/params1, (opt_result2.x - params2)/params2)
        print(opt_result1.status, opt_result2.status)

    opt_result = np.array([float(opt_result1.x[0]), float(opt_result1.x[1]), float(opt_result2.x)])

    if PLOT:
        fr_func = p.getFRFunction(isotropic)
        start_afc = fr_func(freqs, np.array([float(start_params[0]), float(start_params[1]), beta_start]))
        optimized_afc = fr_func(freqs, opt_result)
        fig, axs = plot_fr(freqs, start_afc, label='Initial')
        plot_fr(freqs, optimized_afc, label='Optimized', fig=fig)
        plot_fr(freqs, ref_afc, label='Reference (experimental)', linestyle='--', fig=fig)

        # plt.plot(freqs, np.abs(ref_afc[:, 0] + 1.j*ref_afc[:, 1]))
        # # plt.plot(freqs, afc_mag)
        # ax = plt.gca()
        # ax.set_yscale('log')
        plt.show()

    if SAVE:
        np.savetxt(f'./optimization/consec_fe_{start_time}', opt_result)

if __name__ == "__main__":
    main()