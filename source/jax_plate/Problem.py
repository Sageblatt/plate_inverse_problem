import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp

from .pyFFInterface import *


class Problem:
    """Defines the geometry and those of the parameters that are known before the experiment. Stores FEM matrices, produces differentiable jax functions.

    :param path_to_edp: Path to .edp file which is used to interact with FreeFem++. 
        As for now, the geometry, mesh and  test point are also defined there. 
        In future, we'll fix one certain .edp file and change this arg to something line <<path_to_geometry>>
    :type path_to_edp: str
    :param thickness: specimen thickness [m]
    :type thickness: double
    :param density: specimen density [kg/m^3]
    :type thickness: double
        """

    MODULI_INDICES = ["11", "12", "16", "22", "26", "66"]

    def __init__(
        self, path_to_edp: str, thickness: np.float64, density: np.float64,
    ):
        """Constructor method"""
        self.h = thickness
        self.e = thickness / 2.0
        self.rho = density

        Ks, fs, M, L, interpC, test_point_coords = processFFOutput(
            getOutput(path_to_edp)
        )
        self.Ks = jnp.array(Ks)
        self.fs = jnp.array(fs)
        self.M = jnp.array(M)
        self.L = jnp.array(L)

    def getAFCFunction(self, params_to_physical):
        """Creates function to evaluate AFC 
            :param params_to_physical: Function that converts chosen model parameters to the physical parameters of the model,
                D_ij storage modulus [Pa], beta_ij loss factor [1], ij in [11, 12, 16, 22, 26, 66], in total 12 parameters.
                Implemented to handle isotropic/orthotropic/general anisotropic elasticity; scaling, shifting of parameters for better minimization, etc.
            :type params_to_physical: callable
            :return: getAFC(frecuencies, params)  functions that takes array of frequencies w[nW] [Hz] and the parameters theta 
                and returns the afc[nW, 2] array, where afc[0, :] is the real and afc[1, :] the imaginary part of complex amplitude in test point
            :rtype: callable
            """

        def _solve(f, params):
            # Pseudocode of function, TODO: implement:D
            omega = 2.0 * np.pi * f
            # D_ij, beta_ij = params_to_physical(x)
            # K_real = -rho*omega**2*(M + 0.5*e**2) + \sum K_ij*D_ij/(2.*e)
            # f_real = ..
            # K_imag = \sum K_ij*D_ij*beta_ij/(2.*e)
            # f_imag = ..
            # construct block matrix
            # [u, v] = block matrix^-1 *[f_real, f_imag]
            # return [c.T@u, c.T@v]
            return jnp.array([0.0, 0.0])  # STUB

        return jax.jit(jax.vmap(_solve, in_axes=(0, None), ))

    def getMSELossFunction(self, params_to_physical, frequencies, reference_afc):
        assert frequencies.shape[0] == reference_afc.shape[0]
        assert reference_afc.shape[1] == 2

        afc_function = self.getAFCFunction(params_to_physical)

        def MSELoss(params):
            afc = afc_function(frequencies, params)
            return jnp.mean((afc - reference_afc) ** 2)

        return MSELoss
