import numpy as np
import scipy as sp

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from jax.config import config

config.update("jax_enable_x64", True)

from .pyFFInterface import *


class Problem:
    """Defines the geometry and those of the parameters that are known before the experiment. 
        Stores FEM matrices on GPU, produces differentiable jax functions.

    :param path_to_edp: Path to .edp file which is used to interact with FreeFem++. 
        As for now, the geometry, mesh and  test point are also defined there. 
        In future, we'll fix one certain .edp file and change this arg to something line <<path_to_geometry>>
    :type path_to_edp: str
    :param thickness: specimen thickness [m]
    :type thickness: double
    :param density: specimen density [kg/m^3]
    :type thickness: double
        """

    def __init__(
        self,
        path_to_edp: str,
        thickness: np.float64,
        density: np.float64,
        accelerometer_param: dict = None,
    ):
        """Constructor method"""
        self.h = thickness
        self.e = thickness / 2.0
        self.rho = density

        processed_ff_output = processFFOutput(getOutput(path_to_edp))

        # Loading necessary data from model to GPU
        self.Ks = jnp.array(processed_ff_output["Ks"], dtype=jnp.float64)
        self.fKs = jnp.array(processed_ff_output["fKs"])
        self.M = jnp.array(processed_ff_output["M"])
        self.fM = jnp.array(processed_ff_output["fM"])
        self.L = jnp.array(processed_ff_output["L"])
        self.fL = jnp.array(processed_ff_output["fL"])

        # Correction of the inertia matrices by the mass of the accelerometer
        # TODO get rid of copypasta programming
        MCorrection = jnp.array(processed_ff_output["MCorrection"])
        fMCorrection = jnp.array(processed_ff_output["fMCorrection"])
        LCorrection = jnp.array(processed_ff_output["LCorrection"])
        fLCorrection = jnp.array(processed_ff_output["fLCorrection"])
        # TODO load can be not in phase with the BC vibration
        # and both BC and load may have different phases in points
        # so both of them have to be complex in general
        # but it is too comlicated as for now
        self.fLoad = jnp.array(processed_ff_output["fLoad"])

        # Total (regular + rotational) inertia
        self.MInertia = self.rho * (self.M + 1.0 / 3.0 * self.e ** 2 * self.L)
        # BC term
        self.fInertia = self.rho * (self.fM + 1.0 / 3.0 * self.e ** 2 * self.fL)

        if accelerometer_param is not None:
            # TODO: check e vs 2e=h
            rho_corr = (
                accelerometer_param["mass"]
                / (np.pi * accelerometer_param["radius"] ** 2)
                / self.e
            )
            self.MInertia += rho_corr * (
                MCorrection + 1.0 / 3.0 * self.e ** 2 * LCorrection
            )
            self.fInertia += rho_corr * (
                fMCorrection + 1.0 / 3.0 * self.e ** 2 * fLCorrection
            )

        self.interpolation_vector = jnp.array(
            processed_ff_output["interpolation_vector"]
        )
        self.interpolation_value_from_bc = jnp.float32(
            processed_ff_output["interpolation_value_from_bc"]
        )
        # TODO decide if we need to calculate in f64 in jax
        self.test_point = processed_ff_output["test_point_coord"]
        self.constrained_idx = processed_ff_output["constrained_idx"]

    def getAFCFunction(self, params_frequency_dependent, batch_size=None):
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
            # solve for one frequency f (in [Hz])
            omega = 2.0 * np.pi * f
            e = self.e
            # params_to_physical is a function D_ij = D_ij(theta), beta_ij = beta_ij(theta)
            # theta is the set of parameters; for example see Utils.isotropic_to_full
            D, beta = params_frequency_dependent(params, omega)
            loss_moduli = beta * D

            # K_real = \sum K_ij*D_ij/(2.*e)
            # K_imag = \sum K_ij*D_ij/(2.*e)
            # Ks are matrices from eq (4.1.7)
            K_real = jnp.einsum(self.Ks, [0, ...], D, [0]) / 2.0 / e
            K_imag = jnp.einsum(self.Ks, [0, ...], loss_moduli, [0]) / 2.0 / e

            # f_imag = ..
            # fs are vectors from (4.1.11), they account for the Clamped BC (u = du/dn = 0)
            fK_real = jnp.einsum(self.fKs, [0, ...], D, [0]) / 2.0 / e
            fK_imag = jnp.einsum(self.fKs, [0, ...], loss_moduli, [0]) / 2.0 / e

            # Formulate system (A_real + i*A_imag)(u_real + i*u_imag) = (b_real + i*b_imag)
            # and create matrix from blocks for real solution
            # MInertia == rho*(M + 1/3 e^2 L) from
            A_real = -(omega ** 2) * self.MInertia + K_real
            A_imag = K_imag
            b_real = -(omega ** 2) * self.fInertia + fK_real + self.fLoad / 2.0 / e
            b_imag = fK_imag

            A = jnp.vstack(
                (jnp.hstack((A_real, -A_imag)), jnp.hstack((-A_imag, -A_real)))
            )
            b = jnp.concatenate((b_real, -b_imag))

            u = jsp.linalg.solve(A, b, check_finite=False)

            # interpolation_vector == c
            # interpolation_value_from_bc == c_0 from 4.1.18
            u_in_test_point = jnp.array(
                [self.interpolation_value_from_bc, 0.0]
            ) + self.interpolation_vector @ u.reshape(  # real, imag
                (-1, 2), order="F"
            )

            return u_in_test_point

        _get_afc = jax.jit(jax.vmap(_solve, in_axes=(0, None),))

        if batch_size is None:
            return _get_afc

        # Workaround to avoid memory error
        def _get_afc_batched(fs, params):
            N_omega = fs.shape[0]
            if batch_size >= N_omega:
                return _get_afc(fs, params)
            n_batches = (N_omega + batch_size - 1) // batch_size
            afc = _get_afc(fs[:batch_size], params)
            for i in range(1, n_batches - 1):
                afc = jnp.vstack(
                    (afc, _get_afc(fs[i * batch_size : (i + 1) * batch_size], params))
                )
            i += 1
            afc = jnp.vstack((afc, _get_afc(fs[i * batch_size :], params)))
            return afc

        return _get_afc_batched

    def getSolutionMatrices(self, D, beta):
        loss_moduli = beta * D
        e = self.e

        K_real = jnp.einsum(self.Ks, [0, ...], D, [0]) / 2.0 / e
        K_imag = jnp.einsum(self.Ks, [0, ...], loss_moduli, [0]) / 2.0 / e

        return K_real, K_imag, self.MInertia

    def getMSELossFunction(
        self, params_to_physical, frequencies, reference_afc, batch_size=None
    ):
        assert frequencies.shape[0] == reference_afc.shape[0]
        assert reference_afc.shape[1] == 2

        afc_function = self.getAFCFunction(params_to_physical, batch_size)

        def MSELoss(params):
            afc = afc_function(frequencies, params)
            return jnp.mean((afc - reference_afc) ** 2)

        return MSELoss

    def getRMSELossFunction(
        self, params_to_physical, frequencies, reference_afc, batch_size=None
    ):
        assert frequencies.shape[0] == reference_afc.shape[0]
        assert reference_afc.shape[1] == 2

        afc_function = self.getAFCFunction(params_to_physical, batch_size)

        def RMSELoss(params):
            afc = afc_function(frequencies, params)
            return jnp.mean(((afc - reference_afc) / reference_afc) ** 2)

        return RMSELoss

    # def getLossAndDerivatives(
    #    self, params_to_physical, frequencies, reference_afc, batch_size=None
    # ):
    #    _loss = self.getMSELossFunction(params_to_physical, frequencies, reference_afc)
    #    _grad = jax.grad(_loss)
    #    _hess = jax.hessian(_loss)

    #    if batch_size is None:
    #        return _loss, _grad, _hess

    #    N_omega = frequencies.shape[0]

    #    def loss_batched(params):
    #        loss = 0.
