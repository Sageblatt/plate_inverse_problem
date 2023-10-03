import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from typing import Callable

import os
import json

from .Accelerometer import Accelerometer, AccelerometerParams
from .Material import Material, MaterialParams
from .Geometry import Geometry, GeometryParams

from .pyFFInterface import getOutput, evaluateMatrixAndRHS, processFFOutput

from jax.config import config
config.update("jax_enable_x64", True)



class Problem:
    """
    Defines the geometry and those of the parameters that are known before the
    experiment. Stores FEM matrices on GPU, produces differentiable jax functions.
    """
    def __init__(self,
                 setup_path_or_geometry: str | os.PathLike | Geometry,
                 material: Material = None,
                 accel: Accelerometer = None,
                 ref_fr: tuple[np.ndarray, np.ndarray] = None): # TODO: independent from geometry import from setup
        """
        Constructor method.

        Parameters
        ----------
        setup_path_or_geometry : str | os.PathLike | Geometry
            A path to a setup folder or a Geometry object.
            Setup folder should contain setup.json file, that contains 'geometry',
            'accelerometer' and 'material' entities with corresponding parameters,
            so the Geometry, Accelerometer and Material object can be created.
            If this argument is the Geometry object, then `material` and `accel`
            arguments become mandatory.
        material : Material, optional
            Material object, has to contain `density`, `atype`, `transform`
            attributes. The default is None.
        accel : Accelerometer, optional
            Accelerometer object, has to contain `mass` and `radius` attributes.
            The default is None.
        ref_fr : tuple[np.ndarray, np.ndarray], optional
            Reference frequency response obtained from experiment,
            first element is an array of frequencies, second one is an array of
            complex amplitudes. The default is None.

        Returns
        -------
        None

        """

        if isinstance(setup_path_or_geometry, str | os.PathLike):
            if not os.path.exists(setup_path_or_geometry):
                raise ValueError(f'Path of the setup {setup_path_or_geometry} '
                                 'does not exist.')

            elif not os.path.isdir(setup_path_or_geometry):
                raise ValueError(f'Selected path {setup_path_or_geometry} is '
                                 'not a directory.')

            setup_fpath = os.path.join(setup_path_or_geometry, 'setup.json')

            if not os.path.exists(setup_fpath):
                raise FileNotFoundError(f'`setup.json` file was not found in '
                                        'setup directory '
                                        f'{setup_path_or_geometry}.')

            with open(setup_fpath, 'r') as file:
                setup_params = json.load(file)

            try:
                param_dict = {'accelerometer': (Accelerometer, AccelerometerParams),
                              'material': (Material, MaterialParams)}

                for key in param_dict:
                    if isinstance(setup_params[key], str):
                        setattr(self, key,
                                param_dict[key][0](setup_params[key]))

                    elif isinstance(setup_params[key], dict):
                        setattr(self, key,
                                param_dict[key][0](param_dict[key][1](**setup_params[key])))

                    else:
                        raise TypeError(f'In file {setup_fpath} key "{key}" '
                                        'should have a value with type `str` or'
                                        ' `dict`.')

                if 'template' in setup_params['geometry']:
                    templ = setup_params['geometry']['template']
                    del setup_params['geometry']['template']

                    self.geometry = Geometry(templ,
                                             accelerometer=self.accelerometer,
                                             params=GeometryParams(**setup_params['geometry']))

                elif 'edp' in setup_params['geometry']:
                    edp_file = setup_params['geometry']['edp']
                    del setup_params['geometry']['edp']

                    if 'length' in setup_params['geometry']:
                        self.geometry = Geometry(edp_file,
                                                 accelerometer=self.accelerometer,
                                                 params=GeometryParams(**setup_params['geometry']))
                    else:
                        self.geometry = Geometry(edp_file,
                                                 accelerometer=self.accelerometer,
                                                 height=setup_params['geometry']['height'])

                else:
                    raise ValueError('Cannot create Geometry object, file '
                                     f'{setup_fpath} should contain `template` '
                                     'or `edp` keyword inside `geometry`.')

                if material is not None:
                    self.material = material

                if accel is not None:
                    self.accelerometer = accel

            except KeyError as ex:
                raise KeyError(f'Key {ex} was not found in a setup '
                               f'file {setup_fpath}.')

            freq_file = os.path.join(setup_path_or_geometry, 'freqs.npy')
            if os.path.exists(freq_file) and ref_fr is None:
                amp_file = os.path.join(setup_path_or_geometry, 'amp.npy')

                freqs = np.load(freq_file)
                amp = np.load(amp_file)

                ph_path = os.path.join(setup_path_or_geometry, 'phase.npy')

                if os.path.exists(ph_path):
                    phase = np.load(ph_path)

                else:
                    phase = np.zeros_like(amp)

                ref_fr = (freqs, amp * np.exp(1j * phase))



        elif isinstance(setup_path_or_geometry, Geometry):
            self.geometry = setup_path_or_geometry
            if material is None or accel is None:
                raise ValueError('Both `material` and `accelerometer` arguments '
                                 'should not be `None` when creating a Problem '
                                 'from Geometry object.')

            self.material = material
            self.accelerometer = accel

        else:
            raise TypeError('Argument `setup_path_or_geometry` should have one '
                            'of the following types: str | os.PathLike | '
                            f'Geometry, not {type(setup_path_or_geometry)}.')

        if self.material.atype == 'isotropic':
            if (self.material.E is not None and
               self.material.G is not None and
               self.material.beta is not None):
                   self.nu = self.material.E / (2.0 * self.material.G) - 1.0
                   self.D = (self.material.E * self.geometry.height**3 /
                             (12.0 * (1.0 - self.nu**2)))
                   self.beta = self.material.beta
                   self.parameters = jnp.array([self.D, self.nu, self.beta])

        if ref_fr is not None:
            self.reference_fr = ref_fr

        # OLD STARTS HERE
        self.h = self.geometry.height
        self.e = self.h / 2.0
        self.rho = self.material.density

        processed_ff_output = processFFOutput(getOutput(self.geometry.current_file))

        # Loading necessary data from model to GPU
        self.Ks = jnp.array(processed_ff_output["Ks"], dtype=jnp.float64)
        self.fKs = jnp.array(processed_ff_output["fKs"])
        self.M = jnp.array(processed_ff_output["M"])
        self.fM = jnp.array(processed_ff_output["fM"])
        self.L = jnp.array(processed_ff_output["L"])
        self.fL = jnp.array(processed_ff_output["fL"])

        # Correction of the inertia matrices by the mass of the accelerometer
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

        if self.accelerometer is not None:
            # TODO: check e vs 2e=h
            rho_corr = (
                self.accelerometer.mass
                / (np.pi * self.accelerometer.radius ** 2)
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

    def getFRFunction(self, params_to_physical = None, batch_size=None):
        """
        Creates a function to evaluate AFC.

        Parameters
        ----------
        params_to_physical : Callable, optional
            Function that converts chosen model parameters to the physical
            parameters of the model, D_ij storage modulus [Pa], beta_ij loss
            factor [1], ij in [11, 12, 16, 22, 26, 66], in total 12 parameters.
            Implemented to handle isotropic/orthotropic/general anisotropic
            elasticity; scaling, shifting of parameters for better minimization, etc..
            If not present, self.material.transform is used.
        batch_size : int, optional
            If present, optimization will use batches to avoid memory error.

        Returns
        -------
        callable
            Function that takes array of frequencies `omega` [Hz] and the
            parameters `theta` and returns the FR array of complex amplitude in
            test point.

        """
        if params_to_physical is None:
            params_to_physical = self.material.transform

        def _solve(f, params):
            # solve for one frequency f (in [Hz])
            omega = 2.0 * np.pi * f
            e = self.e
            # params_to_physical is a function D_ij = D_ij(theta), beta_ij = beta_ij(theta)
            # theta is the set of parameters; for example see Utils.isotropic_to_full
            D, beta = params_to_physical(params, omega)
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
            u_in_test_point = jnp.array([self.interpolation_value_from_bc, 0.0]) +\
            self.interpolation_vector @ u.reshape(  # real, imag
                (-1, 2), order="F"
            )

            return u_in_test_point[0] + 1j * u_in_test_point[1]

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

    def solveForward(self, freqs: np.ndarray) -> np.ndarray:
        """
        Solve forward problem for a given set of frequencies.

        Uses self.parameters as a parameter vector.

        Parameters
        ----------
        freqs : np.ndarray
            Set of frequencies, for which the complex amplitudes will be computed.

        Returns
        -------
        np.ndarray
            Array of complex amplitudes.

        """
        fr_func = self.getFRFunction()
        return fr_func(freqs, self.parameters)

    def getSolutionMatrices(self, D, beta):
        loss_moduli = beta * D
        e = self.e

        K_real = jnp.einsum(self.Ks, [0, ...], D, [0]) / 2.0 / e
        K_imag = jnp.einsum(self.Ks, [0, ...], loss_moduli, [0]) / 2.0 / e

        return K_real, K_imag, self.MInertia

    def getLossFunction(
        self,
        params_to_physical: Callable,
        frequencies: jax.Array,
        reference_fr: jax.Array,
        func_type: str, # Available options are: MSE, RMSE, MSE_AFC
        batch_size: int = None
    ) -> Callable:
        assert frequencies.shape[0] == reference_fr.shape[0]

        fr_function = self.getFRFunction(params_to_physical, batch_size)

        if func_type == "MSE":
            def MSELoss(params):
                fr = fr_function(frequencies, params)
                return jnp.mean(jnp.abs(fr - reference_fr) ** 2)

            return MSELoss

        elif func_type == "RMSE":
            def RMSELoss(params):
                fr = fr_function(frequencies, params)
                return jnp.mean(jnp.abs((fr - reference_fr) / reference_fr) ** 2)

            return RMSELoss

        elif func_type == "MSE_AFC":
            def MSE_AFCLoss(params):
                fr = fr_function(frequencies, params)
                return jnp.mean((jnp.abs(fr) - jnp.abs(reference_fr)) ** 2)

            return MSE_AFCLoss

        elif func_type == "MSE_LOG_AFC":
            def MSE_LOG_AFCLoss(params):
                fr = fr_function(frequencies, params)
                return jnp.mean((jnp.log(jnp.abs(fr)) -
                                 jnp.log(jnp.abs(reference_fr))) ** 2)

            return MSE_LOG_AFCLoss

        else:
            raise ValueError(f'Function type "{func_type}" is not supported!')

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
