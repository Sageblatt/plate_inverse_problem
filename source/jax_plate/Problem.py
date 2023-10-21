import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from typing import Callable
import warnings

import os
import sys
import json

from .Accelerometer import Accelerometer, AccelerometerParams
from .Material import Material, MaterialParams, ATYPES
from .Geometry import Geometry, GeometryParams

from .pyFFInterface import getOutput, evaluateMatrixAndRHS, processFFOutput
from .Utils import get_source_dir

from jax.config import config
config.update("jax_enable_x64", True)


class Problem:
    """
    Defines the geometry and those of the parameters that are known before the
    experiment. Stores FEM matrices on GPU, produces differentiable jax functions.
    """
    def __init__(self,
                 geometry: Geometry = None,
                 material: Material = None,
                 accel: Accelerometer = None,
                 ref_fr: tuple[np.ndarray, np.ndarray] = None,
                 *,
                 spath: str | os.PathLike = None):
        """
        Constructor method.

        Parameters
        ----------
        geometry : Geometry
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
        spath: str | os.PathLike, optional
            A path to a setup folder. Relative path can be used to locate
            folders within `JAX_PLATE_SOURCE_DIR/setups`. Setup folder should
            contain `setup.json` file, that contains any of 'geometry',
            'accelerometer' and 'material' entities with corresponding
            parameters, so the Geometry, Accelerometer and Material object can
            be created. If any of these entities is missing then corresponding
            argument should be provided.

        Returns
        -------
        None

        """
        if (geometry, accel, material, spath) == (None, ) * 4:
            raise ValueError('Cannot create a Problem object without arguments.')

        # Branch for creation without spath arg
        if spath is None:
            if None in (geometry, accel, material):
                raise ValueError('Cannot create a Problem object without `spath` '
                                 'argument if any of `geometry`, `accel`, '
                                 '`material` arguments is `None`.')

            self.geometry = geometry
            self.material = material
            self.accelerometer = accel

        # Branch for creation with spath arg
        else:
            if not isinstance(spath, str | os.PathLike):
                raise TypeError('Argument `spath` should have one '
                                'of the following types: str | os.PathLike, not '
                                f'{type(spath)}.')

            if not os.path.isabs(spath):
                spath = os.path.join(get_source_dir(), 'setups', spath)

            if not os.path.exists(spath):
                raise ValueError(f'Path of the setup {spath} does not exist.')

            elif not os.path.isdir(spath):
                raise ValueError(f'Selected path {spath} is not a directory.')

            setup_fpath = os.path.join(spath, 'setup.json')

            if not os.path.exists(setup_fpath):
                raise FileNotFoundError(f'`setup.json` file was not found in '
                                        f'setup directory {spath}.')

            with open(setup_fpath, 'r') as file:
                setup_params = json.load(file)

            param_dict = {'accelerometer': (Accelerometer, AccelerometerParams),
                          'material': (Material, MaterialParams)}

            # Try to read material and accel from setup.json if possible
            for key in param_dict:
                if key not in setup_params:
                    continue

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

            # Override material and accel if explicit argument is given
            if material is not None:
                self.material = material

            if accel is not None:
                self.accelerometer = accel

            # Try to read geometry from setup.json
            if 'geometry' in setup_params:
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

            # Override geometry if explicit argument is given
            if geometry is not None:
                self.geometry = geometry

            # Try to load reference frequency response if possible
            freq_file = os.path.join(spath, 'freqs.npy')
            if os.path.exists(freq_file):
                amp_file = os.path.join(spath, 'amp.npy')

                freqs = np.load(freq_file)
                amp = np.load(amp_file)

                ph_path = os.path.join(spath, 'phase.npy')

                if os.path.exists(ph_path):
                    phase = np.load(ph_path)

                else:
                    phase = np.zeros_like(amp)

                self.reference_fr = (freqs, amp * np.exp(1j * phase))

            if None in (self.accelerometer, self.geometry, self.material):
                raise RuntimeError('One of the `geometry`, `accelerometer`, '
                                   '`materials` attributes was not provided '
                                   'in setup.json nor as an argument.')
        try:
            if self.material.atype == 'isotropic':
                self.nu = self.material.E / (2.0 * self.material.G) - 1.0
                self.D = (self.material.E * self.geometry.height**3 /
                          (12.0 * (1.0 - self.nu**2)))
                self.beta = self.material.beta
                self.parameters = jnp.array([self.D, self.nu, self.beta])


            elif self.material.atype == 'orthotropic':
                self.nu21 = self.material.E1/self.material.E2 * self.material.nu12

                self.D11 = (self.material.E1 * self.geometry.height**3 /
                          (12 * (1 - self.material.nu12*self.nu21)))
                self.D66 = self.material.G12*self.geometry.height**3/12
                self.beta = self.material.beta
                self.parameters = jnp.array([self.D11, self.material.nu12,
                                             self.material.E1/self.material.E2,
                                             self.D66, self.beta])

            else:
                raise NotImplementedError(f'Only {ATYPES.keys()} atypes are supported.')

        except TypeError:
            warnings.warn('Some elastic moduli of a material were not provided, '
                          'solving forward problem as standalone will not be '
                          'possible.', RuntimeWarning)

        if ref_fr is not None:
            self.reference_fr = ref_fr

        self.e = self.geometry.height / 2.0
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

    def getEG(self, D: float = None, nu: float = None) -> tuple[float, float]:
        """
        Get Young's modulus E and shear modulus G for an isotropic material
        from flexural rigidity D and Poisson's ratio nu. If one of the arguments
        is None tries to get values from stored in Problem object ones.

        Parameters
        ----------
        D : float
            Flexural rigidity of a plate.

        nu : float
            Poisson's ratio of a plate.

        Returns
        -------
        tuple[float, float]
            Young's modulus E and shear modulus G in a tuple.

        """
        if self.material.atype != 'isotropic':
            raise ValueError('Cannot define E and G of a non-isotropic material.')

        if None in (D, nu):
            try:
                D, nu = self.parameters[:2]
            except AttributeError:
                raise RuntimeError('Cannot get E, G parameters from function '
                                   'args or Problem attributes.')

        E = 12 * D * (1 - nu ** 2) / self.geometry.height ** 3
        return E, E / (2 * (1 + nu))

    def getOModuli(self, D11: float = None,
                   nu12: float = None,
                   E_rat: float = None,
                   D66: float = None) -> tuple[float, float, float, float]:
        """
        Get E_1, E_2, nu_12 and G_12 moduli for an orthotropic material from
        D_11, nu_12, E_1 / E_2 ratio and D_66. If one of the arguments
        is None tries to get values from stored in Problem object ones.

        Parameters
        ----------
        D11 : float, optional
        nu12 : float, optional
        E_rat : float, optional
        D66 : float, optional

        Returns
        -------
        E_1 : float
        E_2 : float
        nu_12 : float
        G_12 : float

        """
        if self.material.atype != 'orthotropic':
            raise ValueError('Cannot define E1, E2, G12, nu12 of a '
                             'non-orthotropic material.')

        if None in (D11, nu12, E_rat, D66):
            try:
                D11, nu12, E_rat, D66 = self.parameters[0:4]
            except AttributeError:
                raise RuntimeError('Cannot get E, G parameters from function '
                                   'args or Problem attributes.')

        nu21 = E_rat * nu12
        E1 = D11 * 12 * (1 - nu12*nu21) / self.geometry.height**3
        return E1, E1 / E_rat, nu12, D66 * 12.0 / self.geometry.height**3


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
