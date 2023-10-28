import os
import sys
import json
import warnings
from typing import Callable
from time import perf_counter, gmtime, strftime

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpy.typing as npt

from .Accelerometer import Accelerometer, AccelerometerParams
from .Material import Material, MaterialParams, ATYPES
from .Geometry import Geometry, GeometryParams

from .Input import Compressor
from .pyFFInterface import getOutput, evaluateMatrixAndRHS, processFFOutput
from .Utils import get_source_dir
from .Optimizers import optimize_trust_region, optimize_cd, optimize_gd, optimize_cd_mem2

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

    def getFRFunction(self, batch_size: int = None):
        """
        Creates a function to evaluate AFC.

        Parameters
        ----------
        batch_size : int, optional
            If present, optimization will use batches to avoid memory error.

        Returns
        -------
        callable
            Function that takes array of frequencies `omega` [Hz] and the
            parameters `theta` and returns the FR array of complex amplitude in
            test point.

        """
        def _solve(f, params):
            # solve for one frequency f (in [Hz])
            omega = 2.0 * np.pi * f
            e = self.e
            # transform is a function D_ij = D_ij(theta), beta_ij = beta_ij(theta)
            # theta is the set of parameters; for example see ParamTransforms.isotropic_to_full
            D, beta = self.material.transform(params, omega)
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

    def solveForward(self, freqs: np.ndarray, params=None) -> np.ndarray:
        """
        Solve forward problem for a given set of frequencies.

        Uses self.parameters as a parameter vector or a custom array specified
        in `params` argument.

        Parameters
        ----------
        freqs : np.ndarray
            Set of frequencies, for which the complex amplitudes will be computed.

        params : np.ndarray, optional
            Parameters to be used in forward problem, if provided.

        Returns
        -------
        np.ndarray
            Array of complex amplitudes.

        """
        if params is None:
            params = self.parameters

        params = jnp.array(params)

        fr_func = self.getFRFunction()
        return fr_func(freqs, params)

    def solveInverseLocal(self, x0: npt.ArrayLike,
                          loss_type: str,
                          optimizer: str,
                          compression: tuple[bool, int] = (False, 0),
                          comp_alg: int = 1,
                          ref_fr: tuple[np.ndarray, np.ndarray] = None,
                          use_rel: bool = False,
                          report: bool = True,
                          log: bool = True,
                          report_moduli: bool = True,
                          case_name: str = '',
                          uid: str = None,
                          extra_info: str = '',
                          **opt_kwargs):
        """
        Solve the local inverse problem for given initial guess.

        Parameters
        ----------
        x0 : numpy.typing.ArrayLike
            Initial guess for each parameter. If `use_rel` argument is `True`
            then `x0` describes a relative difference between initial guess and
            parameters provided in self.parameters.
        loss_type : str
            Type of a loss to be optimized. Available options are:
                - `MSE`, mean squared error.
                - `RMSE`, relative mean squared error.
                - `MSE_AFC`, mean squared error, amplitudes of frequency
                response are optimized only.
                - `MSE_LOG_AFC`, similar to `MSE_AFC`, but the logarithm of
                amplitudes is used.
        optimizer : str
            Type of an optimizing algorithm to be used. Available options are:
                - `trust_region`, second-order optimizer.
                See jax_plate.Optimizers.solve_trust_region.
                - `coord_descent`, coordinate descent.
                See jax_plate.Optimizers.optimize_cd.
                - `coord_descent_mem`, memory-efficient version of coordinate
                descent. See jax_plate.Optimizers.optimize_cd_mem.
                - `grad_descent`, gradient descent optimization.
                See jax_plate.Optimizers.optimize_gd.
        compression : tuple[bool, int], optional
            A tuple, in which the first element defines whether a compression
            algorithm for reference frequency response will be used or not.
            The second element defines amount of points in the dataset after
            compression. The default is (False, 0).
        comp_alg : int, optional
            If `compression[0]` is `True` defines the type of algorithm used
            for compression. See jax_plate.Input.Compressor for more details.
            The default is 1.
        ref_fr : tuple[np.ndarray, np.ndarray], optional
            Reference frequency response. If `None` the value
            from self.reference_fr will be used. The default is None.
        use_rel : bool, optional
            See `x0` argument description. The default is False.
        report : bool, optional
            Generate human-readable report with optimization parameters.
            The default is True.
        log : bool, optional
            Write all optimization steps to `.npz` archive. The default is True.
        report_moduli : bool, optional
            If `True` the information about physical elastic moduli will be
            included in the report (etc. Young's modulus and shear modulus for
            isotropic material). The default is True.
        case_name : str, optional
            Name which will be used as prefix to log and report filenames.
            The default is ''.
        uid : str, optional
            Unique id - a name which will be used as a body of log and report
            filenames. If `None` system date and time will be used. The default
            is None.
        extra_info : str, optional
            Additional info to include into the report. The default is ''.
        **opt_kwargs
            Optimizer keyword arguments.

        Returns
        -------
        Optimizers.optResult
            A namedtuple with optimization result and iteration history.

        """
        if ref_fr is None:
            ref_fr = getattr(self, 'reference_fr', None)
            if ref_fr is None:
                raise ValueError('Cannot solve inverse problem as `ref_fr` '
                                 'argument was not provided and the '
                                 "Problem object doesn't have a reference_fr "
                                 'attribute.')
            else:
                ref_fr = [*ref_fr]

        if not isinstance(compression, tuple):
            raise TypeError('`compression` argument should have a type `tuple`,'
                            f'not {type(compression)}.')

        elif len(compression) != 2:
            raise ValueError('`compression` tuple should have 2 elements, not '
                             f'{len(compression)}.')


        if compression[0]:
            comp = Compressor(ref_fr[0], ref_fr[1], compression[1], comp_alg)
            ref_fr[0], ref_fr[1] = comp(compression[1])

        loss = self.getLossFunction(ref_fr[0], ref_fr[1], loss_type)

        if use_rel:
            if getattr(self, 'parameters', None) is None:
                raise ValueError('Cannot use `x0` as relative coefficients of '
                                 'correction as Problem object has no '
                                 '`parameters` attribute.')

            else:
                x0 = jnp.array(self.parameters) * (jnp.array(x0) + 1)
        else:
            x0 = jnp.array(x0)

        if optimizer == 'trust_region':
            optimizer_func = optimize_trust_region

        elif optimizer == 'coord_descent':
            optimizer_func = optimize_cd

        elif optimizer == 'coord_descent_mem':
            optimizer_func = optimize_cd_mem2

        elif optimizer == 'grad_descent':
            optimizer_func = optimize_gd

        else:
            raise ValueError(f'Optimizer type `{optimizer}` is not supported!')

        t_start = perf_counter()
        result = optimizer_func(loss, x0, **opt_kwargs)
        t_end = perf_counter()
        elapsed = (t_end - t_start) / 60

        date_str = strftime("%d_%m_%Y_%H_%M_%S", gmtime())

        if uid is None:
            full_str = case_name + date_str

        else:
            full_str = case_name + uid

        if report:
            rel_err1 = 'Unknown'
            rel_err2 = 'Unknown'
            if getattr(self, 'parameters', None) is not None:
                params0 = np.array(self.parameters)
                rel_err1 = (params0 - np.array(x0)) / params0
                rel_err2 = (params0 - np.array(result.x)) / params0

                if report_moduli:
                    other_moduli0 = np.array(self.getPhysicalModuli(x0))
                    other_moduli = np.array(self.getPhysicalModuli(result.x))
                    other_moduli_real = np.array(self.getPhysicalModuli(params0))
                    om_r_err1 = np.array((other_moduli0 - other_moduli_real) / other_moduli_real)
                    om_r_err2 = np.array((other_moduli - other_moduli_real) / other_moduli_real)

            def a2s(s):
                if isinstance(s, str):
                    return s

                return np.array2string(np.array(s), separator=', ')

            mod_str1 = ''
            mod_str2 = ''
            if report_moduli:
                mod_str1 = (f'In physical moduli: {a2s(other_moduli0)}\n'
                            f'With rel. error: {a2s(om_r_err1)}\n')
                mod_str2 = (f'In physical moduli: {a2s(other_moduli)}\n'
                            f'With rel. error: {a2s(om_r_err2)}\n')


            rep_str = (f'{self.accelerometer}\n{self.material}\n{self.geometry}\n'
                       + extra_info +
                       f'Starting parameters: {a2s(x0)}.\n'
                       f'With relative error: {a2s(rel_err1)}.\n' +
                       mod_str1 +
                       f'Initial loss: {result.f_history[0]}.\n'
                       f'Elapsed time: {elapsed} min.\n'
                       f'After optimization: {a2s(result.x)}.\n'
                       f'With relative error: {a2s(rel_err2)}.\n' +
                       mod_str2 +
                       f'Resulting loss: {result.f}.\n'
                       f'Optimization status: {result.status}.\n'
                       f'Optimizer parameters: {opt_kwargs}\n')
            print(rep_str, end='')

            full_path = os.path.join(get_source_dir(), 'optimization',
                                     full_str + '.txt')
            with open(full_path, 'w+') as file:
                file.write(rep_str)

        if log:
            f_ = np.array(result.f_history + [result.f])
            x_ = np.array(result.x_history + [result.x])
            k_ = np.array([result.niter])
            np.savez_compressed(os.path.join(get_source_dir(), 'optimization',
                                full_str), x=x_, f=f_, k=k_)

        return result

    def getSolutionMatrices(self, D, beta):
        loss_moduli = beta * D
        e = self.e

        K_real = jnp.einsum(self.Ks, [0, ...], D, [0]) / 2.0 / e
        K_imag = jnp.einsum(self.Ks, [0, ...], loss_moduli, [0]) / 2.0 / e

        return K_real, K_imag, self.MInertia

    def getLossFunction(
        self,
        frequencies: jax.Array,
        reference_fr: jax.Array,
        func_type: str, # Available options are: MSE, RMSE, MSE_AFC, MSE_LOG_AFC
        batch_size: int = None
    ) -> Callable:
        assert frequencies.shape[0] == reference_fr.shape[0]
        fr_function = self.getFRFunction(batch_size)

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

    def getPhysicalModuli(self, params: np.ndarray | jax.Array) -> tuple:
        """
        Wrapper around Problem.getEG and other similar functions that
        automatically chooses the correct one depending in self.material.atype.

        Parameters
        ----------
        params : np.ndarray | jax.Array
            Parameters to be tranformed.

        Returns
        -------
        tuple
            Tuple of float containing physical moduli for given material.

        """
        if self.material.atype == 'isotropic':
            return self.getEG(params[0], params[1])

        elif self.material.atype == 'orthotropic':
            return self.getOModuli(params[0], params[1], params[2], params[3])

        else:
            return NotImplementedError(f'Only {ATYPES.keys()} atypes are supported.')

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
