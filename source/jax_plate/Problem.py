import os
import json
import warnings
from typing import Callable
from time import perf_counter, gmtime, strftime

import jax
import jax.numpy as jnp
import jax.experimental.sparse as sparse
import numpy as np
import numpy.typing as npt

import functools

from .Accelerometer import Accelerometer, AccelerometerParams
from .Material import Material, MaterialParams
from .Geometry import Geometry, GeometryParams

from .Input import Compressor
from .pyFFInterface import getOutput, processFFOutput
from .Utils import get_source_dir
from .Optimizers import optimize_trust_region, optimize_cd, optimize_gd, optimize_cd_mem2
from .Optimizers import optResult
from .Sparse import spsolve

from jax.config import config
config.update("jax_enable_x64", True)


class StaticNdArrayWrapper(np.ndarray):
    """
    Hashable numpy.ndarray to be used as static argument in jax.jit.
    """
    def __hash__(self):
        return 1

    def __eq__(self, other):
        return True


class Problem:
    """
    Defines the geometry and those of the parameters that are known before the
    experiment. Stores FEM matrices, produces differentiable jax functions.
    """
    def __init__(self,
                 geometry: Geometry = None,
                 material: Material = None,
                 accel: Accelerometer = None,
                 ref_fr: tuple[np.ndarray, np.ndarray] = None,
                 *,
                 cpu: int | None = 0,
                 spath: str | os.PathLike = None):
        """
        Constructor method.

        Parameters
        ----------
        geometry : Geometry
            If this argument is a Geometry object, then `material` and `accel`
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
        cpu : int | None, optional
            A number of CPU cores to use when solving forward problem for
            multiple frequencies. If value is `0` uses maximum amount of cores
            available. The default is 0.
        spath : str | os.PathLike, optional
            A path to a setup folder. Relative path can be used to locate
            folders within `JAX_PLATE_SOURCE_DIR/setups`. Setup folder should
            contain `setup.json` file, that contains any of 'geometry',
            'accelerometer' and 'material' entities with corresponding
            parameters, so the Geometry, Accelerometer and Material objects can
            be created. If any of these entities is missing then corresponding
            argument should be provided. The default is None.

        Returns
        -------
        None

        """
        if (geometry, accel, material, spath) == (None, ) * 4:
            raise ValueError('Cannot create a Problem object without arguments.')

        self.n_cpu = cpu
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

            # Override geometry if explicit argument is given
            if geometry is not None:
                self.geometry = geometry

            # Try to read geometry from setup.json
            elif 'geometry' in setup_params:
                if 'template' in setup_params['geometry']:
                    templ = setup_params['geometry']['template']
                    del setup_params['geometry']['template']

                    self.geometry = Geometry(templ,
                                             accelerometer=self.accelerometer,
                                             params=GeometryParams(**setup_params['geometry']))

                elif 'edp' in setup_params['geometry']:
                    edp_file = setup_params['geometry']['edp']
                    del setup_params['geometry']['edp']

                    if not os.path.isabs(edp_file):
                        edp_file = os.path.join(spath, edp_file)

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
            self.parameters = jnp.array(self.material.get_params(self.geometry.height))

        except TypeError:
            warnings.warn('Some elastic moduli of a material were not provided, '
                          'solving forward problem as standalone will not be '
                          'possible.', RuntimeWarning)

        if ref_fr is not None:
            self.reference_fr = ref_fr

        self.e = self.geometry.height / 2.0
        self.rho = self.material.density

        processed_ff_output = processFFOutput(getOutput(self.geometry.current_file))

        m_names = ['M', 'L', "MCorrection", "LCorrection"]
        matrices = []

        for name in m_names:
            matrices.append(processed_ff_output[name])

        _mats = np.array(matrices) # TODO: check if we can avoid copying here

        _Ks = np.array(processed_ff_output['Ks'], dtype=np.float64)

        nz_mask = (np.sum(np.abs(_Ks), axis=0) + np.sum(np.abs(_mats), axis=0)).nonzero()

        self.M = _mats[0][nz_mask]
        self.L = _mats[1][nz_mask]
        MCorrection = _mats[2][nz_mask]
        LCorrection = _mats[3][nz_mask]
        self.Ks = np.zeros((6, self.M.size))

        for i in range(0, 6):
            self.Ks[i] = _Ks[i][nz_mask]

        v_names = ['fKs', 'fM', 'fL']
        for name in v_names:
            setattr(self, name, np.array(processed_ff_output[name],
                                         dtype=np.float64).view(StaticNdArrayWrapper))

        fMCorrection = np.array(processed_ff_output["fMCorrection"], dtype=np.float64)
        fLCorrection = np.array(processed_ff_output["fLCorrection"], dtype=np.float64)

        # TODO load can be not in phase with the BC vibration
        # and both BC and load may have different phases in points
        # so both of them have to be complex in general
        # but it is too comlicated as for now
        self.fLoad = np.array(processed_ff_output["fLoad"],
                              dtype=np.float64).view(StaticNdArrayWrapper)

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

        self.interpolation_vector = np.array(processed_ff_output["interpolation_vector"],
                                             dtype=np.float64)
        self.interpolation_value_from_bc = np.float64(processed_ff_output["interpolation_value_from_bc"])

        self.test_point = processed_ff_output["test_point_coord"]
        self.constrained_idx = processed_ff_output["constrained_idx"]

        sz = self.fM.size
        rs, cs = nz_mask

        rows = np.concatenate((rs, rs, rs + sz, rs + sz))
        cols = np.concatenate((cs, cs + sz, cs, cs + sz))

        self.indices = np.vstack((rows, cols), dtype=np.int32).T.view(StaticNdArrayWrapper)

    @functools.cache
    def getFRFunction(self):
        """
        Creates a function to evaluate AFC.

        Returns
        -------
        callable
            Function that takes array of frequencies `omega` [Hz] and the
            parameters `theta` and returns the FR array of complex amplitude in
            test point.

        """
        def _solve(f, params, Ks, fKs, MInertia, fInertia, fLoad,
                   interpolation_vector, interpolation_value_from_bc,
                   transform, indx, cpu):
            # solve for one frequency f (in [Hz])
            omega = 2.0 * np.pi * f

            # transform is a function D_ij = D_ij(theta), beta_ij = beta_ij(theta)
            # theta is the set of parameters; for example see ParamTransforms.isotropic_to_full
            D, beta = transform(params, omega)
            loss_moduli = beta * D

            # K_real = \sum K_ij*D_ij/(2.*e)
            # K_imag = \sum K_ij*D_ij/(2.*e)
            # Ks are matrices from eq (4.1.7)
            K_real = jnp.einsum(Ks, [0, ...], D, [0])
            K_imag = jnp.einsum(Ks, [0, ...], loss_moduli, [0])

            # f_imag = ..
            # fs are vectors from (4.1.11), they account for the Clamped BC (u = du/dn = 0)
            fK_real = jnp.einsum(fKs, [0, ...], D, [0])
            fK_imag = jnp.einsum(fKs, [0, ...], loss_moduli, [0])

            # Formulate system (A_real + i*A_imag)(u_real + i*u_imag) = (b_real + i*b_imag)
            # and create matrix from blocks for real solution
            # MInertia == rho*(M + 1/3 e^2 L) from
            A_real = -(omega ** 2) * MInertia + K_real
            A_imag = K_imag
            b_real = -(omega ** 2) * fInertia + fK_real + fLoad
            b_imag = fK_imag

            # A = vstack(
            #     (hstack((A_real, -A_imag)), hstack((-A_imag, -A_real)))
            # )
            data = jnp.concatenate((A_real, -A_imag, -A_imag, -A_real))
            b = jnp.concatenate((b_real, -b_imag))

            u = spsolve(data, indx, b, n_cpu=cpu)

            # interpolation_vector == c
            # interpolation_value_from_bc == c_0 from 4.1.18
            u_in_test_point = jnp.array([interpolation_value_from_bc, 0.0]) +\
            interpolation_vector @ u.reshape(  # real, imag
                (-1, 2), order="F"
            )

            return u_in_test_point[0] + 1j * u_in_test_point[1]


        _solve_p = jax.tree_util.Partial(_solve,
                                         Ks=self.Ks / 2.0 / self.e,
                                         fKs=self.fKs / 2.0 / self.e,
                                         MInertia=self.MInertia,
                                         fInertia=self.fInertia,
                                         fLoad=self.fLoad / 2.0 / self.e,
                                         interpolation_vector=self.interpolation_vector,
                                         interpolation_value_from_bc=self.interpolation_value_from_bc,
                                         transform=self.material.transform,
                                         indx=self.indices,
                                         cpu=self.n_cpu)

        _get_afc = jax.jit(jax.vmap(_solve_p, in_axes=(0, None)))

        return _get_afc

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
                          **opt_kwargs) -> optResult:
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
                    om_r_err1 = np.array((other_moduli_real - other_moduli0) / other_moduli_real)
                    om_r_err2 = np.array((other_moduli_real - other_moduli) / other_moduli_real)

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

            comp_str = ''
            if compression[0]:
                comp_str = (f'Using compression algorithm {comp_alg} with '
                            f'{compression[1]} points.\n')


            rep_str = (f'{self.accelerometer}\n{self.material}\n{self.geometry}\n'
                       + extra_info + comp_str +
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
                       f'Optimizer parameters: {opt_kwargs}\n'
                       f'Optimizer type: {optimizer}.\n')
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
        func_type: str # Available options are: MSE, RMSE, MSE_AFC, MSE_LOG_AFC
    ) -> Callable:
        assert frequencies.shape[0] == reference_fr.shape[0]
        fr_function = self.getFRFunction()

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

    def getPhysicalModuli(self, params: np.ndarray | jax.Array = None) -> tuple:
        """
        Get physical moduli from Ds.
        For example, for isotropic material returns Young's modulus `E` and shear
        modulus `G` from flexural rigidity `D` and Poisson's ratio `nu`.

        Parameters
        ----------
        params : np.ndarray | jax.Array
            Parameters to be tranformed. If `None`, tries to uses parameters
            stored in Problem object.

        Returns
        -------
        tuple
            Tuple of float containing physical moduli for given material.

        """
        if params is None:
            try:
                params = self.parameters
            except AttributeError as err:
                raise RuntimeError('Cannot get parameters from function '
                                   'arguments or Problem attributes.') from err

        return self.material.D_to_phys(self.geometry.height, *params)
