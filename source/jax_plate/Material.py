import abc
import functools
import json
import os
from typing import Callable

import jax
from jax.tree_util import Partial
import jax.numpy as jnp
import numpy as np
import scipy.optimize
import sympy as sp
from sympy.solvers.solveset import linear_eq_to_matrix

from jax_plate.Utils import get_jax_plate_dir


"""
Accelerometer has posisitve z coordinate values,
theta angles are measured counterclockwise from x coordinate.
First layer has minimal z coordinate (negative value).
"""


ATYPES = {'isotropic': {'E', 'G', 'beta'},
          'orthotropic': {'E1', 'E2', 'G12', 'nu12', 'beta'},
          'orthotropic_d4': {'E1', 'E2', 'G12', 'nu12',
                             'b1', 'b2', 'b3', 'b4'}, # each modulus has own loss factor
          'sol': {'E1', 'E2', 'G12', 'nu12', 'beta', # simple orthotropic laminate with identical unidirectional layers
                              'angles'}, # tuple of angles in DEGREES for each lamina, angles are counter-clockwise, starting from the lowest
          'symm_sol': {'E1', 'G12', 'nu12', 'beta', 'angles'}
          }


class Material(abc.ABC):
    """
    Interface class for all materials.

    Attributes
    ----------
    density : float
        Materials density in kg/m^3.
    Elastic moduli : floats
        Attributes listed in jax_plate.Material.ATYPES for each anisotropy type.
    is_mps : bool
        `Is midplane symmetric` -- this attribute impacts the choice of solver.

    """
    def get_parameters(self) -> jax.Array | None:
        """
        Checks if Material has all elastic moduli as attributes and returns
        them on success.

        Returns
        -------
        jax.Array | None
            If `Material.has_params` then returns an array with elasic moduli.
            Else returns None.

        """
        if self.has_params:
            return jnp.array(self._get_param_tuple())
        else:
            return None

    @abc.abstractmethod
    def _get_param_tuple(self) -> tuple:
        """
        Helper function, returns all elastic moduli of a particular Material
        in one tuple.

        Returns
        -------
        tuple
            Elastic moduli in order, described in jax_plate.Material.ATYPES.

        """
        pass

    # TODO: decide if we need such method
    # @abc.abstractmethod
    # def set_parameters(self, params: jax.Array) -> None:
    #     pass

    @property
    def has_params(self) -> bool:
        """
        Bool property, shows if all elastic moduli are defined in a class.

        Returns
        -------
        bool
            `True` if all moduli are defined, else `False`.

        """
        if None in self._get_param_tuple():
            return False
        else:
            return True

    @staticmethod
    @abc.abstractmethod
    def get_ABD_transform(h: float) -> Callable:
        """
        Returns a function that transforms elastic moduli into A_ij, B_ij, D_ij
        matrix components (complex moduli) in the following order:
        [11, 12, 16, 22, 26, 66]. May include frequency dependece,
        taking frequency omega as the second argument.

        Parameters
        ----------
        h : float
            Height of a plate.

        Returns
        -------
        Callable
            Tranform function with signature `f(array_of_parameters, omega) ->
            tuple[jax.Array, jax.Array, jax.Array]`.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_D_transform(h: float) -> Callable:
        """
        Returns a function that transforms elastic moduli into flexural rigity
        matrix components (complex moduli) D_ij in the following
        order: [11, 12, 16, 22, 26, 66]. May include frequency dependece,
        taking frequency omega as the second argument.

        Parameters
        ----------
        h : float
            Height of a plate.

        Returns
        -------
        Callable
            Tranform function with signature `f(array_of_parameters, omega) ->
            jax.Array`.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _get_nonlin_constr(scaling_params: np.ndarray | float = 1.0
                           ) -> tuple[Callable, np.ndarray, np.ndarray]:
        """
        Method that returns nonlinear constraints of material parameters in the
        following form:
            lb <= fun(x) <= ub,
        where `x` is the array of material parameters, `lb`, `fun(x)` and `ub`
        are arrays of size `n` -- overall amount of inequalities.

        Parameters
        ----------
        scaling_params : numpy.ndarray
            Parameters that are used for scaling elastic moduli during
            optimization. The default is None (no scaling applied).

        Returns
        -------
        (Callable, np.ndarray, np.ndarray)
            `fun`, `lb` and `ub` respectively. `fun` is compatible with `jax`
            transforms.

        None
            If the material doesn`t have any nonlinear constraints.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _get_lin_constr(scaling_params: np.ndarray | float = 1.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Method that returns linear constraints of material parameters in the
        following form:
            lb <= A @ x <= ub,
        where `x` is the array (size `n`) of material parameters, `A` is a
        matrix of shape (`m`, `n`), `lb`, and `ub` are arrays of size `m` --
        overall amount of inequalities, `n` is a size of parameter space.

        Parameters
        ----------
        scaling_params : numpy.ndarray
            Parameters that are used for scaling elastic moduli during
            optimization. The default is None (no scaling applied).

        Returns
        -------
        (np.ndarray, np.ndarray, np.ndarray)
            `A`, `lb` and `ub` respectively.

        None
            If the material doesn`t have any linear constraints.

        """
        pass

    @classmethod
    def get_constraints(cls,
                        scaling_params: np.ndarray = None
                        ) -> list[scipy.optimize.NonlinearConstraint,
                                  scipy.optimize.LinearConstraint]:
        """
        Method that returns scipy constraints to be used in optimization via
        scipy.optimize routines.

        Parameters
        ----------
        scaling_params : numpy.ndarray
            Parameters that are used for scaling elastic moduli during
            optimization. The default is None (no scaling applied).

        Returns
        -------
        One of the following:

        list[scipy.optimize.NonlinearConstraint, scipy.optimize.LinearConstraint]
            When a material class has both constraints on elastic parameters.

        scipy.optimize.NonlinearConstraint
            When a material has only nonlinear constraints.

        scipy.optimize.LinearConstraint
            When a material has only linear constraints.

        """
        if scaling_params is None:
            scaling_params = 1.0
        else:
            scaling_params = scaling_params.copy()

        res_nonlin = cls._get_nonlin_constr(scaling_params)
        if res_nonlin is not None:
            fun, lb, ub = res_nonlin
            fun_jac = jax.jit(jax.jacobian(fun))

            def dot_func(x, v):
                return jax.numpy.dot(fun(x), v)

            fun_hess = jax.jit(jax.hessian(dot_func))
            res_nonlin = scipy.optimize.NonlinearConstraint(fun, lb, ub,
                                                            jac=fun_jac,
                                                            hess=fun_hess)

        res_lin = cls._get_lin_constr(scaling_params)
        if res_lin is not None:
            A, lb, ub = res_lin
            res_lin = scipy.optimize.LinearConstraint(A, lb, ub)

        if res_nonlin is None:
            return res_lin

        elif res_lin is None:
            return res_nonlin

        else:
            return [res_lin, res_nonlin]

    @classmethod
    def check_parameters(cls, params: np.ndarray,
                         scaling_params: np.ndarray | float = 1.0) -> bool:
        """
        Method that checks if parameters are correct. For instance, for isotropic
        material Young's modulus E cannot be negative.

        Parameters
        ----------
        params : numpy.ndarray
            Material parameters to check for correctness.

        caling_params : numpy.ndarray
            Parameters that are used for scaling elastic moduli during
            optimization. The default is None (no scaling applied).

        Returns
        -------
        bool
            `True` if all elastic moduli are correct, else `False`.

        """
        res_nonlin = cls._get_nonlin_constr(scaling_params)
        if res_nonlin is None:
            nonlin_check = True
        else:
            fun, lb, ub = res_nonlin
            fun_vals = np.array(fun(params))
            lhs = lb <= fun_vals # TODO: check if `<` should be here
            rhs = fun_vals <= ub
            nonlin_check = np.all(lhs * rhs)

        res_lin = cls._get_lin_constr(scaling_params)
        if res_lin is None:
            lin_check = True
        else:
            A, lb, ub = res_lin
            fun_vals = A @ params
            lhs = lb <= fun_vals
            rhs = fun_vals <= ub
            lin_check = np.all(lhs * rhs)

        return lin_check and nonlin_check

    def get_save_dict(self) -> dict:
        """
        Returns all Material attributes that are needed for saving value to
        file.

        Returns
        -------
        dict
            Dictionary with all attributes needed for creation of instance's
            copy.

        """
        return self.__dict__

    def __str__(self):
        s = f'{self.__class__.__name__} material with\n'
        d = self.get_save_dict()
        for k, v in d.items():
            s += f'{k} = {v}\n'
        return s.rstrip()

    def save_to_file(self, material_name: str) -> None:
        """
        Method to create a .json file for a material with given name in
        `materials` folder.

        Parameters
        ----------
        material_name : str
            Name of the material to be saved.

        Returns
        -------
        None

        """
        materials_folder = os.path.join(get_jax_plate_dir(), 'materials')

        if not os.path.exists(materials_folder):
            os.mkdir(materials_folder)

        fpath = os.path.join(materials_folder, material_name + '.json')

        with open(fpath, 'w') as file:
            json.dump(self.get_save_dict(), file, indent=4)


class Isotropic(Material):
    def __init__(self, density: float,
                 E: float | None = None,
                 G: float | None = None,
                 beta: float | None = None):
        self.density = density
        self.is_mps = True

        self.E = E
        self.G = G
        self.beta = beta

    def _get_param_tuple(self) -> tuple:
        return (self.E, self.G, self.beta)

    @staticmethod
    def get_ABD_transform(h: float) -> Callable:
        def _transform(params, *args, _h):
            E = params[0]
            G = params[1]
            beta = params[2]

            nu = E / (2.0 * G) - 1.0

            A = E * _h / (1 - nu**2)
            D = A * _h ** 2 / 12.0

            arr = jnp.array([1.0, nu, 0.0, 1.0, 0.0, (1 - nu) / 2]) * (1 + 1j * beta)

            As = A * arr
            Bs = jnp.zeros_like(arr)
            Ds = D * arr
            return As, Bs, Ds

        return Partial(_transform, _h=h)

    @staticmethod
    def get_D_transform(h: float) -> Callable:
        def _transform(params, *args, _h):
            E = params[0]
            G = params[1]
            beta = params[2]

            nu = E / (2.0 * G) - 1.0
            D = E * _h ** 3 / (12.0 * (1.0 - nu ** 2))

            Ds = jnp.array([D, nu * D, 0.0, D, 0.0, D * (1 - nu) / 2]) * (1 + 1j * beta)
            return Ds

        return Partial(_transform, _h=h)

    @staticmethod
    def _get_nonlin_constr(scaling_params: np.ndarray | float = 1.0):
        return

    @staticmethod
    def _get_lin_constr(scaling_params: np.ndarray | float = 1.0):
        """
        E > 0
        G > 0
        beta > 0
        E - 2G > 0
        -E + 3G > 0 (last two are equal to 2G < E < 3G or 0 < nu < 0.5)
        """
        A = np.eye(5, 3)
        A[3, 0] = 1.0
        A[4, 0] = -1.0
        A[3, 1] = -2.0
        A[4, 1] = 3.0

        if isinstance(scaling_params, float):
            A *= scaling_params
        else:
            A = A * scaling_params[:, None].T

        eps = 1e-12
        return A, eps, np.inf


class Orthotropic(Material):
    def __init__(self, density: float,
                 E1: float | None = None,
                 E2: float | None = None,
                 G12: float | None = None,
                 nu12: float | None = None,
                 beta: float | None = None):
        self.density = density
        self.is_mps = True

        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.beta = beta

    def _get_param_tuple(self) -> tuple:
        return (self.E1, self.E2, self.G12, self.nu12, self.beta)

    @staticmethod
    def get_ABD_transform(h: float) -> Callable:
        def _transform(params, *args, _h):
            E1 = params[0]
            E2 = params[1]
            G12 = params[2]
            nu12 = params[3]
            beta = params[4]

            E_ratio = E2 / E1
            nu21 = E_ratio * nu12

            A11 = E1 * _h / (1 - nu12 * nu21)
            A12 = nu21 * A11
            A22 = E2/E1 * A11
            A66 = G12 * _h

            D11 = E1 * _h ** 3 / (12 * (1 - nu12 * nu21))
            D66 = G12 * _h ** 3 / 12
            D12 = nu21*D11
            D22 = D11/E_ratio

            As = jnp.array([A11, A12, 0.0, A22, 0.0, A66]) * (1 + 1j * beta)
            Bs = jnp.zeros_like(As)
            Ds = jnp.array([D11, D12, 0.0, D22, 0.0, D66]) * (1 + 1j * beta)
            return As, Bs, Ds

        return Partial(_transform, _h=h)

    @staticmethod
    def get_D_transform(h: float) -> Callable:
        def _transform(params, *args, _h):
            E1 = params[0]
            E2 = params[1]
            G12 = params[2]
            nu12 = params[3]
            beta = params[4]

            E_ratio = E2 / E1
            nu21 = E_ratio * nu12

            D11 = E1 * _h ** 3 / (12 * (1 - nu12 * nu21))
            D66 = G12 * _h ** 3 / 12
            D12 = nu21*D11
            D22 = D11/E_ratio

            Ds = jnp.array([D11, D12, 0.0, D22, 0.0, D66])  * (1 + 1j * beta)
            return Ds

        return Partial(_transform, _h=h)

    @staticmethod
    def _get_nonlin_constr(scaling_params: np.ndarray | float = 1.0):
        """
        sqrt(E1/E2) - nu12 > 0
        """
        def constr_func(params):
            params = params * scaling_params
            nu12_coef = jnp.sqrt(params[0] / params[1]) - params[3]
            return jnp.array([nu12_coef], dtype=jnp.float64)

        eps = 1e-12
        return constr_func, eps, np.inf

    @staticmethod
    def _get_lin_constr(scaling_params: np.ndarray | float = 1.0):
        """
        E1 > 0
        E2 > 0
        G12 > 0
        nu12 > 0
        beta > 0
        E1 - E2 > 0
        1.05*E1 - G12 > 0 # G12 in theory can be greater than E1, but rarely in practice
        """
        A = np.eye(7, 5)
        A[5, 0] = 1.0
        A[5, 1] = -1.0
        A[6, 0] = 1.05
        A[6, 2] = -1.0

        if isinstance(scaling_params, float):
            A *= scaling_params
        else:
            A = A * scaling_params[:, None].T

        eps = 1e-12
        return A, eps, np.inf


class OrthotropicD4(Material):
    def __init__(self, density: float,
                 E1: float | None = None,
                 E2: float | None = None,
                 G12: float | None = None,
                 nu12: float | None = None,
                 b1: float | None = None,
                 b2: float | None = None,
                 b3: float | None = None,
                 b4: float | None = None):
        self.density = density
        self.is_mps = True

        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4

    def _get_param_tuple(self) -> tuple:
        return (self.E1, self.E2, self.G12, self.nu12,
                self.b1, self.b2, self.b3, self.b4)

    @staticmethod
    def get_ABD_transform(h: float) -> Callable:
        def _transform(params, *args, _h):
            b1 = params[4]
            b2 = params[5]
            b3 = params[6]
            b4 = params[7]

            E1 = params[0] * (1 + 1j * b1)
            E2 = params[1] * (1 + 1j * b2)
            G12 = params[2] * (1 + 1j * b3)
            nu12 = params[3] * (1 + 1j * b4)

            E_ratio = E2 / E1
            nu21 = E_ratio * nu12

            A11 = E1 * _h / (1 - nu12 * nu21)
            A12 = nu21 * A11
            A22 = E2/E1 * A11
            A66 = G12 * _h

            D11 = E1 * h ** 3 / (12 * (1 - nu12 * nu21))
            D66 = G12 * h ** 3 / 12
            D12 = nu21*D11
            D22 = D11/E_ratio

            As = jnp.array([A11, A12, 0.0, A22, 0.0, A66])
            Bs = jnp.zeros_like(As)
            Ds = jnp.array([D11, D12, 0.0, D22, 0.0, D66])
            return As, Bs, Ds

        return Partial(_transform, _h=h)

    @staticmethod
    def get_D_transform(h: float) -> Callable:
        def _transform(params, *args, _h):
            b1 = params[4]
            b2 = params[5]
            b3 = params[6]
            b4 = params[7]

            E1 = params[0] * (1 + 1j * b1)
            E2 = params[1] * (1 + 1j * b2)
            G12 = params[2] * (1 + 1j * b3)
            nu12 = params[3] * (1 + 1j * b4)

            E_ratio = E2 / E1
            nu21 = E_ratio * nu12

            D11 = E1 * h ** 3 / (12 * (1 - nu12 * nu21))
            D66 = G12 * h ** 3 / 12
            D12 = nu21*D11
            D22 = D11/E_ratio

            Ds = jnp.array([D11, D12, 0.0, D22, 0.0, D66])
            return Ds

        return Partial(_transform, _h=h)

    @staticmethod
    def _get_nonlin_constr(scaling_params: np.ndarray | float = 1.0):
        raise NotImplementedError() # TODO: implement

    @staticmethod
    def _get_lin_constr(scaling_params: np.ndarray | float = 1.0):
        raise NotImplementedError()


class SOL(Orthotropic):
    """Simple Orthotropic Laminate"""
    def __init__(self, density: float,
                 angles: tuple | list,
                 E1: float | None = None,
                 E2: float | None = None,
                 G12: float | None = None,
                 nu12: float | None = None,
                 beta: float | None = None):
        super().__init__(density, E1, E2, G12, nu12, beta)

        self.angles = np.array(angles)

        if np.sum(np.abs(self.angles - self.angles[::-1])) > 1e-6:
            self.is_mps = False

    def get_save_dict(self):
        sup_dict = super().get_save_dict()
        return ({x: sup_dict[x] for x in sup_dict if x != '_Q_to_ABD_matrices'} |
                {'angles': list(self.angles)})

    @functools.cached_property
    def _Q_to_ABD_matrices(self):
        Q11, Q12, Q22, Q26, Q16, Q66, h = sp.symbols('Q_11 Q_12 Q_22 Q_26 Q_16 Q_66 h')

        Q = sp.zeros(3)
        Q[0, 0] = Q11
        Q[1, 1] = Q22
        Q[0, 1] = Q[1, 0] = Q12
        Q[2, 2] = Q66

        def T(t):
            t *= sp.pi/180
            T = sp.zeros(3)
            m = sp.cos(t)
            n = -sp.sin(t)
            mn = m*n

            T = sp.Matrix([[m**2, n**2,       -2*mn],
                           [n**2, m**2,        2*mn],
                           [  mn,  -mn, m**2 - n**2]])
            return T

        def linspace(n):
            res = []
            step = h / n
            for i in range(n+1):
                res.append(-h/2 + i*step)

            return sp.Array(res)

        zs = linspace(self.angles.size)
        z1 = zs
        z2 = zs.applyfunc(lambda x: x**2)
        z3 = zs.applyfunc(lambda x: x**3)

        def diff(arr):
            res = []
            for i in range(1, len(arr)):
                res.append(arr[i] - arr[i-1])
            return res

        zd1 = diff(z1)
        zd2 = diff(z2)
        zd3 = diff(z3)

        A = sp.zeros(3)
        B = sp.zeros(3)
        D = sp.zeros(3)

        for i in range(self.angles.size):
            QT = T(self.angles[i]) @ Q @ T(self.angles[i]).T
            A += QT * zd1[i]
            B += QT * zd2[i]
            D += QT * zd3[i]

        B /= 2
        D /= 3

        A, _ = linear_eq_to_matrix([A[0, 0],
                                    A[0, 1],
                                    A[0, 2],
                                    A[1, 1],
                                    A[1, 2],
                                    A[2, 2]],
                                   (Q11, Q12, Q16, Q22, Q26, Q66))

        B, _ = linear_eq_to_matrix([B[0, 0],
                                    B[0, 1],
                                    B[0, 2],
                                    B[1, 1],
                                    B[1, 2],
                                    B[2, 2]],
                                   (Q11, Q12, Q16, Q22, Q26, Q66))

        D, _ = linear_eq_to_matrix([D[0, 0],
                                    D[0, 1],
                                    D[0, 2],
                                    D[1, 1],
                                    D[1, 2],
                                    D[2, 2]],
                                   (Q11, Q12, Q16, Q22, Q26, Q66))
        return A, B, D

    def get_ABD_transform(self, h: float) -> Callable:
        _matA, _matB, _matD = self._Q_to_ABD_matrices

        A = np.array(_matA.evalf(subs={'h': h}), dtype=np.float64)
        B = np.array(_matB.evalf(subs={'h': h}), dtype=np.float64)
        D = np.array(_matD.evalf(subs={'h': h}), dtype=np.float64)

        def _transform(params, *args, _MA, _MB, _MD):
            E1 = params[0]
            E2 = params[1]
            G12 = params[2]
            nu12 = params[3]
            beta = params[4]

            den = 1 - E2 / E1 * nu12 ** 2
            Q = jnp.array([E1/den, nu12 * E2 / den, 0, E2 / den, 0, G12])
            As = (_MA @ Q)  * (1 + 1j * beta)
            Bs = (_MB @ Q)  * (1 + 1j * beta)
            Ds = (_MD @ Q)  * (1 + 1j * beta)
            return As, Bs, Ds

        return Partial(_transform, _MA=A, _MB=B, _MD=D)

    def get_D_transform(self, h: float) -> Callable:
        if self.is_mps:
            _, _, _mat = self._Q_to_ABD_matrices

            A = np.array(_mat.evalf(subs={'h': h}), dtype=np.float64)

            def _transform(params, *args, _M):
                E1 = params[0]
                E2 = params[1]
                G12 = params[2]
                nu12 = params[3]
                beta = params[4]

                den = 1 - E2 / E1 * nu12 ** 2
                Q = jnp.array([E1/den, nu12 * E2 / den, 0, E2 / den, 0, G12])
                Ds = (_M @ Q)  * (1 + 1j * beta)
                return Ds

            return Partial(_transform, _M=A)

        else: # Will not be implemented, as physical problem would be incorrect
            raise NotImplementedError('Transform without A_ij and B_ij matrices'
                                      'for not midplane-symmetric composites.')


class SymmetricalSOL(SOL):
    """Simple Orthotropic Laminate with layers with E1=E2"""
    def __init__(self, density: float,
                 angles: tuple | list,
                 E1: float | None = None,
                 G12: float | None = None,
                 nu12: float | None = None,
                 beta: float | None = None):
        super().__init__(density, angles, E1, E1, G12, nu12, beta)

    @property
    def E2(self):
        return self.E1

    @E2.setter
    def E2(self, val):
        self.E1 = val

    def _get_param_tuple(self) -> tuple:
        return (self.E1, self.G12, self.nu12, self.beta)

    def get_ABD_transform(self, h: float) -> Callable:
        _matA, _matB, _matD = self._Q_to_ABD_matrices

        A = np.array(_matA.evalf(subs={'h': h}), dtype=np.float64)
        B = np.array(_matB.evalf(subs={'h': h}), dtype=np.float64)
        D = np.array(_matD.evalf(subs={'h': h}), dtype=np.float64)

        def _transform(params, *args, _MA, _MB, _MD):
            E1 = params[0]
            E2 = params[0]
            G12 = params[1]
            nu12 = params[2]
            beta = params[3]

            den = 1 - E2 / E1 * nu12 ** 2
            Q = jnp.array([E1/den, nu12 * E2 / den, 0, E2 / den, 0, G12])
            As = (_MA @ Q)  * (1 + 1j * beta)
            Bs = (_MB @ Q)  * (1 + 1j * beta)
            Ds = (_MD @ Q)  * (1 + 1j * beta)
            return As, Bs, Ds

        return Partial(_transform, _MA=A, _MB=B, _MD=D)

    def get_D_transform(self, h: float) -> Callable:
        if self.is_mps:
            _, _, _mat = self._Q_to_ABD_matrices

            A = np.array(_mat.evalf(subs={'h': h}), dtype=np.float64)

            def _transform(params, *args, _M):
                E1 = params[0]
                E2 = params[0]
                G12 = params[1]
                nu12 = params[2]
                beta = params[3]

                den = 1 - E2 / E1 * nu12 ** 2
                Q = jnp.array([E1/den, nu12 * E2 / den, 0, E2 / den, 0, G12])
                Ds = (_M @ Q)  * (1 + 1j * beta)
                return Ds

            return Partial(_transform, _M=A)

        else: # Will not be implemented, as physical problem would be incorrect
            raise NotImplementedError('Transform without A_ij and B_ij matrices'
                                      'for not midplane-symmetric composites.')

    @staticmethod
    def _get_nonlin_constr(scaling_params: np.ndarray | float = 1.0):
        return

    @staticmethod
    def _get_lin_constr(scaling_params: np.ndarray | float = 1.0):
        """
        E1 > 0
        G12 > 0
        0 < nu12 < 1
        beta > 0
        E1 - G12 > 0
        """
        A = np.eye(5, 4)
        A[4, 0] = 1.0
        A[4, 1] = -1.0

        if isinstance(scaling_params, float):
            A *= scaling_params
        else:
            A = A * scaling_params[:, None].T

        eps = 1e-12
        lb = np.full(5, eps)
        ub = np.full(5, np.inf)
        ub[2] = 1.0 - eps
        return A, lb, ub


def get_material(main_arg: str | float | int | dict,
                 atype: str = None, **kwargs) -> Material:
    """
    Function to create Material object with specific type.

    Parameters
    ----------
    main_arg : str | float | dict
        One of the following:
            1) Name of the material to search for in `materials` folder
            (without `.json` extension)
            2) A float or int number representing density
            3) A dict with all material parameters
            4) Path to `.json` file with material properties.
    atype : str
        Type of anisotropy, required when `main_arg` is density.
    **kwargs
        Keyword arguments with float values for each elastic modulus needed.

    Returns
    -------
    Material
        One of subclasses of the Material interface class.

    """
    params = None

    if isinstance(main_arg, str):
        fname, ext = os.path.splitext(main_arg)
        if ext == '.json':
            fpath = os.path.abspath(main_arg)
        elif ext == '':
            fpath = os.path.join(get_jax_plate_dir(), 'materials',
                                 main_arg + '.json')
        else:
            raise ValueError('Unsupported extinsion for material properties '
                             f'file: `{ext}`.')

        if os.path.exists(fpath):
            with open(fpath, 'r') as file:
                params_json = json.load(file)
                try:
                    params = {k:v for k, v in params_json.items() if k not in ('density', 'atype')}
                    density = params_json['density']
                    atype = params_json['atype']

                except KeyError as err:
                    raise RuntimeError(f'Required parameter {err.args[0]} was '
                                       'not provided by the .json file '
                                       f'{fpath}.')

        else:
            raise ValueError(f'Could not find file {main_arg} or '
                             'such material in `materials` folder.')

    elif isinstance(main_arg, (float, int)):
        density = float(main_arg)
        if not isinstance(atype, str):
            raise ValueError('Atype argument was not provided.')
        params = kwargs

    elif isinstance(main_arg, dict):
        try:
            density = main_arg['density']
            atype = main_arg['atype']
            params = {k:v for k, v in main_arg.items() if k not in ('density', 'atype')}
        except KeyError as err:
            raise RuntimeError(f'Required parameter {err.args[0]} was '
                               'not provided in dictionary, cannot create Material.')

    else:
        raise TypeError('Argument `name_or_density` should have type '
                        '`str` or `float.`')

    if density <= 0:
        raise ValueError('Cannot create Material with negative material '
                         f'density: {density}.')

    if atype not in ATYPES.keys():
        raise ValueError(f'Invalid anisotropy type {atype} for material. '
                         f'Supported options are: {list(ATYPES.keys())}.')

    if atype == 'sol' and 'angles' not in params.keys():
        raise ValueError('Cannot create simple orthotropic laminate material '
                         'without `angles` tuple.')

    if not set(params.keys()).issubset(ATYPES[atype]):
        raise ValueError('Mismatching anisotropy type and provided arguments: '
                         f'expected values of {ATYPES[atype]}, got {params.keys()}.')

    if atype == 'isotropic':
        return Isotropic(density, **params)

    elif atype == 'orthotropic':
        return Orthotropic(density, **params)

    elif atype == 'orthotropic_d4':
        return OrthotropicD4(density, **params)

    elif atype in ('sol', 'orth_lam_simple'):
        return SOL(density, **params)

    elif atype == 'symm_sol':
        return SymmetricalSOL(density, **params)

    else: # shouldn't reach there
        raise NotImplementedError()
