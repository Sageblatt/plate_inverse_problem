"""Modified CPU version of jax.experimental.sparse.linalg.spsolve with batching"""
import functools
from multiprocessing import cpu_count

import jax
from jax import core
from jax.interpreters import ad, batching, mlir, xla
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix

from jax_plate.jax_plate_lib import InnerState


# Ensure that jax uses CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

class SolverState:
    def __init__(self):
        self.state = InnerState()
        self.patterns = []
        self.permutations = []

    @property
    def state_size(self):
        return len(self.patterns)

    def add_mat(self, mat, mat_T, indices, permutation) -> int:
        self.patterns.append(indices.copy())
        self.permutations.append(permutation.copy())

        self.state.add_mat(mat.shape[0], mat.indices, mat.indptr,
                           mat_T.indices, mat.indptr, permutation, mat.data)

        return self.state_size - 1

    def solve(self, data, b, solver_num, transpose, n_cpu, _mode):
        return self.state.solve(data, b, solver_num, transpose, n_cpu, _mode)

    def matvec(self, mat, vec, solver_num, transpose, n_cpu, _mode):
        return self.state.matvec(mat, vec, solver_num, transpose, n_cpu, _mode)

_SOLVER_STATE = SolverState()

def find_permutation(arr1: np.ndarray[int], arr2: np.ndarray[int],
                     max_val: int = None) -> np.ndarray[int]:
    """
    Finds numpy mask array such that `arr1[find_permutation(arr1, arr2)]`
    equals `arr2`. May give wrong results if permutation is not unique.

    Parameters
    ----------
    arr1 : np.ndarray[int]
        Array of non-negative integers with shape (N, 2).
    arr2 : np.ndarray[int]
        Array of non-negative integers with shape (N, 2).
    max_val : int
        Maximum value of both arrays. If `None`, this value
        will be calculated via np.max(). The default is None.

    Returns
    -------
    mask : np.ndarray
        Masking array.

    """
    assert arr1.shape == arr2.shape
    assert arr1.shape[1] == 2

    if max_val is None:
        max_val = np.max(arr1) + 1

    def unwind(arr):
        return arr[:, 0] + arr[:, 1] * max_val

    u1 = unwind(arr1)
    u2 = unwind(arr2)

    is1 = np.argsort(u1)
    is2 = np.argsort(u2)

    inverse_is2 = is2[is2[is2]]
    res = np.arange(u1.size)[is1][inverse_is2]
    return np.array(res, dtype=arr1.dtype)

FAMILIES = {(np.float64, np.int32): 'di',
            (np.float64, np.int64): 'dl',
            (np.complex128, np.int32): 'zi',
            (np.complex128, np.int64): 'zl'}

def create_symbolic(N: int,
                    indices: np.ndarray[np.int32 | np.int64],
                    mat_dtype: np.dtype) -> (np.ndarray, int):
    fam_idx = (mat_dtype, indices.dtype.type)
    if not fam_idx in FAMILIES:
        raise TypeError(f'Invalid dtypes of arguments: got {fam_idx=}, '
                        f'expected one of {FAMILIES.keys()}')

    data = np.linspace(1.0, 42.0, indices.shape[0], dtype=mat_dtype)
    mat = coo_matrix((data, (indices[:, 0], indices[:, 1])), (N, N),
                     dtype=mat_dtype, copy=True).tocsc()
    mat_T = coo_matrix((data, (indices[:, 1], indices[:, 0])), (N, N),
                       dtype=mat_dtype, copy=True).tocsc()

    m1 = mat.tocoo()
    m1_T = mat_T.tocoo()

    coo_indices = np.vstack((m1.row, m1.col), dtype=indices.dtype).T
    coo_indices_T = np.vstack((m1_T.row, m1_T.col), dtype=indices.dtype).T

    indices_T = coo_indices[:, ::-1]
    perm = find_permutation(indices_T, coo_indices_T, N)

    res = _SOLVER_STATE.add_mat(mat, mat_T, coo_indices, perm)
    return (m1.row, m1.col), res
#---------------------

# Abstract eval for both matvec and spsolve
def _abstract_eval(data, b, *, solver_num, transpose=False, n_cpu=None, _mode=0):
    if data.dtype != b.dtype:
        raise ValueError(f"data types do not match: {data.dtype=} {b.dtype=}")
    if not isinstance(n_cpu, int):
        raise TypeError(f'invalid type of `n_cpu` argument, expected `int`, '
                         f'got {type(n_cpu)}')
    if n_cpu < 0:
        raise ValueError('n_cpu argument should be non-negative')
    if not isinstance(_mode, int):
        raise TypeError('invalid type of `_mode` argument, expected `int`, got '
                         f'{type(_mode)}')
    transpose = bool(transpose)
    if not isinstance(solver_num, int):
        raise TypeError('invalid type of `solver_num` argument, expected '
                        f'`int`, got {type(solver_num)}')
    if _mode in (0, 2, 3, 4):
        return b
    elif _mode == 1:
        return core.ShapedArray((data.shape[0], b.shape[0]), b.dtype)
    else:  # should't reach here as handling is done in _spsolve_batch
        raise NotImplementedError()
# ----------------

# custom matvec primitive ----------------------------------------------------
def matvec(mat, vec, *, solver_num, transpose=False, n_cpu=None, _mode=0):
    if n_cpu == 0:
        n_cpu = cpu_count()
    return _matvec_p.bind(mat, vec, solver_num=solver_num, transpose=transpose,
                          n_cpu=n_cpu, _mode=_mode)

def _matvec_cpu_lowering(ctx, data, b, *, solver_num, transpose, n_cpu, _mode):
    args = [data, b]

    def _callback(data, b, **kwargs):
        res = _SOLVER_STATE.matvec(data, b, solver_num, transpose, n_cpu, _mode)
        return (res,)

    result, _, _ = mlir.emit_python_callback(
        ctx, _callback, None, args, ctx.avals_in, ctx.avals_out,
        has_side_effect=False)
    return result

def _matvec_jvp_mat(data_dot, data, v, **kwargs):
    return matvec(data_dot, v, **kwargs)

def _matvec_jvp_vec(v_dot, data, v, **kwargs):
    return matvec(data, v_dot, **kwargs)

def _matvec_transpose(ct, data, v, **kwargs):
    if ad.is_undefined_primal(v):
        kwargs['transpose'] = not kwargs['transpose']
        return data, matvec(data, ct, **kwargs)
    else:
        num = kwargs['solver_num']
        patt = _SOLVER_STATE.patterns[num]
        row, col = patt[:, 0], patt[:, 1]
        return ct[..., row] * v[..., col], v

_matvec_p = core.Primitive('custom_matvec')
_matvec_p.def_impl(functools.partial(xla.apply_primitive, _matvec_p))
_matvec_p.def_abstract_eval(_abstract_eval)
ad.defjvp(_matvec_p, _matvec_jvp_mat, _matvec_jvp_vec)
ad.primitive_transposes[_matvec_p] = _matvec_transpose
mlir.register_lowering(_matvec_p, _matvec_cpu_lowering, platform='cpu')
#--------------------------------------------------------------------

# custom spsolve --------------------------------------------------------------
def _spsolve_cpu_lowering(ctx, data, b, *, solver_num, transpose, n_cpu, _mode):
    args = [data, b]

    def _callback(data, b, **kwargs):
        res = _SOLVER_STATE.solve(data, b, solver_num, transpose, n_cpu, _mode)
        return (res,)

    result, _, _ = mlir.emit_python_callback(
        ctx, _callback, None, args, ctx.avals_in, ctx.avals_out,
        has_side_effect=False)
    return result


def _spsolve_jvp_lhs(data_dot, data, b, **kwds):
    # d/dM M^-1 b = M^-1 M_dot M^-1 b
    p = spsolve(data, b, **kwds)
    q = matvec(data_dot, p, **kwds)
    return -spsolve(data, q, **kwds)


def _spsolve_jvp_rhs(b_dot, data, b, **kwds):
    # d/db M^-1 b = M^-1 b_dot
    return spsolve(data, b_dot, **kwds)

def _spsolve_transpose(ct, data, b, **kwds):
    if ad.is_undefined_primal(b):
        if isinstance(ct, ad.Zero):
            rhs = jnp.zeros(b.aval.shape, b.aval.dtype)
        else:
            rhs = ct
        kwds['transpose'] = not kwds['transpose']
        ct_out = spsolve(data, rhs, **kwds)
        return data, ct_out
    else:
        # Should never reach here, because JVP is linear wrt data.
        raise NotImplementedError("spsolve transpose with respect to data")

_spsolve_p = core.Primitive('cpu_spsolve')
_spsolve_p.def_impl(functools.partial(xla.apply_primitive, _spsolve_p))
_spsolve_p.def_abstract_eval(_abstract_eval)
ad.defjvp(_spsolve_p, _spsolve_jvp_lhs, _spsolve_jvp_rhs)
ad.primitive_transposes[_spsolve_p] = _spsolve_transpose
mlir.register_lowering(_spsolve_p, _spsolve_cpu_lowering, platform='cpu')

def spsolve(data, b, *, solver_num: int, transpose=False, n_cpu=None, _mode=0):
    """A sparse direct solver, based on scipy.sparse.linalg.spsolve."""
    if n_cpu == 0:
        n_cpu = cpu_count()
    return _spsolve_p.bind(data, b, solver_num=solver_num,
                           transpose=transpose, n_cpu=n_cpu, _mode=_mode)

# Batching _mode`s:
# 0 - no vectorization
# 1 - data is vectorized
# 2 - b is vectorized
# 3 - data and b are vectorized
# 4 - data and b are vectorized, b has 2 batch dims # needed for jax.hessian
# TODO: if dims are higher, extend existing axes with values, compute evrth, then return to original shape
def _batch_template(func, vals, axes, **kwargs):
    data, b = vals
    ad, ab = axes
    res = jnp.zeros_like(b)

    assert b.ndim < 4, 'Only two batch dimensions are supported'
    assert data.ndim < 3, 'Only one batch dimension is supported'

    if ad is not None and ab is None:
        data = jnp.moveaxis(data, ad, 0)
        kwargs['_mode'] = 1

    elif ad is None and ab is not None and b.ndim == 2:
        b = jnp.moveaxis(b, ab, 0)
        kwargs['_mode'] = 2

    elif ad is not None and ab is not None and b.ndim == 2:
        assert data.shape[ad] == b.shape[ab], 'Matrices and bs should have same batching size'
        data = jnp.moveaxis(data, ad, 0)
        b = jnp.moveaxis(b, ab, 0)
        kwargs['_mode'] = 3

    elif data.ndim == 2 and b.ndim == 3:
        b = jnp.moveaxis(b, ab, 0)
        assert b.shape[1] == data.shape[0]
        kwargs['_mode'] = 4

    else:
        raise NotImplementedError('Batching of spsolve with arguments shapes: '
                                  f'{data.shape=}, {b.shape=}')

    res = func(data, b, **kwargs)
    return res, 0

_spsolve_batch = functools.partial(_batch_template, spsolve)
batching.primitive_batchers[_spsolve_p] = _spsolve_batch
_matvec_batch = functools.partial(_batch_template, matvec)
batching.primitive_batchers[_matvec_p] = _matvec_batch
