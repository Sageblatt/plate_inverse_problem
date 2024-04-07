"""Modified CPU version of jax.experimental.sparse.linalg.spsolve with batching"""
import functools
from multiprocessing import Pool, cpu_count

import jax
from jax import core, tree_util
from jax.interpreters import ad, batching, mlir, xla
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scikits import umfpack


# Ensure that jax uses CPU
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

FAMILIES = {(np.float64, np.int32): 'di',
            (np.float64, np.int64): 'dl',
            (np.complex128, np.int32): 'zi',
            (np.complex128, np.int64): 'zl'}

class SolverState:
    def __init__(self, N: int, indices: np.ndarray, mat_dtype):
        fam_idx = (mat_dtype, indices.dtype.type)
        if not fam_idx in FAMILIES:
            raise TypeError(f'Invalid dtypes of arguments: got {fam_idx=}, '
                            f'expected one of {FAMILIES.keys()}')

        fam = FAMILIES[fam_idx]
        self.context = umfpack.UmfpackContext(fam)
        self.transpose_context = umfpack.UmfpackContext(fam)

        self.coo_indices = indices
        self.dtype = mat_dtype
        self.N = N

        data = np.linspace(1.0, 42.0, indices.shape[0], dtype=mat_dtype)
        self.mat = coo_matrix((data, (indices[:, 0], indices[:, 1])), (N, N),
                              dtype=mat_dtype, copy=True).tocsc()
        self.mat_T = coo_matrix((data, (indices[:, 1], indices[:, 0])), (N, N),
                                dtype=mat_dtype, copy=True).tocsc()

        self.index = self.mat.indices
        self.index_T = self.mat_T.indices

        self.indptr = self.mat.indptr
        self.indptr_T = self.mat_T.indptr

        m1 = self.mat.tocoo()
        m1_T = self.mat_T.tocoo()

        coo_indices = np.vstack((m1.row, m1.col), dtype=np.int32).T
        coo_indices_T = np.vstack((m1_T.row, m1_T.col), dtype=np.int32).T

        helper = np.arange(0, indices.shape[0])
        ind = np.where(np.equal(indices[:, None], coo_indices[None, :]))[1]
        inv_mask = ind[np.where(np.diff(ind)==0)]
        self.mask = np.copy(inv_mask)
        self.mask[inv_mask] = helper

        indices_T = indices[:, ::-1]
        ind = np.where(np.equal(indices_T[:, None], coo_indices_T[None, :]))[1]
        inv_mask_T = ind[np.where(np.diff(ind)==0)]
        self.mask_T = np.copy(inv_mask_T)
        self.mask_T[inv_mask_T] = helper

        self.context.symbolic(self.mat)
        self.transpose_context.symbolic(self.mat_T)

    def solve(self, data, b, transpose):
        if transpose:
            ctx = self.transpose_context
            indx = self.index_T
            indptr = self.indptr_T
            data = data[self.mask_T]
        else:
            ctx = self.context
            indx = self.index
            indptr = self.indptr
            data = data[self.mask]

        # numeric ---------------
        ctx.free_numeric()

        assert ctx._symbolic is not None

        if ctx.isReal:
            mtx = data.copy()
            status, ctx._numeric\
                    = ctx.funs.numeric(indptr, indx, mtx,
                                       ctx._symbolic,
                                       ctx.control, ctx.info)
        else:
            real, imag = data.real.copy(), data.imag.copy()
            status, ctx._numeric\
                    = ctx.funs.numeric(indptr, indx,
                                       real, imag,
                                       ctx._symbolic,
                                       ctx.control, ctx.info)

        # -----------------------------------------
        sys = umfpack.UMFPACK_A
        sol = np.zeros_like(b)
        if ctx.isReal:
            status = ctx.funs.solve(sys, indptr, indx, data, sol, b,
                                    ctx._numeric, ctx.control, ctx.info)
        else:
            mreal, mimag = data.real.copy(), data.imag.copy()
            sreal, simag = np.zeros_like(b, dtype=np.float64), np.zeros_like(b, dtype=np.float64)
            rreal, rimag = b.real.copy(), b.imag.copy()
            status = ctx.funs.solve(sys, indptr, indx,
                                    mreal, mimag, sreal, simag, rreal, rimag,
                                    ctx._numeric, ctx.control, ctx.info)
            sol.real, sol.imag = sreal, simag

        return sol

    def matvec(self, mat, vec, transpose):
        if transpose:
            mat = mat[self.mask_T]
            A = csc_matrix((mat, self.index_T, self.indptr_T),
                           shape=(vec.shape[0], vec.shape[0]))
        else:
            mat = mat[self.mask]
            A = csc_matrix((mat, self.index, self.indptr),
                           shape=(vec.shape[0], vec.shape[0]))
        return A.dot(vec)


def flatten_solver_state(obj):
    children = (obj.N, obj.coo_indices, obj.dtype)
    aux_data = None
    return children, aux_data

def unflatten_solver_state(aux_data, children):
    obj = SolverState(*children)
    return obj

tree_util.register_pytree_node(SolverState, flatten_solver_state, unflatten_solver_state)

#---------------------

# Abstract eval for both matvec and spsolve
def _abstract_eval(data, b, *, s_state, transpose=False, n_cpu=None, _mode=0):
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
    if not isinstance(s_state, SolverState):
        raise TypeError('invalid type of `s_state` argument, expected '
                        f'`jax_plate.Sparse.SolverState`, got {type(s_state)}')
    if _mode in (0, 2, 3, 4):
        return b
    elif _mode == 1:
        return core.ShapedArray((data.shape[0], b.shape[0]), b.dtype)
    else:  # should't reach here as handling is done in _spsolve_batch
        raise NotImplementedError()

# ----------------

# custom matvec primitive ----------------------------------------------------
def matvec(mat, vec, *, s_state, transpose=False, n_cpu=None, _mode=0):
    return _matvec_p.bind(mat, vec, s_state=s_state, transpose=transpose,
                          n_cpu=n_cpu, _mode=_mode)

def _matvec_parallel_func(data, b, s_state, transpose):
    return s_state.matvec(data, b, transpose)

def _matvec_cpu_lowering(ctx, data, b, *, s_state, transpose, n_cpu, _mode):
    args = [data, b]

    def _callback(data, b, **kwargs):
        if _mode == 0:
            return (s_state.matvec(data, b, transpose),)

        if n_cpu != 1: # Multiprocessing is not functional
            if n_cpu == 0:
                _n_cpu = cpu_count()
            else:
                _n_cpu = n_cpu
            pool = Pool(_n_cpu)

            _pfunc0 = functools.partial(_matvec_parallel_func, s_state=s_state,
                                        transpose=transpose)

            if _mode == 1:
                _pfunc = functools.partial(_pfunc0, b=b)
                _res = pool.starmap(_pfunc, data)
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 2:
                _pfunc = functools.partial(_pfunc0, data=data)
                _res = pool.starmap(_pfunc, b)
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 3:
                _pfunc = _pfunc0
                _res = pool.starmap(_pfunc, zip(data, b))
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 4: # lazy implementation, may be better without loop
                res = []
                _pfunc = _pfunc0
                for i in range(b.shape[0]):
                    _res = pool.starmap(_pfunc, zip(data, b[i]))
                    res.append(_res)
                res = np.array(res, dtype=b.dtype)
            else:
                raise NotImplementedError()

            pool.close()
            pool.join()

        else:
            if _mode == 1:
                res = np.zeros((data.shape[0], b.shape[0]), dtype=b.dtype)
                for i in range(data.shape[0]):
                    res[i, :] = s_state.matvec(data[i, :], b, transpose)
            elif _mode == 2:
                res = np.zeros_like(b)
                for i in range(b.shape[0]):
                    res[i, :] = s_state.matvec(data, b[i, :], transpose)
            elif _mode == 3:
                res = np.zeros_like(b)
                for i in range(b.shape[0]):
                    res[i, :] = s_state.matvec(data[i, :], b[i, :], transpose)
            elif _mode == 4:
                res = np.zeros_like(b)
                for i in range(b.shape[1]):
                    for j in range(b.shape[0]):
                        res[j, i, :] = s_state.matvec(data[i, :],
                                                      b[j, i, :], transpose).astype(b.dtype)
            else:
                raise NotImplementedError()

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
        row, col = kwargs['s_state'].coo_indices[:, 0], kwargs['s_state'].coo_indices[:, 1]
        return ct[..., row] * v[..., col], v


_matvec_p = core.Primitive('custom_matvec')
_matvec_p.def_impl(functools.partial(xla.apply_primitive, _matvec_p))
_matvec_p.def_abstract_eval(_abstract_eval)
ad.defjvp(_matvec_p, _matvec_jvp_mat, _matvec_jvp_vec)
ad.primitive_transposes[_matvec_p] = _matvec_transpose
mlir.register_lowering(_matvec_p, _matvec_cpu_lowering, platform='cpu')

#--------------------------------------------------------------------

# custom spsolve --------------------------------------------------------------
def _parallel_loop_func(data, b, s_state, transpose):
    return s_state.solve(data, b, transpose)


def _spsolve_cpu_lowering(ctx, data, b, *, s_state, transpose, n_cpu, _mode):
    args = [data, b]

    def _callback(data, b, **kwargs):
        if _mode == 0:
            return (s_state.solve(data, b, transpose),)

        if n_cpu != 1:
            if n_cpu == 0:
                _n_cpu = cpu_count()
            else:
                _n_cpu = n_cpu
            pool = Pool(_n_cpu)

            _pfunc0 = functools.partial(_parallel_loop_func, s_state=s_state,
                                        transpose=transpose)

            if _mode == 1:
                _pfunc = functools.partial(_pfunc0, b=b)
                _res = pool.starmap(_pfunc, data)
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 2:
                _pfunc = functools.partial(_pfunc0, data=data)
                _res = pool.starmap(_pfunc, b)
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 3:
                _pfunc = _pfunc0
                _res = pool.starmap(_pfunc, zip(data, b))
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 4: # lazy implementation, may be better without loop
                res = []
                _pfunc = _pfunc0
                for i in range(b.shape[0]):
                    _res = pool.starmap(_pfunc, zip(data, b[i]))
                    res.append(_res)
                res = np.array(res, dtype=b.dtype)
            else:
                raise NotImplementedError()

            pool.close()
            pool.join()

        else:
            if _mode == 1:
                res = np.zeros((data.shape[0], b.shape[0]), dtype=b.dtype)
                for i in range(data.shape[0]):
                    res[i, :] = s_state.solve(data[i, :], b, transpose)
            elif _mode == 2:
                res = np.zeros_like(b)
                for i in range(b.shape[0]):
                    res[i, :] = s_state.solve(data, b[i, :], transpose)
            elif _mode == 3:
                res = np.zeros_like(b)
                for i in range(b.shape[0]):
                    res[i, :] = s_state.solve(data[i, :], b[i, :], transpose)
            elif _mode == 4:
                res = np.zeros_like(b)
                for i in range(b.shape[1]):
                    for j in range(b.shape[0]):
                        res[j, i, :] = s_state.solve(data[i, :],
                                                     b[j, i, :], transpose).astype(b.dtype)
            else:
                raise NotImplementedError()

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


def spsolve(data, b, *, s_state, transpose=False, n_cpu=None, _mode=0):
    """A sparse direct solver, based on scipy.sparse.linalg.spsolve."""
    return _spsolve_p.bind(data, b, s_state=s_state,
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
