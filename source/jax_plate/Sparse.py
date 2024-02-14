"""Modified CPU version of jax.experimental.sparse.linalg.spsolve with batching"""
import functools

import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import ad, batching, mlir, xla
from jax.experimental import sparse

import numpy as np
from scipy.sparse import csr_matrix, linalg

from multiprocessing import Pool, cpu_count

# Ensure that jax uses CPU
jax.config.update('jax_platform_name', 'cpu')


def _spsolve_abstract_eval(data, indices, b, *, permc_spec, use_umfpack, n_cpu, _mode):
    if data.dtype != b.dtype:
        raise ValueError(f"data types do not match: {data.dtype=} {b.dtype=}")
    if not (jnp.issubdtype(indices.dtype, jnp.integer)):
        raise ValueError(f"index arrays must be integer typed; got {indices.dtype=}")
    if permc_spec not in ['NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD']:
        raise ValueError(f"{permc_spec=} not valid, must be one of "
                         "['NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD']")
    use_umfpack = bool(use_umfpack)
    if n_cpu is not None:
        if not isinstance(n_cpu, int):
            raise ValueError(f'invalid type of `n_cpu` argument, expected `int` or `None`, '
                             f'got {type(n_cpu)}')
        if n_cpu < 0:
            raise ValueError('n_cpu argument should be non-negative')
    if not isinstance(_mode, int):
        raise ValueError('invalid type of `_mode` argument, expected `int`, got '
                         f'{type(_mode)}')
    if _mode in (0, 2, 3, 4):
        return b
    elif _mode == 1:
        return core.ShapedArray((data.shape[0], b.shape[0]), b.dtype)
    else:  # should't reach here as handling is done in _spsolve_batch
        raise NotImplementedError()


def _parallel_loop_func(data, b, indx, permc_spec, use_umfpack):
    A = csr_matrix((data, indx.T), shape=(b.shape[0], b.shape[0]),
                                             dtype=b.dtype)
    A.eliminate_zeros()
    return linalg.spsolve(A, b, permc_spec=permc_spec,
                               use_umfpack=use_umfpack).astype(b.dtype)


def _spsolve_cpu_lowering(ctx, data, indices, b, permc_spec, use_umfpack, n_cpu, _mode):
    args = [data, indices, b]

    def _callback(data, indices, b, **kwargs):
        if _mode == 0:
            A = csr_matrix((data, indices.T),
                           shape=(b.shape[1], b.shape[1]), dtype=b.dtype)
            A.eliminate_zeros()
            return (linalg.spsolve(A, b, permc_spec=permc_spec,
                                 use_umfpack=use_umfpack).astype(b.dtype),)

        if n_cpu is not None:
            if n_cpu == 0:
                _n_cpu = cpu_count()
            else:
                _n_cpu = n_cpu
            pool = Pool(_n_cpu)

            if _mode == 1:
                _pfunc = functools.partial(_parallel_loop_func, indx=indices,
                                           b=b, permc_spec=permc_spec,
                                           use_umfpack=use_umfpack)
                _res = pool.starmap(_pfunc, data)
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 2:
                _pfunc = functools.partial(_parallel_loop_func, data=data, indx=indices,
                                           permc_spec=permc_spec,
                                           use_umfpack=use_umfpack)
                _res = pool.starmap(_pfunc, b)
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 3:
                _pfunc = functools.partial(_parallel_loop_func, indx=indices,
                                           permc_spec=permc_spec,
                                           use_umfpack=use_umfpack)
                _res = pool.starmap(_pfunc, zip(data, b))
                res = np.array(_res, dtype=b.dtype)
            elif _mode == 4: # lazy implementation, may be better without loop
                res = []
                _pfunc = functools.partial(_parallel_loop_func, indx=indices,
                                           permc_spec=permc_spec,
                                           use_umfpack=use_umfpack)
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
                    A = csr_matrix((data[i, :], indices.T),
                                   shape=(b.shape[0], b.shape[0]), dtype=b.dtype)
                    A.eliminate_zeros()
                    res[i, :] = linalg.spsolve(A, b, permc_spec=permc_spec,
                                               use_umfpack=use_umfpack).astype(b.dtype)
            elif _mode == 2:
                res = np.zeros_like(b)
                A = csr_matrix((data, indices.T),
                               shape=(b.shape[1], b.shape[1]), dtype=b.dtype)
                A.eliminate_zeros()
                for i in range(b.shape[0]):
                    res[i, :] = linalg.spsolve(A, b[i, :], permc_spec=permc_spec,
                                               use_umfpack=use_umfpack).astype(b.dtype)
            elif _mode == 3:
                res = np.zeros_like(b)
                for i in range(b.shape[0]):
                    A = csr_matrix((data[i, :], indices.T),
                                   shape=(b.shape[1], b.shape[1]), dtype=b.dtype)
                    A.eliminate_zeros()
                    res[i, :] = linalg.spsolve(A, b[i, :], permc_spec=permc_spec,
                                               use_umfpack=use_umfpack).astype(b.dtype)
            elif _mode == 4:
                res = np.zeros_like(b)
                for i in range(b.shape[1]):
                    A = csr_matrix((data[i, :], indices.T),
                                   shape=(b.shape[2], b.shape[2]), dtype=b.dtype)
                    A.eliminate_zeros()
                    for j in range(b.shape[0]):
                        res[j, i, :] = linalg.spsolve(A, b[j, i, :],
                                                      permc_spec=permc_spec,
                                                      use_umfpack=use_umfpack).astype(b.dtype)
            else:
                raise NotImplementedError()

        return (res,)

    result, _, _ = mlir.emit_python_callback(
        ctx, _callback, None, args, ctx.avals_in, ctx.avals_out,
        has_side_effect=False)
    return result


def _spsolve_jvp_lhs(data_dot, data, indices, b, **kwds):
    # d/dM M^-1 b = M^-1 M_dot M^-1 b
    p = spsolve(data, indices, b, **kwds)
    md = kwds['_mode']
    if md == 0:
        A = sparse.BCOO((data_dot, indices), shape=(b.shape[0], b.shape[0]))
        q = A @ p
    elif md == 1:  # TODO: implement.
        A = sparse.empty((data.shape[0], b.shape[0], b.shape[0]),
                         dtype=b.dtype, index_dtype='int32', sparse_format='bcoo',
                         n_batch=1, n_dense=0, nse=data.shape[1])
        A.data = data_dot
        A.indices = jnp.tile(indices, (data.shape[0], 1, 1))
        q = sparse.bcoo_dot_general(A, p,
                                    dimension_numbers=(((1), (0)), ((0), ())))
    elif md == 2:
        A = sparse.empty((b.shape[1], b.shape[1]),
                         dtype=b.dtype, index_dtype='int32', sparse_format='bcoo',
                         n_batch=0, n_dense=0, nse=data.shape[1])
        A.data = data_dot
        A.indices = jnp.tile(indices, (data.shape[0], 1, 1))
        q = sparse.bcoo_dot_general(A, p,
                                    dimension_numbers=(((0), (1)), ((), (0))))
    elif md == 3:
        A = sparse.empty((data.shape[0], b.shape[1], b.shape[1]),
                         dtype=b.dtype, index_dtype='int32', sparse_format='bcoo',
                         n_batch=1, n_dense=0, nse=data.shape[1])
        A.data = data_dot
        A.indices = jnp.tile(indices, (data.shape[0], 1, 1))
        q = sparse.bcoo_dot_general(A, p,
                                    dimension_numbers=(((1), (1)), ((0), (0))))
    elif md == 4:
        A = sparse.empty((data.shape[0], b.shape[2], b.shape[2]),
                         dtype=b.dtype, index_dtype='int32', sparse_format='bcoo',
                         n_batch=1, n_dense=0, nse=data.shape[1])
        A.data = data_dot
        A.indices = jnp.tile(indices, (data.shape[0], 1, 1))
        q = sparse.bcoo_dot_general(A, p,
                                    dimension_numbers=(((1), (1)), ((0), (0, 1))))
    else:  # shouldn't reach here as handling is done in _spsolve_batch
        return NotImplementedError()

    return -spsolve(data, indices, q, **kwds)


def _spsolve_jvp_rhs(b_dot, data, indices, b, **kwds):
    # d/db M^-1 b = M^-1 b_dot
    return spsolve(data, indices, b_dot, **kwds)


def _spsolve_transpose(ct, data, indices, b, **kwds):
    assert not ad.is_undefined_primal(indices)
    if ad.is_undefined_primal(b):
        indices_T = jnp.flip(indices, axis=1)
        if isinstance(ct, ad.Zero):
            rhs = jnp.zeros(b.aval.shape, b.aval.dtype)
        else:
            rhs = ct
        ct_out = spsolve(data, indices_T, rhs, **kwds)
        return data, indices, ct_out
    else:
        # Should never reach here, because JVP is linear wrt data.
        raise NotImplementedError("spsolve transpose with respect to data")


_spsolve_p = core.Primitive('cpu_spsolve')
_spsolve_p.def_impl(functools.partial(xla.apply_primitive, _spsolve_p))
_spsolve_p.def_abstract_eval(_spsolve_abstract_eval)
ad.defjvp(_spsolve_p, _spsolve_jvp_lhs, None, _spsolve_jvp_rhs)
ad.primitive_transposes[_spsolve_p] = _spsolve_transpose
mlir.register_lowering(_spsolve_p, _spsolve_cpu_lowering, platform='cpu')


def spsolve(data, indices, b, permc_spec='COLAMD', use_umfpack=True, n_cpu=None, _mode=0):
    """A sparse direct solver, based completely on scipy.sparse.linalg.spsolve."""
    return _spsolve_p.bind(data, indices, b, permc_spec=permc_spec, use_umfpack=use_umfpack,
                           n_cpu=n_cpu,_mode=_mode)


# Batching _mode`s:
# 0 - no vectorization
# 1 - data is vectorized
# 2 - b is vectorized
# 3 - data and b are vectorized
# 4 - data and b are vectorized, b has 2 batch dims # needed for jax.hessian
def _spsolve_batch(vals, axes, **kwargs):
    data, indices, b = vals
    ad, ai, ab = axes
    res = jnp.zeros_like(b)

    assert b.ndim < 4, 'Only two batch dimensions are supported'
    assert data.ndim < 3, 'Only one batch dimension is supported'
    assert indices.ndim == 2, '`indices` batching is not supported'
    assert ai is None, '`indices` batching is not supported'

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
                                  f'{data.shape=}, {indices.shape=}, {b.shape=}')

    res = spsolve(data, indices, b, **kwargs)
    return res, 0


batching.primitive_batchers[_spsolve_p] = _spsolve_batch
