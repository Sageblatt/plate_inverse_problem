"""Modified CPU version of jax.experimental.sparse.linalg.spsolve with batching"""
import functools

import jax
import jax.numpy as jnp

from jax.experimental import sparse
from jax import core
from jax.interpreters import ad, batching, mlir, xla

import numpy as np
from scipy.sparse import csr_matrix, linalg


# Ensure that jax uses CPU
jax.config.update('jax_platform_name', 'cpu')


def _spsolve_abstract_eval(data, indices, indptr, b, *, permc_spec, use_umfpack):
  if data.dtype != b.dtype:
    raise ValueError(f"data types do not match: {data.dtype=} {b.dtype=}")
  if not (jnp.issubdtype(indices.dtype, jnp.integer) and jnp.issubdtype(indptr.dtype, jnp.integer)):
    raise ValueError(f"index arrays must be integer typed; got {indices.dtype=} {indptr.dtype=}")
  if not data.ndim == indices.ndim == indptr.ndim == b.ndim == 1:
    raise ValueError("Arrays must be one-dimensional. "
                     f"Got {data.shape=} {indices.shape=} {indptr.shape=} {b.shape=}")
  if indptr.size != b.size + 1 or  data.shape != indices.shape:
    raise ValueError(f"Invalid CSR buffer sizes: {data.shape=} {indices.shape=} {indptr.shape=}")
  if permc_spec not in ['NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD']:
    raise ValueError(f"{permc_spec=} not valid, must be one of "
                     "['NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A', 'COLAMD']")
  use_umfpack = bool(use_umfpack)
  return b


def _spsolve_cpu_lowering(ctx, data, indices, indptr, b, permc_spec, use_umfpack):
  args = [data, indices, indptr, b]

  def _callback(data, indices, indptr, b, **kwargs):
    A = csr_matrix((data, indices, indptr), shape=(b.size, b.size), copy=True) # TODO: maybe we can avoid copying
    A.eliminate_zeros()
    return (linalg.spsolve(A, b, permc_spec=permc_spec, use_umfpack=use_umfpack).astype(b.dtype),)

  result, _, _ = mlir.emit_python_callback(
      ctx, _callback, None, args, ctx.avals_in, ctx.avals_out,
      has_side_effect=False)
  return result


def _spsolve_jvp_lhs(data_dot, data, indices, indptr, b, **kwds):
    # d/dM M^-1 b = M^-1 M_dot M^-1 b
    p = spsolve(data, indices, indptr, b, **kwds)
    q = sparse.csr_matvec_p.bind(data_dot, indices, indptr, p,
                                 shape=(indptr.size - 1, len(b)),
                                 transpose=False)
    return -spsolve(data, indices, indptr, q, **kwds)


def _spsolve_jvp_rhs(b_dot, data, indices, indptr, b, **kwds):
    # d/db M^-1 b = M^-1 b_dot
    return spsolve(data, indices, indptr, b_dot, **kwds)


def _csr_transpose(data, indices, indptr):
  # Transpose of a square CSR matrix
  m = indptr.size - 1
  row = jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1
  row_T, indices_T, data_T = jax.lax.sort((indices, row, data), num_keys=2)
  indptr_T = jnp.zeros_like(indptr).at[1:].set(
      jnp.cumsum(jnp.bincount(row_T, length=m)).astype(indptr.dtype))
  return data_T, indices_T, indptr_T


def _spsolve_transpose(ct, data, indices, indptr, b, **kwds):
  assert not ad.is_undefined_primal(indices)
  assert not ad.is_undefined_primal(indptr)
  if ad.is_undefined_primal(b):
    # TODO(jakevdp): can we do this without an explicit transpose?
    data_T, indices_T, indptr_T = _csr_transpose(data, indices, indptr)
    ct_out = spsolve(data_T, indices_T, indptr_T, ct, **kwds)
    return data, indices, indptr, ct_out
  else:
    # Should never reach here, because JVP is linear wrt data.
    raise NotImplementedError("spsolve transpose with respect to data")


_spsolve_p = core.Primitive('cpu_spsolve')
_spsolve_p.def_impl(functools.partial(xla.apply_primitive, _spsolve_p))
_spsolve_p.def_abstract_eval(_spsolve_abstract_eval)
ad.defjvp(_spsolve_p, _spsolve_jvp_lhs, None, None, _spsolve_jvp_rhs)
ad.primitive_transposes[_spsolve_p] = _spsolve_transpose
mlir.register_lowering(_spsolve_p, _spsolve_cpu_lowering, platform='cpu')


def spsolve(data, indices, indptr, b, permc_spec='COLAMD', use_umfpack=True):
  """A sparse direct solver, based completely on scipy.sparse.linalg.spsolve."""
  return _spsolve_p.bind(data, indices, indptr, b, permc_spec=permc_spec, use_umfpack=use_umfpack)


def _spsolve_batch(vals, axes, **kwargs):
    Ad, Ai, Ap, b = vals
    aAd, aAi, aAp, ab = axes
    res = jnp.zeros_like(b)

    assert b.ndim < 3, 'Only one batch dimension is supported'

    if aAd is None and ab is not None:
        b = jnp.moveaxis(b, ab, 0)
        for i in range(0, b.shape[0]):
            res = res.at[i, :].set(spsolve(Ad, Ai, Ap, b[i, :], **kwargs))

    elif aAd is not None and ab is not None and aAi is None:
        assert aAi is None, 'Batched arrays should have the same structure'
        assert aAp is None, 'Batched arrays should have the same structure'

        assert Ad.shape[aAd] == b.shape[ab]

        Ad = jnp.moveaxis(Ad, aAd, 0)
        b = jnp.moveaxis(b, ab, 0)

        for i in range(0, Ad.shape[0]):
            # TODO: use the fact that the sparsity pattern should be the same
            # And maybe this loop can be vectorized across CPU cores
            res = res.at[i, :].set(spsolve(Ad[i, :], Ai, Ap, b[i, :], **kwargs))
    else:
        raise NotImplementedError('Matrices with different indices and indptrs')

    return res, 0

batching.primitive_batchers[_spsolve_p] = _spsolve_batch
