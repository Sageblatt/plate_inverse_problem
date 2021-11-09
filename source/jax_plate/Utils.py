import numpy as np
import jax
import jax.numpy as jnp

# TODO check if it is possible in jax to use LA and grad on dictionary (google pyTree)
def isotropic_to_full(isotropic_params):

    D = isotropic_params[0]
    nu = isotropic_params[1]
    beta = isotropic_params[2]

    Ds = jnp.array([D, nu * D, 0.0, D, 0.0, D * (1.0 - nu)])
    betas = jnp.full_like(Ds, beta)

    return Ds, betas
