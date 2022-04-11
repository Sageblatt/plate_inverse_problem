import jax
import jax.numpy as jnp

MODULI_INDICES = ["11", "12", "16", "22", "26", "66"]

def isotropic_to_full(isotropic_params, *args):

    D = isotropic_params[0]
    nu = isotropic_params[1]
    beta = isotropic_params[2]

    Ds = jnp.array([D, nu * D, 0.0, D, 0.0, D * (1.0 - nu)])
    betas = jnp.full_like(Ds, beta)

    return Ds, betas

def orthotropic(orthotropic_params, *args):

    D_11 = orthotropic_params[0] # D_11 = E_1 h**3/12(1 - nu_12*nu_21)
    nu_12 = orthotropic_params[1]
    E_ratio = orthotropic_params[2] # E1/E2
    D_66 = orthotropic_params[3] # D_66 = G_12*h**3/12
    beta = orthotropic_params[4] # loss factor

    nu_21 = E_ratio*nu_12
    D_12 = nu_21*D_11
    D_22 = D_11/E_ratio

    Ds = jnp.array([D_11, D_12, 0.0, D_22, 0.0, D_66])
    betas = jnp.full_like(Ds, beta)

    return Ds, betas

def four_parameter_fd_isotropic(params, omega):

    D0 = params[0]
    nu = params[1]
    # four-parameter damping model
    a = params[2]
    b = params[3]
    alpha = params[4] 

    fd_part = (1. + a*(1.j*omega)**alpha)/(1. + b*(1.j*omega)**alpha)
    beta = fd_part.imag
    D = fd_part.real*D0

    Ds = jnp.array([D, nu * D, 0.0, D, 0.0, D * (1.0 - nu)])
    betas = jnp.full_like(Ds, beta)

    return Ds, betas
    
#def orthotropic_four_parameter_fd(params, omega):
#    D_11 = params[0] # D_11 = E_1 h**3/12(1 - nu_12*nu_21)
#    nu_12 = params[1]
#    E_ratio = params[2] # E1/E2
#    D_66_0 = params[3] # D_66 = G_12*h**3/12
#
#    # four-parameter damping model
#    a = params[4]
#    b = params[5]
#    alpha = params[6] 
#
#    nu_21 = E_ratio*nu_12
#    D_12 = nu_21*D_11
#    D_22 = D_11/E_ratio
#
#    fd_part = (1. + a*(1.j*omega)**alpha)/(1. + b*(1.j*omega)**alpha)
#
#    beta = fd_part.imag
#    D_66 = D_66_0*fd_part.real
#
#    Ds = jnp.array([D_11, D_12, 0.0, D_22, 0.0, D_66])
#    betas = jnp.zeros_like(Ds)
#    betas = betas.at[-1] = beta
#
#    return Ds, betas
