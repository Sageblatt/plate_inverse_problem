from collections import namedtuple
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


class FixedParameterFunction:
    """
        Wrapper around existing parameter transform function,
        which fixes one of several parameters to constant value.
    """
    def __init__(self, function: Callable,
                 param_size: int,
                 fixed_indices: int | tuple,
                 fixed_values: float | tuple):
        """
        Constructor method.

        Parameters
        ----------
        function : Callable
            Function to be modified.
        param_size : int
            Overall amount of parameters of function.
        fixed_indices : int | tuple
            Fixed parameters' indexes, starting from 0.
        fixed_values : float | tuple
            Values to be fixed.

        Returns
        -------
        None
        """
        self.func = function
        self.array = np.zeros(param_size)
        self.free_idx = [i for i in range(param_size)]

        if isinstance(fixed_indices, int) or isinstance(fixed_values, float):
            assert isinstance(fixed_indices, int), f'got {type(fixed_indices)}'
            assert isinstance(fixed_values, float), f'got {type(fixed_values)}'
            self.array[fixed_indices] = fixed_values
            self.free_idx.remove(fixed_indices)
        else:
            assert len(fixed_indices) == len(fixed_values)
            for i, idx in enumerate(fixed_indices):
                self.array[idx] = fixed_values[i]
                self.free_idx.remove(idx)

        self.free_idx = jnp.array(self.free_idx)

    def __call__(self, params, *args):
        modified_params = jnp.array(self.array)
        modified_params = modified_params.at[self.free_idx].set(params)
        return self.func(modified_params, *args)


@jax.jit
def get_sd_and_norm(
    B, g, lam,
):
    B_cur = B + lam * jnp.eye(B.shape[0])
    sd = jax.scipy.linalg.solve(B_cur, -g,)
    pnorm = jnp.linalg.norm(sd)

    return sd, pnorm


def solve_trust_region_model(B, g, delta, rtol=1e-6, max_iter=100):
    lams, eigenvectors = np.linalg.eigh(B)
    sd = jnp.linalg.solve(B, -g)
    pnorm = jnp.linalg.norm(sd)
    if pnorm <= delta:
        predicted_improvement = -(
            g @ sd + 0.5 * sd.T @ B @ sd
        )  # based on the quadratic model
        if lams.min() >= 0:
            return sd, 0.0, predicted_improvement

    l_left = (-lams).max()
    l_left = 0.0 if l_left < 0 else l_left

    l_right = l_left + 1.0
    sd, pnorm = get_sd_and_norm(B, g, l_right,)

    # searching for lambda for which ||sd|| < delta
    for k in range(max_iter):
        if pnorm <= delta:
            break
        l_left = l_right
        l_right *= 2.0
        sd, pnorm = get_sd_and_norm(B, g, l_right,)

    assert pnorm <= delta, "Failed to find upper bound for lambda"

    lam = l_right

    for m in range(2 * max_iter):
        if pnorm <= delta and delta - pnorm <= delta * rtol:
            break
        lam = (l_right + l_left) / 2.0

        sd, pnorm = get_sd_and_norm(B, g, lam,)

        if pnorm < delta:
            l_right = lam
        else:
            l_left = lam

    if pnorm > delta:
        lam = l_right
        sd, pnorm = get_sd_and_norm(B, g, lam,)

    predicted_improvement = -(
        g @ sd + 0.5 * sd.T @ B @ sd
    )  # based on the quadratic model
    assert (
        predicted_improvement >= 0
    ), "Predicted improvement for quadratic model is negative"

    return sd, lam, predicted_improvement


def get_model_newt(f):
    gr = jax.grad(f)

    def val_gr(x):
        return f(x), gr(x)

    f_value_and_gradient = jax.jit(val_gr)
    f_hessian = jax.jit(jax.jacobian(gr))
    def _update(x):
        return (*f_value_and_gradient(x), f_hessian(x))

    return _update


# @jax.jit
# def get_model_lvmq(u_ref, w, dv, p,):
#    r = get_residual(u_ref, w, dv, p)
#    n = r.shape[0]
#    J = res_jacobian(u_ref, w, dv, p)
#    return r.T@r/n, r.T@J/n, J.T@J/n
optResult = namedtuple(
    "optResult",
    ["x", "f", "f_history", "x_history", "grad_history", "niter", "status"],
)


def optimize_trust_region(
    f,
    x_0,
    N_steps=10,
    delta_max=1.0,
    delta=None,
    eta=0.15,
    method="newt",
    steps_to_stall=10
):
    if delta is None:
        delta = delta_max / 10.0

    if eta > 0.25:
        raise ValueError(f"eta should be below 0.25; got {eta:f}")
    if eta < 0:
        raise ValueError(f"eta should be positive; got {eta:f}")

    if method == "newt":
        update_model = get_model_newt(f)
    else:
        raise NotImplementedError(f"Method <<{method}>> not implemented")

    f_history = []
    x_history = []
    grad_history = []

    model_update_required = True

    status = "Running"

    steps_without_update = 0

    x = x_0

    for k in range(N_steps):
        if model_update_required:
            cur_f, g, B = update_model(x)

        try:
            sd, lam, predicted_improvement = solve_trust_region_model(
                B, g, delta
            )  # find search direction from constrained minimization
        except AssertionError as e:
            status = str(e)
            break

        new_f = f(x + sd)

        rel_improvement = (cur_f - new_f) / predicted_improvement

        if rel_improvement < 1.0 / 4.0:
            delta /= 4.0
        elif rel_improvement >= 3.0 / 4.0 and lam > 0.0:
            # lam > 0 signals that we iterated in solve_trust_region_model and already found |p| ~ delta
            delta = jnp.minimum(2.0 * delta, delta_max)

        if rel_improvement >= eta:
            x += sd
            model_update_required = True
            steps_without_update = 0
        else:
            model_update_required = False
            steps_without_update += 1

        f_history.append(cur_f)
        x_history.append(x)
        grad_history.append(g)

        if cur_f < 1e-16:
            status = "Converged"
            break
        if steps_without_update >= steps_to_stall:
            status = "Stalled"
            break
    return optResult(x, cur_f, f_history, x_history, grad_history, k, status)


def optimize_gd(f, x_0, N_steps=100, h=0.01, f_min=1e-8):
    value_and_gradient = jax.jit(jax.value_and_grad(f))

    x = x_0

    x_history = []
    f_history = []
    grad_history = []
    status = 'Running'

    for k in range(N_steps):
        cur_f, g = value_and_gradient(x)

        x_history.append(x)
        f_history.append(cur_f)
        grad_history.append(g)

        if cur_f <= f_min:
            status = 'Converged'
            break

        x -= h * g

    return optResult(x, cur_f, f_history, x_history, grad_history, k, status)


def optimize_cd(f, x_0, N_steps=100, h=0.01, f_min=1e-8):
    value_and_gradient = jax.jit(jax.value_and_grad(f))

    x = x_0

    n = x_0.size
    assert n >= 2
    template = jnp.eye(n)

    x_history = []
    f_history = []
    grad_history = []
    status = 'Running'

    for k in range(N_steps):
        for i in range(n):
            cur_f, g = value_and_gradient(x)

            g *= template[i, :]

            x_history.append(x)
            f_history.append(cur_f)
            grad_history.append(g)

            if cur_f <= f_min:
                status = 'Converged'
                break

            x -= h * g

    return optResult(x, cur_f, f_history, x_history, grad_history, k, status)


def optimize_cd_mem(f, x_0, N_steps=100, h=0.01, f_min=1e-8):
    f_ = jax.jit(f)

    x = x_0

    n = x_0.size
    assert n >= 2
    template = jnp.reshape(jnp.where(jnp.eye(n)==0)[1], (n, n-1))

    grad_template = jnp.eye(n)

    x_history = []
    f_history = []
    grad_history = []
    status = 'Running'

    for k in range(N_steps):
        for i in range(n):
            fixed_f = FixedParameterFunction(f_, n, template[i], x[template[i]])
            cur_f, g = jax.value_and_grad(fixed_f)(x[fixed_f.free_idx])

            g *= grad_template[i]

            x_history.append(x)
            f_history.append(cur_f)
            grad_history.append(g)

            if cur_f <= f_min:
                status = 'Converged'
                break

            x -= h * g

    return optResult(x, cur_f, f_history, x_history, grad_history, k, status)


def optimize_cd_mem2(f, x_0, N_steps=100, h=0.01, f_min=1e-8):
    def fixed(x, i, other):
        return f(jnp.insert(other, i, x))

    f_ = jax.jit(fixed)

    x = x_0

    n = x_0.size
    assert n >= 2
    template = jnp.reshape(jnp.where(jnp.eye(n)==0)[1], (n, n-1))
    h_ = jnp.full(n, h)

    grad_template = jnp.eye(n)

    x_history = []
    f_history = []
    grad_history = []
    status = 'Running'

    for k in range(N_steps):
        for i in range(n):
            cur_f, g = jax.value_and_grad(f_)(x[i], i, x[template[i]])

            g *= grad_template[i]

            x_history.append(x)
            f_history.append(cur_f)
            grad_history.append(g)

            if cur_f <= f_min:
                status = 'Converged'
                break

            x -= h_[i] * g

            if f_(x[i], i, x[template[i]]) > f_history[-1]:
                h_[i] /= 5
                x = x_history[-1] - h_[i]*g


    return optResult(x, cur_f, f_history, x_history, grad_history, k, status)
