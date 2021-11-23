import jax
import jax.numpy as jnp

import numpy as np

from collections import namedtuple


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
    print(lams)
    print(eigenvectors @ g / np.linalg.norm(g))
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

    print(m, k)
    return sd, lam, predicted_improvement


def get_model_newt(f):
    f_value_and_gradient = jax.jit(jax.value_and_grad(f))
    f_hessian = jax.jit(jax.hessian(f))

    def _update(x):
        return (*f_value_and_gradient(x), f_hessian(x))

    return _update


# @jax.jit
# def get_model_lvmq(u_ref, w, dv, p,):
#    r = get_residual(u_ref, w, dv, p)
#    n = r.shape[0]
#    J = res_jacobian(u_ref, w, dv, p)
#    return r.T@r/n, r.T@J/n, J.T@J/n
trOptResult = namedtuple(
    "trOptResult",
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
    stall_fraction=0.15,
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

    steps_to_stall = int(N_steps * stall_fraction)
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
        else:
            if rel_improvement >= 3.0 / 4.0 and lam > 0.0:
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

        print(delta)

        if cur_f < 1e-7:
            status = "Converged"
            break
        elif steps_without_update >= steps_to_stall:
            status = "Stalled"
            break

    return trOptResult(x, cur_f, f_history, x_history, grad_history, k, status)


gdOptResult = namedtuple(
    "gdOptResult", ["x", "f", "f_history", "x_history", "grad_history", "niter",]
)


def optimize_gd(f, x_0, N_steps, h, f_min=1e-8):
    value_and_gradient = jax.jit(jax.value_and_grad(f))

    x = x_0

    x_history = []
    f_history = []
    grad_history = []

    for k in range(N_steps):
        cur_f, g = value_and_gradient(x)

        x_history.append(x)
        f_history.append(cur_f)
        grad_history.append(g)

        if cur_f <= f_min:
            break

        x -= h * g

    return gdOptResult(x, cur_f, f_history, x_history, grad_history, k,)
