import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

MODULI_INDICES = ["11", "12", "16", "22", "26", "66"]

# TODO check if it is possible in jax to use LA and grad on dictionary (google pyTree)
def isotropic_to_full(isotropic_params):

    D = isotropic_params[0]
    nu = isotropic_params[1]
    beta = isotropic_params[2]

    Ds = jnp.array([D, nu * D, 0.0, D, 0.0, D * (1.0 - nu)])
    betas = jnp.full_like(Ds, beta)

    return Ds, betas


def plot_afc_radial(freqs, afc, label, fig, axs):

    afc_module = jnp.linalg.norm(afc, axis=1, ord=2)
    afc_phase_shift = jnp.arctan2(afc[:, 0], afc[:, 1]) / jnp.pi

    axs[0].set_yscale("log")
    axs[0].plot(freqs, afc_module, label=label)
    axs[0].set_title(r"$\|u\|$")
    axs[0].set_xlabel('$f,\\ Hz$')
    axs[0].grid(True)

    axs[1].plot(freqs, afc_phase_shift, label=label)
    axs[1].set_title(r"$\frac{\delta(\varphi)}{\pi}$")
    axs[1].grid(True)
    axs[1].set_xlabel('$f,\\ Hz$')
    axs[1].legend()
    return fig, axs


def plot_afc_complex(freqs, afc, label, fig, axs):

    axs[0].plot(freqs, afc[:, 0], label=label)
    axs[0].set_title(r"$\Re(u)$")
    axs[0].set_xlabel('$f,\\ Hz$')
    axs[0].grid(True)

    axs[1].plot(freqs, afc[:, 1], label=label)
    axs[1].set_title(r"$\Im(u)$")
    axs[1].set_xlabel('$f,\\ Hz$')
    axs[1].grid(True)
    axs[1].legend()
    return fig, axs


def plot_afc(freqs, afc, label, fig=None, kind="Radial"):
    if fig is None:
        if kind == "Radial":
            fig, axs = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, sharex=True)
        elif kind == "Complex":
            fig, axs = plt.subplots(
                figsize=(20, 10), nrows=1, ncols=2, sharex=True, sharey=True
            )
        else:
            raise ValueError(f"kind can cnly be 'Radial' or 'Complex', got {kind}")
    else:
        axs = fig.axes
        # we ognore the argument 'kind' if this is not hte first plot on the fig
        if axs[0].get_yscale() == "log":
            kind = "Radial"
        else:
            kind = "Complex"

    if kind == "Radial":
        return plot_afc_radial(freqs, afc, label, fig, axs)
    elif kind == "Complex":
        return plot_afc_complex(freqs, afc, label, fig, axs)
