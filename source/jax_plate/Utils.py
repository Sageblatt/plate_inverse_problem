import numpy as np
import matplotlib.pyplot as plt

def plot_afc_radial(freqs, afc, fig, axs, **line_kwargs):

    afc_module = np.linalg.norm(afc, axis=1, ord=2)
    afc_phase_shift = np.arctan2(afc[:, 0], afc[:, 1]) / np.pi

    axs[0].set_yscale("log")
    axs[0].plot(freqs, afc_module, **line_kwargs)
    axs[0].set_title(r"$\|u\|$")
    axs[0].set_xlabel("$f,\\ Hz$")
    axs[0].grid(True)

    axs[1].plot(freqs, afc_phase_shift, **line_kwargs)
    axs[1].set_title(r"$\frac{\delta(\varphi)}{\pi}$")
    axs[1].grid(True)
    axs[1].set_xlabel("$f,\\ Hz$")
    axs[1].legend()
    return fig, axs


def plot_afc_complex(freqs, afc, fig, axs, **line_kwargs):

    axs[0].plot(freqs, afc[:, 0], **line_kwargs)
    axs[0].set_title(r"$\Re(u)$")
    axs[0].set_xlabel("$f,\\ Hz$")
    axs[0].grid(True)

    axs[1].plot(freqs, afc[:, 1], **line_kwargs)
    axs[1].set_title(r"$\Im(u)$")
    axs[1].set_xlabel("$f,\\ Hz$")
    axs[1].grid(True)
    axs[1].legend()
    return fig, axs


def plot_afc(freqs, afc, fig=None, kind="Radial", **line_kwargs):
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
        return plot_afc_radial(freqs, afc, fig, axs, **line_kwargs)
    elif kind == "Complex":
        return plot_afc_complex(freqs, afc, fig, axs, **line_kwargs)
