import numpy as np
import matplotlib.pyplot as plt

def plot_fr_radial(freqs, fr, fig, axs, **line_kwargs):
    """Plots given frequency response on given frequency range as two subplots:
        AFC and PFC.
    """
    afc_module = np.abs(fr)
    afc_phase_shift = np.arctan2(np.real(fr), np.imag(fr)) / np.pi

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


def plot_fr_complex(freqs, fr, fig, axs, **line_kwargs):
    """Plots given frequency response on given frequency range as two subplots:
            Real and Imaginary parts of FR.
        """
    axs[0].plot(freqs, np.real(fr), **line_kwargs)
    axs[0].set_title(r"$\Re(u)$")
    axs[0].set_xlabel("$f,\\ Hz$")
    axs[0].grid(True)

    axs[1].plot(freqs, np.imag(fr), **line_kwargs)
    axs[1].set_title(r"$\Im(u)$")
    axs[1].set_xlabel("$f,\\ Hz$")
    axs[1].grid(True)
    axs[1].legend()
    return fig, axs


def plot_fr(freqs, fr, fig=None, kind="Radial", **line_kwargs):
    if fig is None:
        if kind == "Radial":
            fig, axs = plt.subplots(figsize=(20, 10), nrows=1, ncols=2, sharex=True)
            return plot_fr_radial(freqs, fr, fig, axs, **line_kwargs)
        elif kind == "Complex":
            fig, axs = plt.subplots(
                figsize=(20, 10), nrows=1, ncols=2, sharex=True, sharey=True)
            return plot_fr_complex(freqs, fr, fig, axs, **line_kwargs)
        else:
            raise ValueError(f"kind can only be 'Radial' or 'Complex', got {kind}")
    else: # we ignore the argument 'kind' if this is not the first plot on the fig
        axs = fig.axes
        if axs[0].get_yscale() == "log":
            return plot_fr_radial(freqs, fr, fig, axs, **line_kwargs)
        else:
            return plot_fr_complex(freqs, fr, fig, axs, **line_kwargs)
