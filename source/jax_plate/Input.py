import numpy as np
from scipy.signal import find_peaks, savgol_filter, peak_widths, peak_prominences


class Compressor:
    def __init__(self, freqs: np.ndarray, complex_fr: np.ndarray,
                 max_size: int, use_alg: int):
        """
        Callable object which is used to compress FR data for further
        optimization.

        Parameters
        ----------
        freqs : np.ndarray
            Array of frequencies in FR.
        complex_fr : np.ndarray
            Complex amplitudes of given FR.
        max_size : int
            Maximal array size which can be used in optimization.
        use_alg : int
            Type of compression algorithm:
                0 - uniformly distributed,
                1 - only the highest points from all peaks,
                2 - Not implemented yet.

        Returns
        -------
        None

        """
        assert freqs.size == complex_fr.size
        self.size = freqs.size
        self.freqs = freqs
        self.complex_fr = complex_fr
        self.max_size = max_size
        self.alg = use_alg

    @staticmethod
    def _peak_smoothness(x: np.ndarray, peaks: np.ndarray) -> np.ndarray:
        """
        Returns smoothness of a each peak in signal x judging by 20 nearest
        neighbours.

        Parameters
        ----------
        x : np.ndarray
            Signal to be processed (must include more than 20 points).
        peaks : np.ndarray
            Array of peak indices.

        Returns
        -------
        np.ndarray
            Array of smoothness values for each peak.

        """
        res = np.zeros_like(peaks, dtype=np.float64)
        for i, p in enumerate(peaks):
            bds = 10
            if p <= 10 or x.size - p <= 10:
                bds = min(p, x.size-p) - 1
            interval = x[p-bds:p+bds+1]
            res[i] = np.sum(np.abs(np.diff(interval)))/2/bds*20
        return 1/res

    def __call__(self, desired_size: int) -> (np.ndarray, np.ndarray):
        if desired_size > self.max_size:
            raise ValueError(f'Desired size of compressed data must be lower than {self.max_size+1}')

        bool_mask = np.zeros(self.size, dtype=bool)

        if self.alg == 0:
            step = self.size / desired_size

            current = 0.0
            while current < self.size:
                bool_mask[int(current)] = True
                current += step

            if np.sum(bool_mask) > desired_size:
                bool_mask[0] = False


        elif self.alg == 1:
            # TODO: replace magic numbers with dynamic parameter evaluation
            freq_step = np.max(np.diff(self.freqs))
            dst = int(75/freq_step) # peaks width is around 75 Hz
            # dst = 1 # peaks width is around 75 Hz

            idx = []
            tmp_afc = np.log(savgol_filter(np.abs(self.complex_fr), 30, 3)) # or maybe scipy.signal.medfilt(y, 31)
            # TODO: check if smoothing needs to be done after np.log
            # tmp_afc = savgol_filter(np.log(np.abs(self.complex_fr)), 30, 3)

            for afc in (tmp_afc, -tmp_afc):
                all_peaks = find_peaks(afc, distance=dst)

                ws = peak_widths(afc, all_peaks[0])
                width_filtered = all_peaks[0][ws[0] > 20]

                pr = peak_prominences(afc, width_filtered)
                prom_filtered = width_filtered[pr[0] > 0.1]

                ps = self._peak_smoothness(afc, prom_filtered)
                idx.append(prom_filtered[ps < 50])

                # per = spsig.peak_prominences(afc, prom_filtered[ps < 50])[0]
                # wih = spsig.peak_widths(afc, prom_filtered[ps < 50])[0]
                # print('---------')
                # print('Freq: ', self.freqs[prom_filtered[ps < 50]])
                # print('PR: ', per)
                # print('WH: ', wih)
                # print('Ratio: ', wih/per)
                # print('Product: ', per*wih)
                # print('SM: ', self._peak_smoothness(afc, prom_filtered[ps < 50]))

            idx = np.concatenate(idx)

            npeaks = idx.size

            pts = desired_size - npeaks

            layers = pts // (npeaks * 2)

            left_idx = idx - layers
            right_idx = idx + layers

            left_idx[left_idx < 0] = 0
            right_idx[right_idx + 1 > self.size] = self.size

            for i in range(npeaks):
                bool_mask[left_idx[i]:right_idx[i]+1] = True

            diff = desired_size - np.sum(bool_mask)

            while diff != 0: # breaks when they collide
                for i in range(npeaks-1):
                    if right_idx[i] < left_idx[i+1]:
                        right_idx[i] += 1
                        diff -= 1
                        bool_mask[right_idx[i]+1] = True
                    if diff == 0:
                        break

                if diff == 0:
                    break

                if right_idx[-1] + 1 < self.size:
                    right_idx[-1] += 1
                    diff -= 1
                    bool_mask[right_idx[-1]] = True

                elif left_idx[0] - 1 > 0:
                    left_idx[0] -= 1
                    diff -= 1
                    bool_mask[left_idx[0]] = True

        return self.freqs[bool_mask], self.complex_fr[bool_mask]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    N = 100
    x = np.linspace(0, 10, N)
    # y = -np.abs(np.sin(np.exp(-0.03*x**2)*7.) - 0.01) - x**3*0.0012 + 1.5
    y = -np.abs(np.sin(np.exp(-0.01*x**2)*7.) - 0.01) - x**3*0.0012 + 1.5
    y *= 100
    x = np.linspace(40, 1200, N)
    
    plt.plot(x,y, figure=plt.figure(figsize=(9.0, 6.0), dpi=300))
    
    cm = Compressor(x, y, N, 1)
    n = 30
    x_, y_ = cm(n)
    
    
    cm = Compressor(x, y, N, 0)
    x__, y__ = cm(n)
    plt.plot(x__, y__, 'ks')
    plt.plot(x_, y_, 'ro')
    
    ax = plt.gca()
    ax.grid('on')
    # plt.savefig('comp_example.png', bbox_inches='tight')
    