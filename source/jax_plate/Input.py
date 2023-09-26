import numpy as np
from scipy.signal import find_peaks


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
            Type of compression algorighm: 
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
            # TODO: more univesal peaks' width detection
            freq_step = self.freqs[1] - self.freqs[0]
            dst = int(75/freq_step) # peaks width is around 75 Hz
            idx = find_peaks(np.abs(self.complex_fr), height=3, distance=dst)[0]
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
    