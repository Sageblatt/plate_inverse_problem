"""Script to check if jax_plate.jax_plate_lib module is working as intended."""
import numpy as np
from jax_plate.jax_plate_lib import test_function
from time import perf_counter as pf

a = np.linspace(0, 100, 50000000)

t1 = pf()
r0 = 2 * a + np.sin(a)
t2 = pf()
print('Numpy time: ', t2 - t1)

for N_threads in range(1, 5):
    print(f'Threads argument: {N_threads}')
    t1 = pf()
    r1 = test_function(a, N_threads)
    t2 = pf()
    print('Overall call time: ', t2 - t1)
    print(f'Delta: {np.max(np.abs(r1 - r0))}\n')
