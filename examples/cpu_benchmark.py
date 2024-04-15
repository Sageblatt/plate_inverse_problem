"""A script for evaluating performance in solving forward and inverse problems."""
import jax_plate as jp
import numpy as np
from time import perf_counter as pf


acc = jp.Accelerometer.Accelerometer('AP1030')
geom = jp.Geometry.Geometry('sh_i', acc, jp.Geometry.GeometryParams(100e-3, 20e-3, 2e-3, None, None))
mat = jp.Material.get_material(7920.0, 'isotropic', E = 200*1e9, G = 75*1e9 , beta = .003)

p1 = jp.Problem.Problem(geom, mat, acc)

N_comp = 200

N_freq = 3000

freq = np.linspace(40, 600, N_freq)

t1 = pf()
fr = p1.solveForward(freq)
t2 = pf()

res1 = np.sum(np.abs(fr))
expected1 = 8416.439906133683

print(f'Forward problem time: {t2-t1:.3f} seconds')
print(f'With relative error: {(res1 - expected1)/expected1:.3f}')

t1 = pf()
inv_res = p1.solveInverse([0.1, 0.1, 0.2], 'MSE_LOG_AFC', 'gd', ref_fr=[freq, fr], use_rel=True,
                          compression=(True, N_comp), log=False, report=False,
                          N_steps=20, h=0.001, f_min=1e-10)
t2 = pf()

expected2 = np.array([2.2000000e+11, 8.2500000e+10, 5.4951294e-02])

print(f'Inverse problem time: {t2-t1:.3f} seconds')
print(f'With relative error: {np.sum((inv_res.x - expected2)/expected2):.3f}')
