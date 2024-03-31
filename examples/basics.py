"""Simple script to check if everything works"""
import jax_plate as jp
import numpy as np


acc = jp.Accelerometer.Accelerometer('AP1030')
geom = jp.Geometry.Geometry('symm', acc, jp.Geometry.GeometryParams(100e-3, 20e-3, 2e-3, 10e-3, None))
mat = jp.Material.get_material(7920.0, 'isotropic', E = 200*1e9, G = 75*1e9 , beta = .003)

p = jp.Problem.Problem(geom, mat, acc)

N = 50

freq = np.linspace(40, 600, N)
fr = p.solveForward(freq)

p0 = [0.1, 0.1, 0.2]

res = p.solveInverseLocal(p0, 'MSE_LOG_AFC', 'grad_descent', ref_fr=[freq, fr], use_rel=True,
                          compression=(False, N), case_name='Example_',
                          extra_info='Running `basics.py` example.\n',
                          N_steps=2, h=0.001, f_min=1e-5)

hist = res.f_history
res = res.x

r1 = p.solveForward(freq, (np.array(p0) + 1)*p.parameters)
r2 = p.solveForward(freq, res)

print(f'FR: {np.sum(np.abs(fr)):.4f}, expected: 341.9363')
print(f'Initial: {np.sum(np.abs(r1)):.4f}, expected: expected: 91.7139')
print(f'After: {np.sum(np.abs(r2)):.4f}, expected: 90.8778')
print(f'F_hist: {np.sum(np.abs(hist)):.4f}, expected: 0.4389')
