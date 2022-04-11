# plate_inverse_problem

This is my research/master thesis project. 
As for now, a differentiable FEM model for the dynamics of thin plate structures in the frequency domain has been implemented. 
It then has been used to estimate the material's elastic moduli and loss factors, based on the data of the vibration testing (coefficient inverse problem).


FreeFEM++ is used to assemble the matrices of the direct problem. 
The code for solution of the direct problem and computation of the derivatives is then jit-compiled with `jax` library for execution either on CPU or GPU.
Implementation of several first and second order optimization methods is also provided.

***In future***, I plan to expand the code to:
- Solve the inverse problem with frequency-dependent parameters
- Solve the inverse problem with time-domain data
- Solve the optimal design problem

## Installation

1. Install [FreeFEM++](https://freefem.org/)
2. Install [Jax](https://github.com/google/jax)
3. `git clone https://github.com/viviaxenov/pyFreeFem.git`
4. `git clone https://github.com/viviaxenov/plate_inverse_problem.git`
5. Add `plate_inverse_problem/source/jax_plate/` and `pyFreeFem/pyFreeFem/` to your Python environment path
6. Check out `plate_inverse_problem/examples/Demo.ipynb` for examples of code usage
