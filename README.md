# plate_inverse_problem

As for now, a differentiable FEM model for the dynamics of thin plate structures in the frequency domain has been implemented. 
It then has been used to estimate the material's elastic moduli and loss factors, based on the data of the vibration testing (coefficient inverse problem).


FreeFEM++ is used to assemble the matrices of the direct problem. 
The code for solution of the direct problem and computation of the derivatives is then jit-compiled with `jax` library for execution either on CPU or GPU.
Implementation of several first and second order optimization methods is also provided.

## Installation

1. Install [FreeFEM++](https://freefem.org/)
2. Install [Jax](https://github.com/google/jax)
3. `git clone --recurse-submodules https://github.com/viviaxenov/plate_inverse_problem.git`
