# plate_inverse_problem

As for now, a differentiable FEM model for the dynamics of thin plate structures in the frequency domain has been implemented. 
It has been used to estimate the material's elastic moduli and loss factors, based on the data of the vibration testing (coefficient inverse problem).


FreeFEM++ is used to assemble the matrices of the direct problem. 
The code for solution of the direct problem and computation of the derivatives is then jit-compiled with `jax` library for execution on CPU.
Implementation of several first and second order optimization methods is also provided.

## Installation

1. Install [FreeFEM++](https://freefem.org/)
2. Install [Jax](https://github.com/google/jax)
3. `git clone https://github.com/Sageblatt/plate_inverse_problem.git`
4. `cd ./plate_inverse_problem/source`
5. `pip install -e .`
6. `cd jax_plate_lib`
7. `./run_cmake.sh` (before that you might need to change SuiteSparse directories in `CMakeLists.txt`)
8. `./create.sh`

## Requirements

1. Python 3.10+
2. NumPy
3. SciPy 1.12.0+
4. SuiteSparse v5.x (particulary UMFPACK)
5. Jax 0.4.20+
6. Matplotlib
7. SymPy
8. pybind11
9. C++ compiler with OpenMP support
10. CMake 3.10+
11. Make

## References

1. Aksenov, V., Vasyukov, A. and Beklemysheva, K. (2024) Acquiring elastic properties of thin composite structure from vibrational testing data. Journal of Inverse and Ill-posed Problems, Vol. 32 (Issue 3), pp. 467-484. https://doi.org/10.1515/jiip-2022-0081
2. Lavrenkov, S., Vasyukov, A. (2024). Experimental Data Compression for GPU-Based Solution of Inverse Coefficient Problem for Vibrational Testing Data. Communications in Computer and Information Science, Vol 1914. Springer, Cham. https://doi.org/10.1007/978-3-031-52470-7_24
3. Lavrenkov, S., Smirnov, I., Kravchenko, D., Beklemysheva, K. and Vasyukov, A. (2025) “Numerical Modeling of Complex Geometry Thin Composite Structures under Vibrational Testing”, Supercomputing Frontiers and Innovations, 11(4), pp. 15–28. https://doi.org/10.14529/jsfi240402.2
4. Original project repo: https://github.com/viviaxenov/plate_inverse_problem.
