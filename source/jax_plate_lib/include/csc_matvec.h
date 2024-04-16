#ifndef JAX_PLATE_LIB_CSC_MATVEC_H
#define JAX_PLATE_LIB_CSC_MATVEC_H

/*
 * Modified version of https://github.com/scipy/scipy/blob/v1.13.0/scipy/sparse/sparsetools/csc.h
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
 * All rights reserved.
 * Check csc_matvec_License.txt.
 */

/*
 * Compute Y += A*X for CSC matrix A and dense vectors X,Y
 *
 *
 * Input Arguments:
 *   I  n_col         - number of columns in A
 *   I  Ap[n_row+1]   - column pointer
 *   I  Ai[nnz(A)]    - row indices
 *   T  Ax[n_col]     - nonzeros
 *   T  Xx[n_col]     - input vector
 *
 * Output Arguments:
 *   T  Yx[n_row]     - output vector
 *
 * Note:
 *   Output array Yx must be preallocated
 *
 *   Complexity: Linear.  Specifically O(nnz(A) + n_col)
 *
 */
template <class I, class T>
void csc_matvec(const I n_col,
                const I Ap[],
                const I Ai[],
                const T Ax[],
                const T Xx[],
                T Yx[]) {
    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            I i    = Ai[ii];
            Yx[i] += Ax[ii] * Xx[j];
        }
    }
}

template <class I, class T>
void csc_matvec_transpose(const I n_col,
                          const I Ap[], // Ap and Ai are for transposed matrix
                          const I Ai[],
                          const T Ax[],
                          const T Xx[],
                          const I perm[], // permutation array, Ax is arranged for matrix before transpose
                          T Yx[]) {
    for(I j = 0; j < n_col; j++){
        I col_start = Ap[j];
        I col_end   = Ap[j+1];

        for(I ii = col_start; ii < col_end; ii++){
            I i    = Ai[ii];
            Yx[i] += Ax[perm[ii]] * Xx[j];
        }
    }
}

#endif //JAX_PLATE_LIB_CSC_MATVEC_H
