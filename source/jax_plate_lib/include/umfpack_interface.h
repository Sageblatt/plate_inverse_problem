#ifndef JAX_PLATE_LIB_UMFPACK_ADAPTER_H
#define JAX_PLATE_LIB_UMFPACK_ADAPTER_H

#include <complex>
#include <umfpack.h>

using complex128_t = std::complex<double>;

namespace umfpack {
    int symbolic(int32_t n_row,
                 int32_t n_col,
                 const int32_t Ap[],
                 const int32_t Ai[],
                 const double Ax[],
                 void **Symbolic,
                 const double Control[UMFPACK_CONTROL],
                 double Info[UMFPACK_INFO]) {
        return umfpack_di_symbolic(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info);
    }

    int symbolic(int64_t n_row,
                 int64_t n_col,
                 const int64_t Ap[],
                 const int64_t Ai[],
                 const double Ax[],
                 void **Symbolic,
                 const double Control[UMFPACK_CONTROL],
                 double Info[UMFPACK_INFO]) {
        return umfpack_dl_symbolic(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info);
    }

    int symbolic(int32_t n_row,
                 int32_t n_col,
                 const int32_t Ap[],
                 const int32_t Ai[],
                 const complex128_t Ax[],
                 void **Symbolic,
                 const double Control[UMFPACK_CONTROL],
                 double Info[UMFPACK_INFO]) {
        auto Ax_p = reinterpret_cast<const double *>(Ax);
        return umfpack_zi_symbolic(n_row, n_col, Ap, Ai, Ax_p, NULL, Symbolic, Control, Info);
    }

    int symbolic(int64_t n_row,
                 int64_t n_col,
                 const int64_t Ap[],
                 const int64_t Ai[],
                 const complex128_t Ax[],
                 void **Symbolic,
                 const double Control[UMFPACK_CONTROL],
                 double Info[UMFPACK_INFO]) {
        auto Ax_p = reinterpret_cast<const double *>(Ax);
        return umfpack_zl_symbolic(n_row, n_col, Ap, Ai, Ax_p, NULL, Symbolic, Control, Info);
    }

    int numeric(const int32_t Ap[],
                const int32_t Ai[],
                const double Ax[],
                void *Symbolic,
                void **Numeric,
                const double Control[UMFPACK_CONTROL],
                double Info[UMFPACK_INFO]) {
        return umfpack_di_numeric(Ap, Ai, Ax, Symbolic, Numeric, Control, Info);
    }

    int numeric(const int64_t Ap[],
                const int64_t Ai[],
                const double Ax[],
                void *Symbolic,
                void **Numeric,
                const double Control[UMFPACK_CONTROL],
                double Info[UMFPACK_INFO]) {
        return umfpack_dl_numeric(Ap, Ai, Ax, Symbolic, Numeric, Control, Info);
    }

    int numeric(const int32_t Ap[],
                const int32_t Ai[],
                const complex128_t Ax[],
                void *Symbolic,
                void **Numeric,
                const double Control[UMFPACK_CONTROL],
                double Info[UMFPACK_INFO]) {
        auto Ax_p = reinterpret_cast<const double *>(Ax);
        return umfpack_zi_numeric(Ap, Ai, Ax_p, NULL, Symbolic, Numeric, Control, Info);
    }

    int numeric(const int64_t Ap[],
                const int64_t Ai[],
                const complex128_t Ax[],
                void *Symbolic,
                void **Numeric,
                const double Control[UMFPACK_CONTROL],
                double Info[UMFPACK_INFO]) {
        auto Ax_p = reinterpret_cast<const double *>(Ax);
        return umfpack_zl_numeric(Ap, Ai, Ax_p, NULL, Symbolic, Numeric, Control, Info);
    }

    int solve(int sys,
              const int32_t Ap[],
              const int32_t Ai[],
              const double Ax[],
              double X[],
              const double B[],
              void *Numeric,
              const double Control[UMFPACK_CONTROL],
              double Info[UMFPACK_INFO]) {
        return umfpack_di_solve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info);
    }

    int solve(int sys,
              const int64_t Ap[],
              const int64_t Ai[],
              const double Ax[],
              double X[],
              const double B[],
              void *Numeric,
              const double Control[UMFPACK_CONTROL],
              double Info[UMFPACK_INFO]) {
        return umfpack_dl_solve(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info);
    }

    int solve(int sys,
              const int32_t Ap[],
              const int32_t Ai[],
              const complex128_t Ax[],
              complex128_t Xx[],
              const complex128_t Bx[],
              void *Numeric,
              const double Control[UMFPACK_CONTROL],
              double Info[UMFPACK_INFO]) {
        auto Ax_p = reinterpret_cast<const double *>(Ax);
        auto Xx_p = reinterpret_cast<double *>(Xx);
        auto Bx_p = reinterpret_cast<const double *>(Bx);
        return umfpack_zi_solve(sys, Ap, Ai, Ax_p, NULL, Xx_p, NULL, Bx_p, NULL, Numeric, Control, Info);
    }

    int solve(int sys,
              const int64_t Ap[],
              const int64_t Ai[],
              const complex128_t Ax[],
              complex128_t Xx[],
              const complex128_t Bx[],
              void *Numeric,
              const double Control[UMFPACK_CONTROL],
              double Info[UMFPACK_INFO]) {
        auto Ax_p = reinterpret_cast<const double *>(Ax);
        auto Xx_p = reinterpret_cast<double *>(Xx);
        auto Bx_p = reinterpret_cast<const double *>(Bx);
        return umfpack_zl_solve(sys, Ap, Ai, Ax_p, NULL, Xx_p, NULL, Bx_p, NULL, Numeric, Control, Info);
    }

    template <class T, class I>
    void free_numeric(void** Numeric) {
        cout << typeid(T).name() << ' ' << typeid(I).name() << endl;
        if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int32_t>)
            umfpack_di_free_numeric(Numeric);
        else if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int64_t>)
            umfpack_dl_free_numeric(Numeric);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int32_t>)
            umfpack_zi_free_numeric(Numeric);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int64_t>)
            umfpack_zl_free_numeric(Numeric);
    }

    template <class T, class I>
    void free_symbolic(void** Symbolic) {
        if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int32_t>)
            umfpack_di_free_symbolic(Symbolic);
        else if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int64_t>)
            umfpack_dl_free_symbolic(Symbolic);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int32_t>)
            umfpack_zi_free_symbolic(Symbolic);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int64_t>)
            umfpack_zl_free_symbolic(Symbolic);
    }

    template <class T, class I>
    void report_symbolic(void* Symbolic, const double Control [UMFPACK_CONTROL]) {
        if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int32_t>)
            umfpack_di_report_symbolic(Symbolic, Control);
        else if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int64_t>)
            umfpack_dl_report_symbolic(Symbolic, Control);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int32_t>)
            umfpack_zi_report_symbolic(Symbolic, Control);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int64_t>)
            umfpack_zl_report_symbolic(Symbolic, Control);
    }
}

#endif //JAX_PLATE_LIB_UMFPACK_ADAPTER_H
