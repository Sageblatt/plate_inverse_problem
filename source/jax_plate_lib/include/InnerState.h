#ifndef JAX_PLATE_LIB_INNERSTATE_H
#define JAX_PLATE_LIB_INNERSTATE_H

#include <vector>
#include <cstdint>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "csc_matvec.h"
#include "umfpack_interface.h"

constexpr bool DEBUG = false; // Set to `true` to see debug output

using std::vector;
using std::array;
namespace py = pybind11;

using ShapeContainer = py::detail::any_container<ssize_t>;

// TODO: fight with code duplication

/*
 * Helper functions
 */

template<class T>
T* get_arr_ptr(const py::array_t<T, py::array::c_style> arr) {
    py::buffer_info buf = arr.request();
    return static_cast<T*>(buf.ptr);
}

template<class T>
vector<T> arr_to_vec(const py::array_t<T, py::array::c_style> arr){
    auto* ptr = get_arr_ptr(arr);
    auto len = arr.size();

    vector<T> res;
    res.insert(res.end(), &ptr[0], &ptr[len]);
    return res;
}

/*
 * Class for storing arrays of two particular types
 */

class IndexStorage {
private:
    vector<vector<int32_t>> small;
    vector<vector<int64_t>> large;
    vector<uint64_t> layout;

public:
    IndexStorage() = default;

    template<class T>
    void add(const vector<T> new_indices){
        if constexpr (std::is_same_v<T, int32_t>) {
            small.push_back(new_indices);
            layout.push_back(small.size() - 1);
        }
        else if constexpr (std::is_same_v<T, int64_t>) {
            large.push_back(new_indices);
            layout.push_back(large.size() - 1);
        }
    }

    template<class T>
    const vector<T>& get(const T num) const {
        const auto& coords = layout[num];
        if constexpr (std::is_same_v<T, int32_t>) {
            return small[coords];
        }
        else if constexpr (std::is_same_v<T, int64_t>) {
            return large[coords];
        }
    }
};

/*
 * Main class -- state of the sparse linear solver
 */

class InnerState {
private:
    IndexStorage indptrs;
    IndexStorage inds;

    IndexStorage indptrs_T;
    IndexStorage inds_T;

    IndexStorage permutations; // to get data_T from data 1D array

    std::vector<void*> symbolics;
    std::vector<int8_t> symb_types;

public:
    InnerState() {
        if constexpr (DEBUG)
            cout << "Constructor" << endl;
    }

    ~InnerState() {
        for (size_t i = 0; i < symbolics.size(); i++) {
            if (symb_types[i] == 0)
                umfpack::free_symbolic<double, int32_t>(&symbolics[i]);
            else if (symb_types[i] == 1)
                umfpack::free_symbolic<double, int64_t>(&symbolics[i]);
            else if (symb_types[i] == 2)
                umfpack::free_symbolic<complex128_t, int32_t>(&symbolics[i]);
            else if (symb_types[i] == 3)
                umfpack::free_symbolic<complex128_t, int64_t>(&symbolics[i]);
        }
        if constexpr (DEBUG)
            cout << "Destructor" << endl;
    }

    template<class T, class I>
    void add_mat(const py::int_& arg_N,
                 const py::array_t<I, py::array::c_style> arg_indices,
                 const py::array_t<I, py::array::c_style> arg_indptr,
                 const py::array_t<I, py::array::c_style> arg_indices_T,
                 const py::array_t<I, py::array::c_style> arg_indptr_T,
                 const py::array_t<I, py::array::c_style> arg_permutation,
                 const py::array_t<T> _arg_data) {
        if constexpr (DEBUG) {
            cout << "Called add_mat with types: " << typeid(T).name()
                 << ' ' << typeid(I).name() << endl;
        }
        auto indices_vec = arr_to_vec(arg_indices);
        auto indptr_vec = arr_to_vec(arg_indptr);
        auto indices_T_vec = arr_to_vec(arg_indices_T);
        auto indptr_T_vec = arr_to_vec(arg_indptr_T);
        auto perm_vec = arr_to_vec(arg_permutation);
        auto N = static_cast<I>(arg_N);

        inds.add<I>(indices_vec);
        indptrs.add<I>(indptr_vec);

        inds_T.add<I>(indices_T_vec);
        indptrs_T.add<I>(indptr_T_vec);

        permutations.add(perm_vec);

        auto symb_i = symbolics.size();
        symbolics.resize(symb_i + 1);

        if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int32_t>)
            symb_types.push_back(0);
        else if constexpr (std::is_same_v<T, double> and std::is_same_v<I, int64_t>)
            symb_types.push_back(1);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int32_t>)
            symb_types.push_back(2);
        else if constexpr (std::is_same_v<T, complex128_t> and std::is_same_v<I, int64_t>)
            symb_types.push_back(3);

        auto status = umfpack::symbolic(N, N, indptr_vec.data(), indices_vec.data(),
                                        (T*)NULL, &symbolics[symb_i], (double*)NULL, (double *)NULL);
        umfpack::check_status(status, "InnerState::add_mat in umfpack::symbolic.");
    }

    template<class T, class I>
    py::array_t<T, py::array::c_style> solve(const py::array_t<T, py::array::c_style> arg_data,
                                             const py::array_t<T, py::array::c_style> arg_b,
                                             const py::int_& solver_num,
                                             const py::bool_& arg_transpose,
                                             const py::int_& arg_n_cpu,
                                             const py::int_& arg_mode) {
        auto* data = get_arr_ptr(arg_data);
        auto* b = get_arr_ptr(arg_b);

        auto num = static_cast<I>(solver_num);

        const auto& Ai = inds.get(num);
        const auto& Ap = indptrs.get(num);

        const bool transpose = static_cast<bool>(arg_transpose);
        const auto n_cpu = static_cast<int>(arg_n_cpu);
        const auto mode = static_cast<int16_t>(arg_mode);

        int sys = UMFPACK_A;
        if (transpose)
            sys = UMFPACK_Aat;

        ShapeContainer res_shape;

        size_t b_shape_0 = arg_b.shape(0);
        size_t data_shape_0 = arg_data.shape(0);
        size_t mat_N, data_shape_1, b_shape_1, b_shape_2, data_N;
        switch (mode) {
            case 0:
                mat_N = b_shape_0;
                data_N = arg_data.shape(0);
                res_shape = ShapeContainer({mat_N});
                break;

            case 1:
                data_shape_1 = arg_data.shape(1);
                data_N = data_shape_1;
                mat_N = b_shape_0;
                res_shape = ShapeContainer({data_shape_0, mat_N});
                break;

            case 2:
                data_N = data_shape_0;
                b_shape_1 = arg_b.shape(1);
                mat_N = b_shape_1;
                res_shape = ShapeContainer({b_shape_0, mat_N});
                break;

            case 3:
                data_shape_1 = arg_data.shape(1);
                data_N = data_shape_1;
                b_shape_1 = arg_b.shape(1);
                mat_N = b_shape_1;
                res_shape = ShapeContainer({b_shape_0, mat_N});
                break;

            case 4:
                data_shape_1 = arg_data.shape(1);
                data_N = data_shape_1;
                b_shape_1 = arg_b.shape(1);
                b_shape_2 = arg_b.shape(2);
                mat_N = b_shape_2;
                res_shape = ShapeContainer({b_shape_0, b_shape_1, mat_N});
                break;

            default:
                throw std::runtime_error("Invalid `mode` argument in InnerState::solve().");
        }

        py::array_t<T, py::array::c_style> res(res_shape);
        auto res_ptr = get_arr_ptr(res);

        omp_set_num_threads(n_cpu);
        py::gil_scoped_release rel;

        if (mode == 0) {
            void* numeric;
            int s;
            s = umfpack::numeric(Ap.data(), Ai.data(), data, symbolics[num],
                                 &numeric, (double *)NULL, (double *)NULL);
            umfpack::check_status(s, "InnerState::solve in mode == 0 umfpack::numeric.");
            s = umfpack::solve(sys, Ap.data(), Ai.data(), data,
                           res_ptr, b, numeric, (double*)NULL, (double*)NULL);
            umfpack::check_status(s, "InnerState::solve in mode == 0 umfpack::solve.");
            umfpack::free_numeric<T, I>(&numeric);
        } else if (mode == 1) {
            #pragma omp parallel for
            for (size_t i = 0; i < data_shape_0; i++) {
                void* numeric;
                int s;
                s = umfpack::numeric(Ap.data(), Ai.data(), &data[data_N * i], symbolics[num],
                                     &numeric, (double *)NULL, (double *)NULL);
                umfpack::check_status(s, "InnerState::solve in mode == 1 umfpack::numeric.");
                s = umfpack::solve(sys, Ap.data(), Ai.data(), &data[data_N * i],
                                   &res_ptr[mat_N * i], b, numeric, (double*)NULL, (double*)NULL);
                umfpack::check_status(s, "InnerState::solve in mode == 1 umfpack::solve.");
                umfpack::free_numeric<T, I>(&numeric);
            }
        } else if (mode == 2) {
            #pragma omp parallel for
            for (size_t i = 0; i < b_shape_0; i++) {
                void* numeric;
                int s;
                s = umfpack::numeric(Ap.data(), Ai.data(), data, symbolics[num],
                                     &numeric, (double *)NULL, (double *)NULL);
                umfpack::check_status(s, "InnerState::solve in mode == 2 umfpack::numeric.");
                s = umfpack::solve(sys, Ap.data(), Ai.data(), data,
                                   &res_ptr[mat_N * i], &b[mat_N * i], numeric, (double*)NULL, (double*)NULL);
                umfpack::check_status(s, "InnerState::solve in mode == 2 umfpack::solve.");
                umfpack::free_numeric<T, I>(&numeric);
            }
        } else if (mode == 3) {
            #pragma omp parallel for
            for (size_t i = 0; i < data_shape_0; i++) {
                void* numeric;
                int s;
                s = umfpack::numeric(Ap.data(), Ai.data(), &data[data_N * i], symbolics[num],
                                     &numeric, (double *)NULL, (double *)NULL);
                umfpack::check_status(s, "InnerState::solve in mode == 3 umfpack::numeric.");
                s = umfpack::solve(sys, Ap.data(), Ai.data(), &data[data_N * i],
                                   &res_ptr[mat_N * i], &b[mat_N * i], numeric, (double*)NULL, (double*)NULL);
                umfpack::check_status(s, "InnerState::solve in mode == 3 umfpack::solve.");
                umfpack::free_numeric<T, I>(&numeric);
            }
        } else if (mode == 4) {
            for (size_t j = 0; j < b_shape_0; j++) {
                #pragma omp parallel for
                for (size_t i = 0; i < b_shape_1; i++) {
                    void* numeric;
                    int s;
                    s = umfpack::numeric(Ap.data(), Ai.data(), &data[data_N * i], symbolics[num],
                                         &numeric, (double *)NULL, (double *)NULL);
                    umfpack::check_status(s, "InnerState::solve in mode == 4 umfpack::numeric.");
                    auto idx = mat_N * (i + j * b_shape_1);
                    s = umfpack::solve(sys, Ap.data(), Ai.data(), &data[data_N * i],
                                       &res_ptr[idx], &b[idx], numeric, (double*)NULL, (double*)NULL);
                    umfpack::check_status(s, "InnerState::solve in mode == 4 umfpack::solve.");
                    umfpack::free_numeric<T, I>(&numeric);
                }
            }
        }
        py::gil_scoped_acquire acq;
        return res;
    }

    template<class T, class I>
    py::array_t<T, py::array::c_style> matvec(const py::array_t<T, py::array::c_style> arg_mat,
                                              const py::array_t<T, py::array::c_style> arg_vec,
                                              const py::int_& solver_num,
                                              const py::bool_& arg_transpose,
                                              const py::int_& arg_n_cpu,
                                              const py::int_& arg_mode) {
        auto* mat = get_arr_ptr(arg_mat);
        auto* vec = get_arr_ptr(arg_vec);

        const bool transpose = static_cast<bool>(arg_transpose);

        auto num = static_cast<I>(solver_num);

        const auto& Ai = transpose ? inds_T.get(num) : inds.get(num);
        const auto& Ap = transpose ? indptrs_T.get(num) : indptrs.get(num);
        const auto& perm = permutations.get(num);

        const auto n_cpu = static_cast<int>(arg_n_cpu);
        const auto mode = static_cast<int16_t>(arg_mode);

        ShapeContainer res_shape;

        size_t b_shape_0 = arg_vec.shape(0);
        size_t data_shape_0 = arg_mat.shape(0);
        size_t mat_N, data_shape_1, b_shape_1, b_shape_2, data_N;
        switch (mode) {
            case 0:
                mat_N = b_shape_0;
                data_N = arg_mat.shape(0);
                res_shape = ShapeContainer({mat_N});
                break;

            case 1:
                data_shape_1 = arg_mat.shape(1);
                data_N = data_shape_1;
                mat_N = b_shape_0;
                res_shape = ShapeContainer({data_shape_0, mat_N});
                break;

            case 2:
                data_N = data_shape_0;
                b_shape_1 = arg_vec.shape(1);
                mat_N = b_shape_1;
                res_shape = ShapeContainer({b_shape_0, mat_N});
                break;

            case 3:
                data_shape_1 = arg_mat.shape(1);
                data_N = data_shape_1;
                b_shape_1 = arg_vec.shape(1);
                mat_N = b_shape_1;
                res_shape = ShapeContainer({b_shape_0, mat_N});
                break;

            case 4:
                data_shape_1 = arg_mat.shape(1);
                data_N = data_shape_1;
                b_shape_1 = arg_vec.shape(1);
                b_shape_2 = arg_vec.shape(2);
                mat_N = b_shape_2;
                res_shape = ShapeContainer({b_shape_0, b_shape_1, mat_N});
                break;

            default:
                throw std::runtime_error("Invalid `mode` argument in InnerState::matvec().");
        }

        I n_col = static_cast<I>(mat_N);

        py::array_t<T, py::array::c_style> res(res_shape);
        auto res_size = res.size();
        auto res_ptr = get_arr_ptr(res);
        auto res_double_ptr = reinterpret_cast<double *>(res_ptr);

        size_t size_mult = 1;
        if constexpr (std::is_same_v<T, complex128_t>)
            size_mult = 2;

        for (size_t i = 0; i < size_mult * res_size; i++)
            res_double_ptr[i] = 0.0;

        omp_set_num_threads(n_cpu);
        py::gil_scoped_release rel;

        if (transpose) {
            if (mode == 0) {
                csc_matvec_transpose(n_col, Ap.data(), Ai.data(),
                                     mat, vec, perm.data(), res_ptr);
            } else if (mode == 1) {
                #pragma omp parallel for
                for (size_t i = 0; i < data_shape_0; i++) {
                    csc_matvec_transpose(n_col, Ap.data(), Ai.data(),
                                         &mat[data_N * i], vec,
                                         perm.data(), &res_ptr[mat_N * i]);
                }
            } else if (mode == 2) {
                #pragma omp parallel for
                for (size_t i = 0; i < b_shape_0; i++) {
                    csc_matvec_transpose(n_col, Ap.data(), Ai.data(),
                                         mat, &vec[mat_N * i],
                                         perm.data(), &res_ptr[mat_N * i]);
                }
            } else if (mode == 3) {
                #pragma omp parallel for
                for (size_t i = 0; i < data_shape_0; i++) {
                    csc_matvec_transpose(n_col, Ap.data(), Ai.data(),
                                         &mat[data_N * i], &vec[mat_N * i],
                                         perm.data(), &res_ptr[mat_N * i]);
                }
            } else if (mode == 4) {
                for (size_t j = 0; j < b_shape_0; j++) {
                    #pragma omp parallel for
                    for (size_t i = 0; i < b_shape_1; i++) {
                        auto idx = mat_N * (i + j * b_shape_1);
                        csc_matvec_transpose(n_col, Ap.data(), Ai.data(),
                                             &mat[data_N * i], &vec[idx],
                                             perm.data(), &res_ptr[idx]);
                    }
                }
            }
        } else { // transpose == false
            if (mode == 0) {
                csc_matvec(n_col, Ap.data(), Ai.data(),
                           mat, vec, res_ptr);
            } else if (mode == 1) {
                #pragma omp parallel for
                for (size_t i = 0; i < data_shape_0; i++) {
                    csc_matvec(n_col, Ap.data(), Ai.data(),
                               &mat[data_N * i], vec,
                               &res_ptr[mat_N * i]);
                }
            } else if (mode == 2) {
                #pragma omp parallel for
                for (size_t i = 0; i < b_shape_0; i++) {
                    csc_matvec(n_col, Ap.data(), Ai.data(),
                               mat, &vec[mat_N * i],
                               &res_ptr[mat_N * i]);
                }
            } else if (mode == 3) {
                #pragma omp parallel for
                for (size_t i = 0; i < data_shape_0; i++) {
                    csc_matvec(n_col, Ap.data(), Ai.data(),
                               &mat[data_N * i], &vec[mat_N * i],
                               &res_ptr[mat_N * i]);
                }
            } else if (mode == 4) {
                for (size_t j = 0; j < b_shape_0; j++) {
                    #pragma omp parallel for
                    for (size_t i = 0; i < b_shape_1; i++) {
                        auto idx = mat_N * (i + j * b_shape_1);
                        csc_matvec(n_col, Ap.data(), Ai.data(),
                                   &mat[data_N * i], &vec[idx],
                                   &res_ptr[idx]);
                    }
                }
            }
        }
        py::gil_scoped_acquire acq;
        return res;
    }
};

#endif //JAX_PLATE_LIB_INNERSTATE_H
