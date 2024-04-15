#ifndef JAX_PLATE_LIB_TEST_FUNCTION_H
#define JAX_PLATE_LIB_TEST_FUNCTION_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <umfpack.h>

#define DEBUG

namespace py = pybind11;
using std::vector;
using std::cout;
using std::endl;

py::array_t<double> test_function(const py::array_t<double, py::array::c_style>& _x,
                                  const py::int_& pool_size) {
    py::buffer_info buf_x = _x.request();
    auto* x = static_cast<double*>(buf_x.ptr);
    auto len = buf_x.shape[0];
//     auto x = _x.unchecked<1>();
//     auto len = _x.shape(0);
    vector<double> y(len);

    cout << "UMFPACK timer: " << umfpack_timer() << endl;

    int n = pool_size.cast<int>();
    omp_set_num_threads(n);

    py::gil_scoped_release rel;

#pragma omp parallel for
    for (int i = 0; i < len; i++) {
#ifdef DEBUG
        if (i == 0)
            cout << omp_get_num_threads() << ' ' << omp_get_thread_num() << endl;
#endif
        y[i] = 2 * x[i] + sin(x[i]);
    }

    py::gil_scoped_acquire acq;

    auto ret = py::array_t(len, y.data());
    return ret;
}

#endif //JAX_PLATE_LIB_TEST_FUNCTION_H
