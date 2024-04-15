#include <pybind11/pybind11.h>

#include "test_function.h"
#include "InnerState.h"


PYBIND11_MODULE(jax_plate_lib, m) {
    m.def("test_function", &test_function, "Testing purposes.");
    py::class_<InnerState>(m, "InnerState")
            .def(py::init<>())
            .def("add_mat", &InnerState::add_mat<double, int32_t>)
            .def("add_mat", &InnerState::add_mat<complex128_t, int32_t>)
            .def("add_mat", &InnerState::add_mat<double, int64_t>)
            .def("add_mat", &InnerState::add_mat<complex128_t, int64_t>)
            .def("solve", &InnerState::solve<double, int32_t>)
            .def("solve", &InnerState::solve<complex128_t, int32_t>)
            .def("solve", &InnerState::solve<double, int64_t>)
            .def("solve", &InnerState::solve<complex128_t, int64_t>);
}
