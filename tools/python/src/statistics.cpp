#include <pybind11/operators.h>
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>
#include <dlib/statistics.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

void bind_statistics(py::module& m)
{
    py::class_<running_stats<double>>(m, "running_stats")
        .def(py::init<>())
        .def("clear",       &running_stats<double>::clear       )
        .def("add",         &running_stats<double>::add         )
        .def("current_n",   &running_stats<double>::current_n   )
        .def("mean",        &running_stats<double>::mean        )
        .def("max",         &running_stats<double>::max         )
        .def("min",         &running_stats<double>::min         )
        .def("variance",    &running_stats<double>::variance    )
        .def("stddev",      &running_stats<double>::stddev      )
        .def("skewness",    &running_stats<double>::skewness    )
        .def("ex_kurtosis", &running_stats<double>::ex_kurtosis )
        .def("scale",       &running_stats<double>::scale       )
        .def("stddev",      &running_stats<double>::stddev      )
        .def(py::self + py::self);
}
