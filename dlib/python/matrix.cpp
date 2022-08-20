// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/string.h>
#include <pybind11/pybind11.h>

using namespace dlib;
namespace py = pybind11;
using std::string;
using std::ostringstream;


void matrix_set_size(matrix<double>& m, long nr, long nc)
{
    m.set_size(nr,nc);
    m = 0;
}

string matrix_double__repr__(matrix<double>& c)
{
    ostringstream sout;
    sout << "< dlib.matrix containing: \n";
    sout << c;
    return trim(sout.str()) + " >";
}

string matrix_double__str__(matrix<double>& c)
{
    ostringstream sout;
    sout << c;
    return trim(sout.str());
}

std::shared_ptr<matrix<double> > make_matrix_from_size(long nr, long nc)
{
    if (nr < 0 || nc < 0)
    {
        PyErr_SetString( PyExc_IndexError, "Input dimensions can't be negative."
        );
        throw py::error_already_set();
    }
    auto temp = std::make_shared<matrix<double>>(nr,nc);
    *temp = 0;
    return temp;
}


std::shared_ptr<matrix<double> > from_object(py::object obj)
{
    py::tuple s = obj.attr("shape").cast<py::tuple>();
    if (len(s) != 2)
    {
        PyErr_SetString( PyExc_IndexError, "Input must be a matrix or some kind of 2D array."
        );
        throw py::error_already_set();
    }

    const long nr = s[0].cast<long>();
    const long nc = s[1].cast<long>();
    auto temp = std::make_shared<matrix<double>>(nr,nc);
    for ( long r = 0; r < nr; ++r)
    {
        for (long c = 0; c < nc; ++c)
        {
            (*temp)(r,c) = obj[py::make_tuple(r,c)].cast<double>();
        }
    }
    return temp;
}

std::shared_ptr<matrix<double> > from_list(py::list l)
{
    const long nr = py::len(l);
    if (py::isinstance<py::list>(l[0]))
    {
        const long nc = py::len(l[0]);
        // make sure all the other rows have the same length
        for (long r = 1; r < nr; ++r)
            pyassert(py::len(l[r]) == nc, "All rows of a matrix must have the same number of columns.");

        auto temp = std::make_shared<matrix<double>>(nr,nc);
        for ( long r = 0; r < nr; ++r)
        {
            for (long c = 0; c < nc; ++c)
            {
                (*temp)(r,c) = l[r].cast<py::list>()[c].cast<double>();
            }
        }
        return temp;
    }
    else
    {
        // In this case we treat it like a column vector
        auto temp = std::make_shared<matrix<double>>(nr,1);
        for ( long r = 0; r < nr; ++r)
        {
            (*temp)(r) = l[r].cast<double>();
        }
        return temp;
    }
}

long matrix_double__len__(matrix<double>& c)
{
    return c.nr();
}

void matrix_double_serialize(const matrix<double>& m, const std::string& file)
{
    serialize(file) << m;
}

void matrix_double_deserialize(matrix<double>& m, const std::string& file)
{
    deserialize(file) >> m;
}

struct mat_row
{
    mat_row() : data(0),size(0) {}
    mat_row(double* data_, long size_) : data(data_),size(size_) {}
    double* data;
    long size;
};

void mat_row__setitem__(mat_row& c, long p, double val)
{
    if (p < 0) {
        p = c.size + p; // negative index
    }
    if (p > c.size-1) {
        PyErr_SetString( PyExc_IndexError, "3 index out of range"
        );
        throw py::error_already_set();
    }
    c.data[p] = val;
}


string mat_row__str__(mat_row& c)
{
    ostringstream sout;
    sout << mat(c.data,1, c.size);
    return sout.str();
}

string mat_row__repr__(mat_row& c)
{
    ostringstream sout;
    sout << "< matrix row: " << mat(c.data,1, c.size);
    return trim(sout.str()) + " >";
}

long mat_row__len__(mat_row& m)
{
    return m.size;
}

double mat_row__getitem__(mat_row& m, long r)
{
    if (r < 0) {
        r = m.size + r; // negative index
    }
    if (r > m.size-1 || r < 0) {
        PyErr_SetString( PyExc_IndexError, "1 index out of range"
        );
        throw py::error_already_set();
    }
    return m.data[r];
}

mat_row matrix_double__getitem__(matrix<double>& m, long r)
{
    if (r < 0) {
        r = m.nr() + r; // negative index
    }
    if (r > m.nr()-1 || r < 0) {
        PyErr_SetString( PyExc_IndexError, (string("2 index out of range, got ") + cast_to_string(r)).c_str()
        );
        throw py::error_already_set();
    }
    return mat_row(&m(r,0),m.nc());
}


py::tuple get_matrix_size(matrix<double>& m)
{
    return py::make_tuple(m.nr(), m.nc());
}

void bind_matrix(py::module& m)
{
    py::class_<mat_row>(m, "_row")
        .def("__len__", &mat_row__len__)
        .def("__repr__", &mat_row__repr__)
        .def("__str__", &mat_row__str__)
        .def("__setitem__", &mat_row__setitem__)
        .def("__getitem__", &mat_row__getitem__);

    py::class_<matrix<double>, std::shared_ptr<matrix<double>>>(m, "matrix",
        "This object represents a dense 2D matrix of floating point numbers."
        "Moreover, it binds directly to the C++ type dlib::matrix<double>.")
        .def(py::init<>())
        .def(py::init(&from_list))
        .def(py::init(&from_object))
        .def(py::init(&make_matrix_from_size))
        .def("set_size", &matrix_set_size, py::arg("rows"), py::arg("cols"), "Set the size of the matrix to the given number of rows and columns.")
        .def("__repr__", &matrix_double__repr__)
        .def("__str__", &matrix_double__str__)
        .def("nr", &matrix<double>::nr, "Return the number of rows in the matrix.")
        .def("nc", &matrix<double>::nc, "Return the number of columns in the matrix.")
        .def("serialize", &matrix_double_serialize, py::arg("file"), "Serialize the matrix to a file")
        .def("deserialize", &matrix_double_deserialize, py::arg("file"), "Deserialize the matrix from a file")
        .def("__len__", &matrix_double__len__)
        .def("__getitem__", &matrix_double__getitem__, py::keep_alive<0,1>())
        .def_property_readonly("shape", &get_matrix_size)
        .def(py::pickle(&getstate<matrix<double>>, &setstate<matrix<double>>));
}
