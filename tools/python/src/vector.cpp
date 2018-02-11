// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <pybind11/stl_bind.h>
#include "indexing.h"

using namespace dlib;
using namespace std;

typedef matrix<double,0,1> cv;


void cv_set_size(cv& m, long s)
{
    m.set_size(s);
    m = 0;
}

double dotprod ( const cv& a, const cv& b)
{
    return dot(a,b);
}

string cv__str__(const cv& v)
{
    ostringstream sout;
    for (long i = 0; i < v.size(); ++i)
    {
        sout << v(i);
        if (i+1 < v.size())
            sout << "\n";
    }
    return sout.str();
}

string cv__repr__ (const cv& v)
{
    std::ostringstream sout;
    sout << "dlib.vector([";
    for (long i = 0; i < v.size(); ++i)
    {
        sout << v(i);
        if (i+1 < v.size())
            sout << ", ";
    }
    sout << "])";
    return sout.str();
}

std::shared_ptr<cv> cv_from_object(py::object obj)
{
    try {
        long nr = obj.cast<long>();
        auto temp = std::make_shared<cv>(nr);
        *temp = 0;
        return temp;
    } catch(py::cast_error &e) {
        py::list li = obj.cast<py::list>();
        const long nr = len(obj);
        auto temp = std::make_shared<cv>(nr);
        for ( long r = 0; r < nr; ++r)
        {
            (*temp)(r) = li[r].cast<double>();
        }
        return temp;
    }
}

long cv__len__(cv& c)
{
    return c.size();
}


void cv__setitem__(cv& c, long p, double val)
{
    if (p < 0) {
        p = c.size() + p; // negative index
    }
    if (p > c.size()-1) {
        PyErr_SetString( PyExc_IndexError, "index out of range"
        );
        throw py::error_already_set();
    }
    c(p) = val;
}

double cv__getitem__(cv& m, long r)
{
    if (r < 0) {
        r = m.size() + r; // negative index
    }
    if (r > m.size()-1 || r < 0) {
        PyErr_SetString( PyExc_IndexError, "index out of range"
        );
        throw py::error_already_set();
    }
    return m(r);
}


cv cv__getitem2__(cv& m, py::slice r)
{
    size_t start, stop, step, slicelength;
    if (!r.compute(m.size(), &start, &stop, &step, &slicelength))
        throw py::error_already_set();

    cv temp(slicelength);

    for (size_t i = 0; i < slicelength; ++i) {
         temp(i) = m(start); start += step;
    }
    return temp;
}

py::tuple cv_get_matrix_size(cv& m)
{
    return py::make_tuple(m.nr(), m.nc());
}

// ----------------------------------------------------------------------------------------

string point__repr__ (const point& p)
{
    std::ostringstream sout;
    sout << "point(" << p.x() << ", " << p.y() << ")";
    return sout.str();
}

string point__str__(const point& p)
{
    std::ostringstream sout;
    sout << "(" << p.x() << ", " << p.y() << ")";
    return sout.str();
}

long point_x(const point& p) { return p.x(); }
long point_y(const point& p) { return p.y(); }

// ----------------------------------------------------------------------------------------
void bind_vector(py::module& m)
{
    {
    py::class_<cv, std::shared_ptr<cv>>(m, "vector", "This object represents the mathematical idea of a column vector.")
        .def(py::init())
        .def("set_size", &cv_set_size)
        .def("resize", &cv_set_size)
        .def(py::init(&cv_from_object))
        .def("__repr__", &cv__repr__)
        .def("__str__", &cv__str__)
        .def("__len__", &cv__len__)
        .def("__getitem__", &cv__getitem__)
        .def("__getitem__", &cv__getitem2__)
        .def("__setitem__", &cv__setitem__)
        .def_property_readonly("shape", &cv_get_matrix_size)
        .def(py::pickle(&getstate<cv>, &setstate<cv>));

    m.def("dot", &dotprod, "Compute the dot product between two dense column vectors.");
    }
    {
    typedef point type;
    py::class_<type>(m, "point", "This object represents a single point of integer coordinates that maps directly to a dlib::point.")
            .def(py::init<long,long>(), py::arg("x"), py::arg("y"))
            .def("__repr__", &point__repr__)
            .def("__str__", &point__str__)
            .def_property("x", &point_x, [](point& p, long x){p.x()=x;}, "The x-coordinate of the point.")
            .def_property("y", &point_y, [](point& p, long y){p.x()=y;}, "The y-coordinate of the point.")
            .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef std::vector<point> type;
    py::bind_vector<type>(m, "points", "An array of point objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<point>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
}
