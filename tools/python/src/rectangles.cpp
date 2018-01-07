// Copyright (C) 2015 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/geometry.h>
#include <pybind11/stl_bind.h>
#include "indexing.h"

using namespace dlib;
using namespace std;

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<rectangle>);

// ----------------------------------------------------------------------------------------

long left(const rectangle& r) { return r.left(); }
long top(const rectangle& r) { return r.top(); }
long right(const rectangle& r) { return r.right(); }
long bottom(const rectangle& r) { return r.bottom(); }
long width(const rectangle& r) { return r.width(); }
long height(const rectangle& r) { return r.height(); }
unsigned long area(const rectangle& r) { return r.area(); }

double dleft(const drectangle& r) { return r.left(); }
double dtop(const drectangle& r) { return r.top(); }
double dright(const drectangle& r) { return r.right(); }
double dbottom(const drectangle& r) { return r.bottom(); }
double dwidth(const drectangle& r) { return r.width(); }
double dheight(const drectangle& r) { return r.height(); }
double darea(const drectangle& r) { return r.area(); }

template <typename rect_type>
bool is_empty(const rect_type& r) { return r.is_empty(); }

template <typename rect_type>
point center(const rect_type& r) { return center(r); }

template <typename rect_type>
point dcenter(const rect_type& r) { return dcenter(r); }

template <typename rect_type>
bool contains(const rect_type& r, const point& p) { return r.contains(p); }

template <typename rect_type>
bool contains_xy(const rect_type& r, const long x, const long y) { return r.contains(point(x, y)); }

template <typename rect_type>
bool contains_rec(const rect_type& r, const rect_type& r2) { return r.contains(r2); }

template <typename rect_type>
rect_type intersect(const rect_type& r, const rect_type& r2) { return r.intersect(r2); }

template <typename rect_type>
string print_rectangle_str(const rect_type& r)
{
    std::ostringstream sout;
    sout << r;
    return sout.str();
}

template <typename rect_type>
string print_rectangle_repr(const rect_type& r)
{
    std::ostringstream sout;
    sout << "rectangle(" << r.left() << "," << r.top() << "," << r.right() << "," << r.bottom() << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

void bind_rectangles(py::module& m)
{
    {
    typedef rectangle type;
    py::class_<type>(m, "rectangle", "This object represents a rectangular area of an image.")
        .def(py::init<long,long,long,long>(), py::arg("left"),py::arg("top"),py::arg("right"),py::arg("bottom"))
        .def("area",   &::area)
        .def("left",   &::left)
        .def("top",    &::top)
        .def("right",  &::right)
        .def("bottom", &::bottom)
        .def("width",  &::width)
        .def("height", &::height)
        .def("is_empty", &::is_empty<type>)
        .def("center", &::center<type>)
        .def("dcenter", &::dcenter<type>)
        .def("contains", &::contains<type>, py::arg("point"))
        .def("contains", &::contains_xy<type>, py::arg("x"), py::arg("y"))
        .def("contains", &::contains_rec<type>, py::arg("rectangle"))
        .def("intersect", &::intersect<type>, py::arg("rectangle"))
        .def("__str__", &::print_rectangle_str<type>)
        .def("__repr__", &::print_rectangle_repr<type>)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef drectangle type;
    py::class_<type>(m, "drectangle", "This object represents a rectangular area of an image with floating point coordinates.")
        .def(py::init<double,double,double,double>(), py::arg("left"), py::arg("top"), py::arg("right"), py::arg("bottom"))
        .def("area",   &::darea)
        .def("left",   &::dleft)
        .def("top",    &::dtop)
        .def("right",  &::dright)
        .def("bottom", &::dbottom)
        .def("width",  &::dwidth)
        .def("height", &::dheight)
        .def("is_empty", &::is_empty<type>)
        .def("center", &::center<type>)
        .def("dcenter", &::dcenter<type>)
        .def("contains", &::contains<type>, py::arg("point"))
        .def("contains", &::contains_xy<type>, py::arg("x"), py::arg("y"))
        .def("contains", &::contains_rec<type>, py::arg("rectangle"))
        .def("intersect", &::intersect<type>, py::arg("rectangle"))
        .def("__str__", &::print_rectangle_str<type>)
        .def("__repr__", &::print_rectangle_repr<type>)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef std::vector<rectangle> type;
    py::bind_vector<type>(m, "rectangles", "An array of rectangle objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<rectangle>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
}

// ----------------------------------------------------------------------------------------
