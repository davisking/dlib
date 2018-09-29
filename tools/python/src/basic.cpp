// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <sstream>
#include <string>
#include "opaque_types.h"

#include <dlib/string.h>
#include <pybind11/stl_bind.h>

using namespace std;
using namespace dlib;
namespace py = pybind11;


std::shared_ptr<std::vector<double> > array_from_object(py::object obj)
{
    try {
        long nr = obj.cast<long>();
        return std::make_shared<std::vector<double>>(nr);
    } catch (py::cast_error &e) {
        py::list li = obj.cast<py::list>();
        const long nr = len(li);
        auto temp = std::make_shared<std::vector<double>>(nr);
        for ( long r = 0; r < nr; ++r)
        {
            (*temp)[r] = li[r].cast<double>();
        }
        return temp;
    }
}

string array__str__ (const std::vector<double>& v)
{
    std::ostringstream sout;
    for (unsigned long i = 0; i < v.size(); ++i)
    {
        sout << v[i];
        if (i+1 < v.size())
            sout << "\n";
    }
    return sout.str();
}

string array__repr__ (const std::vector<double>& v)
{
    std::ostringstream sout;
    sout << "dlib.array([";
    for (unsigned long i = 0; i < v.size(); ++i)
    {
        sout << v[i];
        if (i+1 < v.size())
            sout << ", ";
    }
    sout << "])";
    return sout.str();
}

string range__str__ (const std::pair<unsigned long,unsigned long>& p)
{
    std::ostringstream sout;
    sout << p.first << ", " << p.second;
    return sout.str();
}

string range__repr__ (const std::pair<unsigned long,unsigned long>& p)
{
    std::ostringstream sout;
    sout << "dlib.range(" << p.first << ", " << p.second << ")";
    return sout.str();
}

struct range_iter
{
    std::pair<unsigned long,unsigned long> range;
    unsigned long cur;

    unsigned long next()
    {
        if (cur < range.second)
        {
            return cur++;
        }
        else
        {
            PyErr_SetString(PyExc_StopIteration, "No more data.");
            throw py::error_already_set();
        }
    }
};

range_iter make_range_iterator (const std::pair<unsigned long,unsigned long>& p)
{
    range_iter temp;
    temp.range = p;
    temp.cur = p.first;
    return temp;
}

string pair__str__ (const std::pair<unsigned long,double>& p)
{
    std::ostringstream sout;
    sout << p.first << ": " << p.second;
    return sout.str();
}

string pair__repr__ (const std::pair<unsigned long,double>& p)
{
    std::ostringstream sout;
    sout << "dlib.pair(" << p.first << ", " << p.second << ")";
    return sout.str();
}

string sparse_vector__str__ (const std::vector<std::pair<unsigned long,double> >& v)
{
    std::ostringstream sout;
    for (unsigned long i = 0; i < v.size(); ++i)
    {
        sout << v[i].first << ": " << v[i].second;
        if (i+1 < v.size())
            sout << "\n";
    }
    return sout.str();
}

string sparse_vector__repr__ (const std::vector<std::pair<unsigned long,double> >& v)
{
    std::ostringstream sout;
    sout << "< dlib.sparse_vector containing: \n" << sparse_vector__str__(v) << " >";
    return sout.str();
}

unsigned long range_len(const std::pair<unsigned long, unsigned long>& r)
{
    if (r.second > r.first)
        return r.second-r.first;
    else
        return 0;
}

template <typename T>
void resize(T& v, unsigned long n) { v.resize(n); }

void bind_basic_types(py::module& m)
{
    {
    typedef double item_type;
    typedef std::vector<item_type> type;
    typedef std::shared_ptr<type> type_ptr;
    py::bind_vector<type, type_ptr >(m, "array", "This object represents a 1D array of floating point numbers. "
        "Moreover, it binds directly to the C++ type std::vector<double>.")
        .def(py::init(&array_from_object))
        .def("__str__", array__str__)
        .def("__repr__", array__repr__)
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef matrix<double,0,1> item_type;
    typedef std::vector<item_type > type;
    py::bind_vector<type>(m, "vectors", "This object is an array of vector objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef std::vector<matrix<double,0,1> > item_type;
    typedef std::vector<item_type > type;
    py::bind_vector<type>(m, "vectorss", "This object is an array of arrays of vector objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    typedef pair<unsigned long,unsigned long> range_type;
    py::class_<range_type>(m, "range", "This object is used to represent a range of elements in an array.")
        .def(py::init<unsigned long,unsigned long>())
        .def(py::init([](unsigned long end){return range_type(0,end); }))
        .def_readwrite("begin",&range_type::first, "The index of the first element in the range.  This is represented using an unsigned integer.")
        .def_readwrite("end",&range_type::second, "One past the index of the last element in the range.  This is represented using an unsigned integer.")
        .def("__str__", range__str__)
        .def("__repr__", range__repr__)
        .def("__iter__", &make_range_iterator)
        .def("__len__", &range_len)
        .def(py::pickle(&getstate<range_type>, &setstate<range_type>));

    py::class_<range_iter>(m, "_range_iter")
        .def("next", &range_iter::next)
        .def("__next__", &range_iter::next);

    {
    typedef std::pair<unsigned long, unsigned long> item_type;
    typedef std::vector<item_type > type;
    py::bind_vector<type>(m, "ranges", "This object is an array of range objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef std::vector<std::pair<unsigned long, unsigned long> > item_type;
    typedef std::vector<item_type > type;
    py::bind_vector<type>(m, "rangess", "This object is an array of arrays of range objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }


    typedef pair<unsigned long,double> pair_type;
    py::class_<pair_type>(m, "pair", "This object is used to represent the elements of a sparse_vector.")
        .def(py::init<unsigned long,double>())
        .def_readwrite("first",&pair_type::first, "This field represents the index/dimension number.")
        .def_readwrite("second",&pair_type::second, "This field contains the value in a vector at dimension specified by the first field.")
        .def("__str__", pair__str__)
        .def("__repr__", pair__repr__)
        .def(py::pickle(&getstate<pair_type>, &setstate<pair_type>));

    {
    typedef std::vector<pair_type> type;
    py::bind_vector<type>(m, "sparse_vector",
"This object represents the mathematical idea of a sparse column vector.  It is    \n\
simply an array of dlib.pair objects, each representing an index/value pair in    \n\
the vector.  Any elements of the vector which are missing are implicitly set to    \n\
zero.      \n\
    \n\
Unless otherwise noted, any routines taking a sparse_vector assume the sparse    \n\
vector is sorted and has unique elements.  That is, the index values of the    \n\
pairs in a sparse_vector should be listed in increasing order and there should    \n\
not be duplicates.  However, some functions work with \"unsorted\" sparse    \n\
vectors.  These are dlib.sparse_vector objects that have either duplicate    \n\
entries or non-sorted index values.  Note further that you can convert an    \n\
\"unsorted\" sparse_vector into a properly sorted sparse vector by calling    \n\
dlib.make_sparse_vector() on it.   "
        )
        .def("__str__", sparse_vector__str__)
        .def("__repr__", sparse_vector__repr__)
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<pair_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef std::vector<pair_type> item_type;
    typedef std::vector<item_type > type;
    py::bind_vector<type>(m, "sparse_vectors", "This object is an array of sparse_vector objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    typedef std::vector<std::vector<pair_type> > item_type;
    typedef std::vector<item_type > type;
    py::bind_vector<type>(m, "sparse_vectorss", "This object is an array of arrays of sparse_vector objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<item_type>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
}

