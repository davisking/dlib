// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include "serialize_pickle.h"


using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> cv;

void cv_set_size(cv& m, long s)
{
    m.set_size(s);
    m = 0;
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

boost::shared_ptr<cv> cv_from_object(object obj)
{
    extract<long> thesize(obj);
    if (thesize.check())
    {
        long nr = thesize;
        boost::shared_ptr<cv> temp(new cv(nr));
        *temp = 0;
        return temp;
    }
    else
    {
        const long nr = len(obj);
        boost::shared_ptr<cv> temp(new cv(nr));
        for ( long r = 0; r < nr; ++r)
        {
            (*temp)(r) = extract<double>(obj[r]);
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
        boost::python::throw_error_already_set();   
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
        boost::python::throw_error_already_set();   
    }
    return m(r);
}


tuple cv_get_matrix_size(cv& m)
{
    return make_tuple(m.nr(), m.nc());
}

void bind_vector()
{
    class_<cv>("vector", "This object represents the mathematical idea of a column vector.", init<>())
        .def("set_size", &cv_set_size)
        .def("resize", &cv_set_size)
        .def("__init__", make_constructor(&cv_from_object))
        .def("__repr__", &cv__repr__)
        .def("__str__", &cv__str__)
        .def("__len__", &cv__len__)
        .def("__getitem__", &cv__getitem__)
        .def("__setitem__", &cv__setitem__)
        .add_property("shape", &cv_get_matrix_size)
        .def_pickle(serialize_pickle<cv>());
}

