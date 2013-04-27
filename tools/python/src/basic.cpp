#include <boost/python.hpp>
#include <dlib/matrix.h>
#include <sstream>
#include <string>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/indexing_suite.hpp>
#include <boost/shared_ptr.hpp>

#include <dlib/string.h>
#include "serialize_pickle.h"

using namespace std;
using namespace dlib;
using namespace boost::python;


boost::shared_ptr<std::vector<double> > array_from_object(object obj)
{
    extract<long> thesize(obj);
    if (thesize.check())
    {
        long nr = thesize;
        boost::shared_ptr<std::vector<double> > temp(new std::vector<double>(nr));
        return temp;
    }
    else
    {
        const long nr = len(obj);
        boost::shared_ptr<std::vector<double> > temp(new std::vector<double>(nr));
        for ( long r = 0; r < nr; ++r)
        {
            (*temp)[r] = extract<double>(obj[r]);
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


void bind_basic_types() 
{
    class_<std::vector<double> >("array", init<>())
        .def(vector_indexing_suite<std::vector<double> >())
        .def("__init__", make_constructor(&array_from_object))
        .def("__str__", array__str__)
        .def("__repr__", array__repr__)
        .def_pickle(serialize_pickle<std::vector<double> >());

    class_<std::vector<matrix<double,0,1> > >("vectors")
        .def(vector_indexing_suite<std::vector<matrix<double,0,1> > >())
        .def_pickle(serialize_pickle<std::vector<matrix<double,0,1> > >());

    typedef pair<unsigned long,double> pair_type;
    class_<pair_type>("pair", "This object is used to represent the elements of a sparse_vector.", init<>() )
        .def(init<unsigned long,double>())
        .def_readwrite("first",&pair_type::first, "This field represents the index/dimension number.")
        .def_readwrite("second",&pair_type::second, "This field contains the value in a vector at dimension specified by the first field.")
        .def("__str__", pair__str__)
        .def("__repr__", pair__repr__)
        .def_pickle(serialize_pickle<pair_type>());

    class_<std::vector<pair_type> >("sparse_vector")
        .def(vector_indexing_suite<std::vector<pair_type> >())
        .def("__str__", sparse_vector__str__)
        .def("__repr__", sparse_vector__repr__)
        .def_pickle(serialize_pickle<std::vector<pair_type> >());

    class_<std::vector<std::vector<pair_type> > >("sparse_vectors")
        .def(vector_indexing_suite<std::vector<std::vector<pair_type> > >())
        .def_pickle(serialize_pickle<std::vector<std::vector<pair_type> > >());

}

