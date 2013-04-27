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


void bind_matrix();
void bind_vector();
void bind_svm_c_trainer();

BOOST_PYTHON_MODULE(dlib)
{
    bind_matrix();
    bind_vector();
    bind_svm_c_trainer();

    class_<std::vector<double> >("array")
        .def(vector_indexing_suite<std::vector<double> >())
        .def_pickle(serialize_pickle<std::vector<double> >());

    class_<std::vector<matrix<double,0,1> > >("vectors")
        .def(vector_indexing_suite<std::vector<matrix<double,0,1> > >())
        .def_pickle(serialize_pickle<std::vector<matrix<double,0,1> > >());

    typedef pair<unsigned long,double> pair_type;
    class_<pair_type>("pair", "help message", init<>() )
        .def(init<unsigned long,double>())
        .def_readwrite("first",&pair_type::first, "THE FIRST, LOVE IT!")
        .def_readwrite("second",&pair_type::second)
        .def_pickle(serialize_pickle<pair_type>());

    class_<std::vector<pair_type> >("sparse_vector")
        .def(vector_indexing_suite<std::vector<pair_type> >())
        .def_pickle(serialize_pickle<std::vector<pair_type> >());

    class_<std::vector<std::vector<pair_type> > >("sparse_vectors")
        .def(vector_indexing_suite<std::vector<std::vector<pair_type> > >())
        .def_pickle(serialize_pickle<std::vector<std::vector<pair_type> > >());

    /*
    def("tomat",tomat);
    def("add_to_map", add_to_map);
    def("getpair", getpair);
    def("getmatrix", getmatrix);
    def("yay", yay);
    def("sum", sum_mat);
    def("getmap", getmap);
    def("go", go);
    def("append_to_vector", append_to_vector);
    */




}
