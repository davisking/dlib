
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include "serialize_pickle.h"
#include <dlib/svm.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> sample_type; 



template <typename kernel_type>
void bind_kernel(
)
{
    typedef svm_c_trainer<kernel_type> trainer;
    class_<trainer>("svm_c_trainer")
        .def("train", &trainer::template train<std::vector<sample_type>,std::vector<double> >);

}


void bind_svm_c_trainer()
{
    bind_kernel<linear_kernel<sample_type> >();

    /*
    class_<cv>("vector", init<>())
        .def("set_size", &cv_set_size)
        .def("__init__", make_constructor(&cv_from_object))
        .def("__repr__", &cv__str__)
        .def("__str__", &cv__str__)
        .def("__len__", &cv__len__)
        .def("__getitem__", &cv__getitem__)
        .add_property("shape", &cv_get_matrix_size)
        .def_pickle(serialize_pickle<cv>());
    */
}


