
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "serialize_pickle.h"
#include <dlib/svm.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> sample_type; 
typedef std::vector<std::pair<unsigned long,double> > sparse_vect;


template <typename decision_function>
double predict (
    const decision_function& df,
    const typename decision_function::kernel_type::sample_type& samp
)
{
    if (df.basis_vectors.size() == 0)
    {
        return 0;
    }
    else if (df.basis_vectors(0).size() != samp.size())
    {
        std::ostringstream sout;
        sout << "Input vector should have " << df.basis_vectors(0).size() 
             << " dimensions, not " << samp.size() << ".";
        PyErr_SetString( PyExc_ValueError, sout.str().c_str() );                                            
        boost::python::throw_error_already_set();   
    }
    return df(samp);
}

template <typename kernel_type>
void add_df (
    const std::string name
)
{
    typedef decision_function<kernel_type> df_type;
    class_<df_type>(name.c_str())
        .def("predict", &predict<df_type>)
        .def_pickle(serialize_pickle<df_type>());
}

template <typename df_type>
typename df_type::sample_type get_weights(
    const df_type& df
)
{
    if (df.basis_vectors.size() == 0)
    {
        PyErr_SetString( PyExc_ValueError, "Decision function is empty." );                                            
        boost::python::throw_error_already_set();   
    }
    df_type temp = simplify_linear_decision_function(df);
    return temp.basis_vectors(0);
}

template <typename df_type>
typename df_type::scalar_type get_bias(
    const df_type& df
)
{
    if (df.basis_vectors.size() == 0)
    {
        PyErr_SetString( PyExc_ValueError, "Decision function is empty." );                                            
        boost::python::throw_error_already_set();   
    }
    return df.b;
}

template <typename df_type>
void set_bias(
    df_type& df,
    double b
)
{
    if (df.basis_vectors.size() == 0)
    {
        PyErr_SetString( PyExc_ValueError, "Decision function is empty." );                                            
        boost::python::throw_error_already_set();   
    }
    df.b = b;
}

template <typename kernel_type>
void add_linear_df (
    const std::string name
)
{
    typedef decision_function<kernel_type> df_type;
    class_<df_type>(name.c_str())
        .def("predict", predict<df_type>)
        .add_property("weights", &get_weights<df_type>)
        .add_property("bias", get_bias<df_type>, set_bias<df_type>)
        .def_pickle(serialize_pickle<df_type>());
}

void bind_decision_functions()
{
    add_linear_df<linear_kernel<sample_type> >("_decision_function_linear");
    add_linear_df<sparse_linear_kernel<sparse_vect> >("_decision_function_sparse_linear");

    add_df<histogram_intersection_kernel<sample_type> >("_decision_function_histogram_intersection");
    add_df<sparse_histogram_intersection_kernel<sparse_vect> >("_decision_function_sparse_histogram_intersection");

    add_df<polynomial_kernel<sample_type> >("_decision_function_polynomial");
    add_df<sparse_polynomial_kernel<sparse_vect> >("_decision_function_sparse_polynomial");

    add_df<radial_basis_kernel<sample_type> >("_decision_function_radial_basis");
    add_df<sparse_radial_basis_kernel<sparse_vect> >("_decision_function_sparse_radial_basis");

    add_df<sigmoid_kernel<sample_type> >("_decision_function_sigmoid");
    add_df<sparse_sigmoid_kernel<sparse_vect> >("_decision_function_sparse_sigmoid");
}



