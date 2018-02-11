// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include "testing_results.h"
#include <dlib/svm.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

typedef matrix<double,0,1> sample_type;
typedef std::vector<std::pair<unsigned long,double> > sparse_vect;

template <typename decision_function>
double predict (
    const decision_function& df,
    const typename decision_function::kernel_type::sample_type& samp
)
{
    typedef typename decision_function::kernel_type::sample_type T;
    if (df.basis_vectors.size() == 0)
    {
        return 0;
    }
    else if (is_matrix<T>::value && df.basis_vectors(0).size() != samp.size())
    {
        std::ostringstream sout;
        sout << "Input vector should have " << df.basis_vectors(0).size() 
             << " dimensions, not " << samp.size() << ".";
        PyErr_SetString( PyExc_ValueError, sout.str().c_str() );
        throw py::error_already_set();
    }
    return df(samp);
}

template <typename kernel_type>
void add_df (
    py::module& m,
    const std::string name
)
{
    typedef decision_function<kernel_type> df_type;
    py::class_<df_type>(m, name.c_str())
        .def("__call__", &predict<df_type>)
        .def(py::pickle(&getstate<df_type>, &setstate<df_type>));
}

template <typename df_type>
typename df_type::sample_type get_weights(
    const df_type& df
)
{
    if (df.basis_vectors.size() == 0)
    {
        PyErr_SetString( PyExc_ValueError, "Decision function is empty." );
        throw py::error_already_set();
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
        throw py::error_already_set();
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
        throw py::error_already_set();
    }
    df.b = b;
}

template <typename kernel_type>
void add_linear_df (
    py::module &m,
    const std::string name
)
{
    typedef decision_function<kernel_type> df_type;
    py::class_<df_type>(m, name.c_str())
        .def("__call__", predict<df_type>)
        .def_property_readonly("weights", &get_weights<df_type>)
        .def_property("bias", get_bias<df_type>, set_bias<df_type>)
        .def(py::pickle(&getstate<df_type>, &setstate<df_type>));
}

// ----------------------------------------------------------------------------------------

std::string binary_test__str__(const binary_test& item)
{
    std::ostringstream sout;
    sout << "class1_accuracy: "<< item.class1_accuracy << "  class2_accuracy: "<< item.class2_accuracy; 
    return sout.str();
}
std::string binary_test__repr__(const binary_test& item) { return "< " + binary_test__str__(item) + " >";}

std::string regression_test__str__(const regression_test& item)
{
    std::ostringstream sout;
    sout << "mean_squared_error: "<< item.mean_squared_error << "  R_squared: "<< item.R_squared; 
    sout << "  mean_average_error: "<< item.mean_average_error << "  mean_error_stddev: "<< item.mean_error_stddev; 
    return sout.str();
}
std::string regression_test__repr__(const regression_test& item) { return "< " + regression_test__str__(item) + " >";}

std::string ranking_test__str__(const ranking_test& item)
{
    std::ostringstream sout;
    sout << "ranking_accuracy: "<< item.ranking_accuracy << "  mean_ap: "<< item.mean_ap; 
    return sout.str();
}
std::string ranking_test__repr__(const ranking_test& item) { return "< " + ranking_test__str__(item) + " >";}

// ----------------------------------------------------------------------------------------

template <typename K>
binary_test  _test_binary_decision_function (
    const decision_function<K>& dec_funct,
    const std::vector<typename K::sample_type>& x_test,
    const std::vector<double>& y_test
) { return binary_test(test_binary_decision_function(dec_funct, x_test, y_test)); }

template <typename K>
regression_test _test_regression_function (
    const decision_function<K>& reg_funct,
    const std::vector<typename K::sample_type>& x_test,
    const std::vector<double>& y_test
) { return regression_test(test_regression_function(reg_funct, x_test, y_test)); }

template < typename K >
ranking_test _test_ranking_function1 (
    const decision_function<K>& funct,
    const std::vector<ranking_pair<typename K::sample_type> >& samples
) { return ranking_test(test_ranking_function(funct, samples)); }

template < typename K >
ranking_test _test_ranking_function2 (
    const decision_function<K>& funct,
    const ranking_pair<typename K::sample_type>& sample
) { return ranking_test(test_ranking_function(funct, sample)); }


void bind_decision_functions(py::module &m)
{
    add_linear_df<linear_kernel<sample_type> >(m, "_decision_function_linear");
    add_linear_df<sparse_linear_kernel<sparse_vect> >(m, "_decision_function_sparse_linear");

    add_df<histogram_intersection_kernel<sample_type> >(m, "_decision_function_histogram_intersection");
    add_df<sparse_histogram_intersection_kernel<sparse_vect> >(m, "_decision_function_sparse_histogram_intersection");

    add_df<polynomial_kernel<sample_type> >(m, "_decision_function_polynomial");
    add_df<sparse_polynomial_kernel<sparse_vect> >(m, "_decision_function_sparse_polynomial");

    add_df<radial_basis_kernel<sample_type> >(m, "_decision_function_radial_basis");
    add_df<sparse_radial_basis_kernel<sparse_vect> >(m, "_decision_function_sparse_radial_basis");

    add_df<sigmoid_kernel<sample_type> >(m, "_decision_function_sigmoid");
    add_df<sparse_sigmoid_kernel<sparse_vect> >(m, "_decision_function_sparse_sigmoid");


    m.def("test_binary_decision_function", _test_binary_decision_function<linear_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<sparse_linear_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<radial_basis_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<sparse_radial_basis_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<polynomial_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<sparse_polynomial_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<histogram_intersection_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<sparse_histogram_intersection_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<sigmoid_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _test_binary_decision_function<sparse_sigmoid_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));

    m.def("test_regression_function", _test_regression_function<linear_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<sparse_linear_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<radial_basis_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<sparse_radial_basis_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<histogram_intersection_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<sparse_histogram_intersection_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<sigmoid_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<sparse_sigmoid_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<polynomial_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));
    m.def("test_regression_function", _test_regression_function<sparse_polynomial_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"), py::arg("targets"));

    m.def("test_ranking_function", _test_ranking_function1<linear_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"));
    m.def("test_ranking_function", _test_ranking_function1<sparse_linear_kernel<sparse_vect> >,
        py::arg("function"), py::arg("samples"));
    m.def("test_ranking_function", _test_ranking_function2<linear_kernel<sample_type> >,
        py::arg("function"), py::arg("sample"));
    m.def("test_ranking_function", _test_ranking_function2<sparse_linear_kernel<sparse_vect> >,
        py::arg("function"), py::arg("sample"));


    py::class_<binary_test>(m, "_binary_test")
        .def("__str__", binary_test__str__)
        .def("__repr__", binary_test__repr__)
        .def_readwrite("class1_accuracy", &binary_test::class1_accuracy,
            "A value between 0 and 1, measures accuracy on the +1 class.")
        .def_readwrite("class2_accuracy", &binary_test::class2_accuracy,
            "A value between 0 and 1, measures accuracy on the -1 class.");

    py::class_<ranking_test>(m, "_ranking_test")
        .def("__str__", ranking_test__str__)
        .def("__repr__", ranking_test__repr__)
        .def_readwrite("ranking_accuracy", &ranking_test::ranking_accuracy,
            "A value between 0 and 1, measures the fraction of times a relevant sample was ordered before a non-relevant sample.")
        .def_readwrite("mean_ap", &ranking_test::mean_ap,
            "A value between 0 and 1, measures the mean average precision of the ranking.");

    py::class_<regression_test>(m, "_regression_test")
        .def("__str__", regression_test__str__)
        .def("__repr__", regression_test__repr__)
        .def_readwrite("mean_average_error", &regression_test::mean_average_error,
            "The mean average error of a regression function on a dataset.")
        .def_readwrite("mean_error_stddev", &regression_test::mean_error_stddev,
            "The standard deviation of the absolute value of the error of a regression function on a dataset.")
        .def_readwrite("mean_squared_error", &regression_test::mean_squared_error,
            "The mean squared error of a regression function on a dataset.")
        .def_readwrite("R_squared", &regression_test::R_squared,
            "A value between 0 and 1, measures the squared correlation between the output of a \n"
            "regression function and the target values.");
}



