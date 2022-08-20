// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include "testing_results.h"
#include <dlib/svm.h>
#include <chrono>

using namespace dlib;
using namespace std;

namespace py = pybind11;

typedef matrix<double,0,1> sample_type;
typedef std::vector<std::pair<unsigned long,double> > sparse_vect;

void np_to_cpp (
    const numpy_image<double>& x_,
    std::vector<matrix<double,0,1>>& samples
)
{
    auto x = make_image_view(x_);
    DLIB_CASSERT(x.nc() > 0);
    DLIB_CASSERT(x.nr() > 0);
    samples.resize(x.nr());
    for (long r = 0; r < x.nr(); ++r)
    {
        samples[r].set_size(x.nc());
        for (long c = 0; c < x.nc(); ++c)
        {
            samples[r](c) = x[r][c];
        }
    }
}

void np_to_cpp (
    const numpy_image<double>& x_,
    const py::array_t<double>& y,
    std::vector<matrix<double,0,1>>& samples,
    std::vector<double>& labels
)
{
    DLIB_CASSERT(y.ndim() == 1 && y.size() > 0);
    labels.assign(y.data(), y.data()+y.size());
    auto x = make_image_view(x_);
    DLIB_CASSERT(x.nr() == y.size(), "The x matrix must have as many rows as y has elements.");
    DLIB_CASSERT(x.nc() > 0);
    samples.resize(x.nr());
    for (long r = 0; r < x.nr(); ++r)
    {
        samples[r].set_size(x.nc());
        for (long c = 0; c < x.nc(); ++c)
        {
            samples[r](c) = x[r][c];
        }
    }
}


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

inline matrix<double,0,1> np_to_mat(
    const py::array_t<double>& samp
)
{
    matrix<double,0,1> temp(samp.size());

    const auto data = samp.data();
    for (long i = 0; i < temp.size(); ++i)
        temp(i) = data[i];
    return temp;
}

template <typename decision_function>
double normalized_predict (
    const normalized_function<decision_function>& df,
    const typename decision_function::kernel_type::sample_type& samp
)
{
    typedef typename decision_function::kernel_type::sample_type T;
    if (df.function.basis_vectors.size() == 0)
    {
        return 0;
    }
    else if (is_matrix<T>::value && df.function.basis_vectors(0).size() != samp.size())
    {
        std::ostringstream sout;
        sout << "Input vector should have " << df.function.basis_vectors(0).size() 
             << " dimensions, not " << samp.size() << ".";
        PyErr_SetString( PyExc_ValueError, sout.str().c_str() );
        throw py::error_already_set();
    }
    return df(samp);
}

template <typename decision_function>
std::vector<double> normalized_predict_vec (
    const normalized_function<decision_function>& df,
    const std::vector<typename decision_function::kernel_type::sample_type>& samps
)
{
    std::vector<double> out;
    out.reserve(samps.size());
    for (const auto& x : samps)
        out.push_back(normalized_predict(df,x));
    return out;
}

template <typename decision_function>
py::array_t<double> normalized_predict_np_vec (
    const normalized_function<decision_function>& df,
    const numpy_image<double>& samps_
)
{
    auto samps = make_image_view(samps_);

    if (df.function.basis_vectors(0).size() != samps.nc())
    {
        std::ostringstream sout;
        sout << "Input vector should have " << df.function.basis_vectors(0).size() 
             << " dimensions, not " << samps.nc() << ".";
        PyErr_SetString( PyExc_ValueError, sout.str().c_str() );
        throw py::error_already_set();
    }

    py::array_t<double, py::array::c_style> out((size_t)samps.nr());
    matrix<double,0,1> temp(samps.nc());
    auto data = out.mutable_data();
    for (long r = 0; r < samps.nr(); ++r)
    {
        for (long c = 0; c < samps.nc(); ++c)
            temp(c) = samps[r][c];
        *data++ = df(temp);
    }
    return out;
}

template <typename decision_function>
double normalized_predict_np (
    const normalized_function<decision_function>& df,
    const py::array_t<double>& samp
)
{
    typedef typename decision_function::kernel_type::sample_type T;
    if (df.function.basis_vectors.size() == 0)
    {
        return 0;
    }
    else if (is_matrix<T>::value && df.function.basis_vectors(0).size() != samp.size())
    {
        std::ostringstream sout;
        sout << "Input vector should have " << df.function.basis_vectors(0).size() 
             << " dimensions, not " << samp.size() << ".";
        PyErr_SetString( PyExc_ValueError, sout.str().c_str() );
        throw py::error_already_set();
    }
    return df(np_to_mat(samp));
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
        .def_property_readonly("alpha", [](const df_type& df) {return df.alpha;})
        .def_property_readonly("b", [](const df_type& df) {return df.b;})
        .def_property_readonly("kernel_function", [](const df_type& df) {return df.kernel_function;})
        .def_property_readonly("basis_vectors", [](const df_type& df) {
            std::vector<matrix<double,0,1>> temp;
            for (long i = 0; i < df.basis_vectors.size(); ++i)
                temp.push_back(sparse_to_dense(df.basis_vectors(i)));
            return temp;
        })
        .def(py::pickle(&getstate<df_type>, &setstate<df_type>));
}

template <typename kernel_type>
void add_normalized_df (
    py::module& m,
    const std::string name
)
{
    using df_type = normalized_function<decision_function<kernel_type>>;

    py::class_<df_type>(m, name.c_str())
        .def("__call__", &normalized_predict<decision_function<kernel_type>>)
        .def("__call__", &normalized_predict_np<decision_function<kernel_type>>)
        .def("batch_predict", &normalized_predict_vec<decision_function<kernel_type>>)
        .def("batch_predict", &normalized_predict_np_vec<decision_function<kernel_type>>)
        .def_property_readonly("alpha", [](const df_type& df) {return df.function.alpha;})
        .def_property_readonly("b", [](const df_type& df) {return df.function.b;})
        .def_property_readonly("kernel_function", [](const df_type& df) {return df.function.kernel_function;})
        .def_property_readonly("basis_vectors", [](const df_type& df) {
            std::vector<matrix<double,0,1>> temp;
            for (long i = 0; i < df.function.basis_vectors.size(); ++i)
            temp.push_back(sparse_to_dense(df.function.basis_vectors(i)));
            return temp;
        })
    .def_property_readonly("means", [](const df_type& df) {return df.normalizer.means();},
        "Input vectors are normalized by the equation, (x-means)*invstd_devs, before being passed to the underlying RBF function.")
    .def_property_readonly("invstd_devs", [](const df_type& df) {return df.normalizer.std_devs();},
        "Input vectors are normalized by the equation, (x-means)*invstd_devs, before being passed to the underlying RBF function.")
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

std::string radial_basis_kernel__repr__(const radial_basis_kernel<sample_type>& item)
{
    std::ostringstream sout;
    sout << "radial_basis_kernel(gamma="<< item.gamma<<")"; 
    return sout.str();
}

std::string linear_kernel__repr__(const linear_kernel<sample_type>& item)
{
    std::ostringstream sout;
    sout << "linear_kernel()"; 
    return sout.str();
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
binary_test  _normalized_test_binary_decision_function (
    const normalized_function<decision_function<K>>& dec_funct,
    const std::vector<typename K::sample_type>& x_test,
    const std::vector<double>& y_test
) { return binary_test(test_binary_decision_function(dec_funct, x_test, y_test)); }

template <typename K>
binary_test  _normalized_test_binary_decision_function_np (
    const normalized_function<decision_function<K>>& dec_funct,
    const numpy_image<double>& x_test_,
    const py::array_t<double>& y_test_
) 
{ 
    std::vector<typename K::sample_type> x_test;
    std::vector<double> y_test;
    np_to_cpp(x_test_,y_test_, x_test,y_test);
    return binary_test(test_binary_decision_function(dec_funct, x_test, y_test)); 
}

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

// ----------------------------------------------------------------------------------------


void setup_auto_train_rbf_classifier (py::module& m)
{
    m.def("auto_train_rbf_classifier", [](
        const std::vector<matrix<double,0,1>>& x,
        const std::vector<double>& y,
        double max_runtime_seconds,
        bool be_verbose 
    ) { return auto_train_rbf_classifier(x,y,std::chrono::microseconds((uint64_t)(max_runtime_seconds*1e6)),be_verbose); },
        py::arg("x"), py::arg("y"), py::arg("max_runtime_seconds"), py::arg("be_verbose")=true,
"requires \n\
    - y contains at least 6 examples of each class.  Moreover, every element in y \n\
      is either +1 or -1. \n\
    - max_runtime_seconds >= 0 \n\
    - len(x) == len(y) \n\
    - all the vectors in x have the same dimension. \n\
ensures \n\
    - This routine trains a radial basis function SVM on the given binary \n\
      classification training data.  It uses the svm_c_trainer to do this.  It also \n\
      uses find_max_global() and 6-fold cross-validation to automatically determine \n\
      the best settings of the SVM's hyper parameters. \n\
    - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the \n\
      returned function, df, should generally satisfy sign(df(x[i])) == y[i] as \n\
      often as possible. \n\
    - The hyperparameter search will run for about max_runtime and will print \n\
      messages to the screen as it runs if be_verbose==true." 
    /*!
        requires
            - y contains at least 6 examples of each class.  Moreover, every element in y
              is either +1 or -1.
            - max_runtime_seconds >= 0
            - len(x) == len(y)
            - all the vectors in x have the same dimension.
        ensures
            - This routine trains a radial basis function SVM on the given binary
              classification training data.  It uses the svm_c_trainer to do this.  It also
              uses find_max_global() and 6-fold cross-validation to automatically determine
              the best settings of the SVM's hyper parameters.
            - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the
              returned function, df, should generally satisfy sign(df(x[i])) == y[i] as
              often as possible.
            - The hyperparameter search will run for about max_runtime and will print
              messages to the screen as it runs if be_verbose==true.
    !*/
    );

    m.def("auto_train_rbf_classifier", [](
        const numpy_image<double>& x_,
        const py::array_t<double>& y_,
        double max_runtime_seconds,
        bool be_verbose 
    ) {
        std::vector<matrix<double,0,1>> x;
        std::vector<double> y;
        np_to_cpp(x_,y_, x, y);
        return auto_train_rbf_classifier(x,y,std::chrono::microseconds((uint64_t)(max_runtime_seconds*1e6)),be_verbose); },
        py::arg("x"), py::arg("y"), py::arg("max_runtime_seconds"), py::arg("be_verbose")=true,
"requires \n\
    - y contains at least 6 examples of each class.  Moreover, every element in y \n\
      is either +1 or -1. \n\
    - max_runtime_seconds >= 0 \n\
    - len(x.shape(0)) == len(y) \n\
    - x.shape(1) > 0 \n\
ensures \n\
    - This routine trains a radial basis function SVM on the given binary \n\
      classification training data.  It uses the svm_c_trainer to do this.  It also \n\
      uses find_max_global() and 6-fold cross-validation to automatically determine \n\
      the best settings of the SVM's hyper parameters. \n\
    - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the \n\
      returned function, df, should generally satisfy sign(df(x[i])) == y[i] as \n\
      often as possible. \n\
    - The hyperparameter search will run for about max_runtime and will print \n\
      messages to the screen as it runs if be_verbose==true." 
    /*!
        requires
            - y contains at least 6 examples of each class.  Moreover, every element in y
              is either +1 or -1.
            - max_runtime_seconds >= 0
            - len(x.shape(0)) == len(y)
            - x.shape(1) > 0
        ensures
            - This routine trains a radial basis function SVM on the given binary
              classification training data.  It uses the svm_c_trainer to do this.  It also
              uses find_max_global() and 6-fold cross-validation to automatically determine
              the best settings of the SVM's hyper parameters.
            - Note that we interpret y[i] as the label for the vector x[i].  Therefore, the
              returned function, df, should generally satisfy sign(df(x[i])) == y[i] as
              often as possible.
            - The hyperparameter search will run for about max_runtime and will print
              messages to the screen as it runs if be_verbose==true.
    !*/
    );


    m.def("reduce", [](const normalized_function<decision_function<radial_basis_kernel<matrix<double,0,1>>>>& df,
            const std::vector<matrix<double,0,1>>& x,
            long num_bv,
            double eps)
        {
            auto out = df;
            // null_trainer doesn't use y so we can leave it empty.
            std::vector<double> y;
            out.function = reduced2(null_trainer(df.function),num_bv,eps).train(x,y);
            return out;
        }, py::arg("df"), py::arg("x"), py::arg("num_basis_vectors"), py::arg("eps")=1e-3
        );

    m.def("reduce", [](const normalized_function<decision_function<radial_basis_kernel<matrix<double,0,1>>>>& df,
            const numpy_image<double>& x_,
            long num_bv,
            double eps)
        {
            std::vector<matrix<double,0,1>> x;
            np_to_cpp(x_, x);
            // null_trainer doesn't use y so we can leave it empty.
            std::vector<double> y;
            auto out = df;
            out.function = reduced2(null_trainer(df.function),num_bv,eps).train(x,y);
            return out;
        }, py::arg("df"), py::arg("x"), py::arg("num_basis_vectors"), py::arg("eps")=1e-3,
"requires \n\
    - eps > 0 \n\
    - num_bv > 0 \n\
ensures \n\
    - This routine takes a learned radial basis function and tries to find a \n\
      new RBF function with num_basis_vectors basis vectors that approximates \n\
      the given df() as closely as possible.  In particular, it finds a \n\
      function new_df() such that new_df(x[i])==df(x[i]) as often as possible. \n\
    - This is accomplished using a reduced set method that begins by using a \n\
      projection, in kernel space, onto a random set of num_basis_vectors \n\
      vectors in x.  Then, L-BFGS is used to further optimize new_df() to match \n\
      df().  The eps parameter controls how long L-BFGS will run, smaller \n\
      values of eps possibly giving better solutions but taking longer to \n\
      execute." 
        /*!
            requires
                - eps > 0
                - num_bv > 0
            ensures
                - This routine takes a learned radial basis function and tries to find a
                  new RBF function with num_basis_vectors basis vectors that approximates
                  the given df() as closely as possible.  In particular, it finds a
                  function new_df() such that new_df(x[i])==df(x[i]) as often as possible.
                - This is accomplished using a reduced set method that begins by using a
                  projection, in kernel space, onto a random set of num_basis_vectors
                  vectors in x.  Then, L-BFGS is used to further optimize new_df() to match
                  df().  The eps parameter controls how long L-BFGS will run, smaller
                  values of eps possibly giving better solutions but taking longer to
                  execute.
        !*/
        );
}

// ----------------------------------------------------------------------------------------

void bind_decision_functions(py::module &m)
{
    add_linear_df<linear_kernel<sample_type> >(m, "_decision_function_linear");
    add_linear_df<sparse_linear_kernel<sparse_vect> >(m, "_decision_function_sparse_linear");

    add_df<histogram_intersection_kernel<sample_type> >(m, "_decision_function_histogram_intersection");
    add_df<sparse_histogram_intersection_kernel<sparse_vect> >(m, "_decision_function_sparse_histogram_intersection");

    add_df<polynomial_kernel<sample_type> >(m, "_decision_function_polynomial");
    add_df<sparse_polynomial_kernel<sparse_vect> >(m, "_decision_function_sparse_polynomial");


    py::class_<radial_basis_kernel<sample_type>>(m, "_radial_basis_kernel")
        .def("__repr__", radial_basis_kernel__repr__)
        .def_property_readonly("gamma", [](const radial_basis_kernel<sample_type>& k){return k.gamma; });

    py::class_<linear_kernel<sample_type>>(m, "_linear_kernel")
        .def("__repr__", linear_kernel__repr__);

    add_df<radial_basis_kernel<sample_type> >(m, "_decision_function_radial_basis");
    add_df<sparse_radial_basis_kernel<sparse_vect> >(m, "_decision_function_sparse_radial_basis");
    add_normalized_df<radial_basis_kernel<sample_type>>(m, "_normalized_decision_function_radial_basis");

    setup_auto_train_rbf_classifier(m); 


    add_df<sigmoid_kernel<sample_type> >(m, "_decision_function_sigmoid");
    add_df<sparse_sigmoid_kernel<sparse_vect> >(m, "_decision_function_sparse_sigmoid");

    m.def("test_binary_decision_function", _normalized_test_binary_decision_function<radial_basis_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));
    m.def("test_binary_decision_function", _normalized_test_binary_decision_function_np<radial_basis_kernel<sample_type> >,
        py::arg("function"), py::arg("samples"), py::arg("labels"));

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



