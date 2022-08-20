// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include "testing_results.h"
#include <dlib/matrix.h>
#include <dlib/svm_threaded.h>

using namespace dlib;
using namespace std;

typedef matrix<double,0,1> sample_type; 
typedef std::vector<std::pair<unsigned long,double> > sparse_vect;

template <typename trainer_type>
typename trainer_type::trained_function_type train (
    const trainer_type& trainer,
    const std::vector<typename trainer_type::sample_type>& samples,
    const std::vector<double>& labels
)
{
    pyassert(is_binary_classification_problem(samples,labels), "Invalid inputs");
    return trainer.train(samples, labels);
}

template <typename trainer_type>
void set_epsilon ( trainer_type& trainer, double eps)
{
    pyassert(eps > 0, "epsilon must be > 0");
    trainer.set_epsilon(eps);
}

template <typename trainer_type>
double get_epsilon ( const trainer_type& trainer) { return trainer.get_epsilon(); }


template <typename trainer_type>
void set_cache_size ( trainer_type& trainer, long cache_size)
{
    pyassert(cache_size > 0, "cache size must be > 0");
    trainer.set_cache_size(cache_size);
}

template <typename trainer_type>
long get_cache_size ( const trainer_type& trainer) { return trainer.get_cache_size(); }


template <typename trainer_type>
void set_c ( trainer_type& trainer, double C)
{
    pyassert(C > 0, "C must be > 0");
    trainer.set_c(C);
}

template <typename trainer_type>
void set_c_class1 ( trainer_type& trainer, double C)
{
    pyassert(C > 0, "C must be > 0");
    trainer.set_c_class1(C);
}

template <typename trainer_type>
void set_c_class2 ( trainer_type& trainer, double C)
{
    pyassert(C > 0, "C must be > 0");
    trainer.set_c_class2(C);
}

template <typename trainer_type>
double get_c_class1 ( const trainer_type& trainer) { return trainer.get_c_class1(); }
template <typename trainer_type>
double get_c_class2 ( const trainer_type& trainer) { return trainer.get_c_class2(); }

template <typename trainer_type>
py::class_<trainer_type> setup_trainer_eps (
    py::module& m,
    const std::string& name
)
{
    return py::class_<trainer_type>(m, name.c_str())
        .def("train", train<trainer_type>)
        .def_property("epsilon", get_epsilon<trainer_type>, set_epsilon<trainer_type>);
}

template <typename trainer_type>
py::class_<trainer_type> setup_trainer_eps_c (
    py::module& m,
    const std::string& name
)
{
    return setup_trainer_eps<trainer_type>(m, name)
        .def("set_c", set_c<trainer_type>)
        .def_property("c_class1", get_c_class1<trainer_type>, set_c_class1<trainer_type>)
        .def_property("c_class2", get_c_class2<trainer_type>, set_c_class2<trainer_type>);
}

template <typename trainer_type>
py::class_<trainer_type> setup_trainer_eps_c_cache (
    py::module& m,
    const std::string& name
)
{
    return setup_trainer_eps_c<trainer_type>(m, name)
        .def_property("cache_size", get_cache_size<trainer_type>, set_cache_size<trainer_type>);
}

template <typename trainer_type>
void set_gamma (
    trainer_type& trainer,
    double gamma
)
{
    pyassert(gamma > 0, "gamma must be > 0");
    trainer.set_kernel(typename trainer_type::kernel_type(gamma));
}

template <typename trainer_type>
double get_gamma (
    const trainer_type& trainer
)
{
    return trainer.get_kernel().gamma;
}

// ----------------------------------------------------------------------------------------

template <
    typename trainer_type
    >
const binary_test _cross_validate_trainer (
    const trainer_type& trainer,
    const std::vector<typename trainer_type::sample_type>& x,
    const std::vector<double>& y,
    const unsigned long folds
)
{
    pyassert(is_binary_classification_problem(x,y), "Training data does not make a valid training set.");
    pyassert(1 < folds && folds <= x.size(), "Invalid number of folds given.");
    return cross_validate_trainer(trainer, x, y, folds);
}

template <
    typename trainer_type
    >
const binary_test _cross_validate_trainer_t (
    const trainer_type& trainer,
    const std::vector<typename trainer_type::sample_type>& x,
    const std::vector<double>& y,
    const unsigned long folds,
    const unsigned long num_threads
)
{
    pyassert(is_binary_classification_problem(x,y), "Training data does not make a valid training set.");
    pyassert(1 < folds && folds <= x.size(), "Invalid number of folds given.");
    pyassert(1 < num_threads, "The number of threads specified must not be zero.");
    return cross_validate_trainer_threaded(trainer, x, y, folds, num_threads);
}

// ----------------------------------------------------------------------------------------

void bind_svm_c_trainer(py::module& m)
{
    namespace py = pybind11;

    // svm_c
    {
        typedef svm_c_trainer<radial_basis_kernel<sample_type> > T;
        setup_trainer_eps_c_cache<T>(m, "svm_c_trainer_radial_basis")
            .def(py::init())
            .def_property("gamma", get_gamma<T>, set_gamma<T>);
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef svm_c_trainer<sparse_radial_basis_kernel<sparse_vect> > T;
        setup_trainer_eps_c_cache<T>(m, "svm_c_trainer_sparse_radial_basis")
            .def(py::init())
            .def_property("gamma", get_gamma<T>, set_gamma<T>);
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef svm_c_trainer<histogram_intersection_kernel<sample_type> > T;
        setup_trainer_eps_c_cache<T>(m, "svm_c_trainer_histogram_intersection")
            .def(py::init());
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef svm_c_trainer<sparse_histogram_intersection_kernel<sparse_vect> > T;
        setup_trainer_eps_c_cache<T>(m, "svm_c_trainer_sparse_histogram_intersection")
            .def(py::init());
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    // svm_c_linear
    {
        typedef svm_c_linear_trainer<linear_kernel<sample_type> > T;
        setup_trainer_eps_c<T>(m, "svm_c_trainer_linear")
            .def(py::init())
            .def_property("max_iterations", &T::get_max_iterations, &T::set_max_iterations)
            .def_property("force_last_weight_to_1", &T::forces_last_weight_to_1, &T::force_last_weight_to_1)
            .def_property("learns_nonnegative_weights", &T::learns_nonnegative_weights, &T::set_learns_nonnegative_weights)
            .def_property_readonly("has_prior", &T::has_prior)
            .def("set_prior", &T::set_prior)
            .def("be_verbose", &T::be_verbose)
            .def("be_quiet", &T::be_quiet);

        m.def("cross_validate_trainer", _cross_validate_trainer<T>,
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>,
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef svm_c_linear_trainer<sparse_linear_kernel<sparse_vect> > T;
        setup_trainer_eps_c<T>(m, "svm_c_trainer_sparse_linear")
            .def(py::init())
            .def_property("max_iterations", &T::get_max_iterations, &T::set_max_iterations)
            .def_property("force_last_weight_to_1", &T::forces_last_weight_to_1, &T::force_last_weight_to_1)
            .def_property("learns_nonnegative_weights", &T::learns_nonnegative_weights, &T::set_learns_nonnegative_weights)
            .def_property_readonly("has_prior", &T::has_prior)
            .def("set_prior", &T::set_prior)
            .def("be_verbose", &T::be_verbose)
            .def("be_quiet", &T::be_quiet);

        m.def("cross_validate_trainer", _cross_validate_trainer<T>,
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>,
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    // rvm
    {
        typedef rvm_trainer<radial_basis_kernel<sample_type> > T;
        setup_trainer_eps<T>(m, "rvm_trainer_radial_basis")
            .def(py::init())
            .def_property("gamma", get_gamma<T>, set_gamma<T>);
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef rvm_trainer<sparse_radial_basis_kernel<sparse_vect> > T;
        setup_trainer_eps<T>(m, "rvm_trainer_sparse_radial_basis")
            .def(py::init())
            .def_property("gamma", get_gamma<T>, set_gamma<T>);
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef rvm_trainer<histogram_intersection_kernel<sample_type> > T;
        setup_trainer_eps<T>(m, "rvm_trainer_histogram_intersection")
            .def(py::init());
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef rvm_trainer<sparse_histogram_intersection_kernel<sparse_vect> > T;
        setup_trainer_eps<T>(m, "rvm_trainer_sparse_histogram_intersection")
            .def(py::init());
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    // rvm linear
    {
        typedef rvm_trainer<linear_kernel<sample_type> > T;
        setup_trainer_eps<T>(m, "rvm_trainer_linear")
            .def(py::init());
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }

    {
        typedef rvm_trainer<sparse_linear_kernel<sparse_vect> > T;
        setup_trainer_eps<T>(m, "rvm_trainer_sparse_linear")
            .def(py::init());
        m.def("cross_validate_trainer", _cross_validate_trainer<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"));
        m.def("cross_validate_trainer_threaded", _cross_validate_trainer_t<T>, 
            py::arg("trainer"),py::arg("x"),py::arg("y"),py::arg("folds"),py::arg("num_threads"));
    }
}


