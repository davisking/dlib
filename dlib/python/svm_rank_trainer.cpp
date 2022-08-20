// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/svm.h>
#include "testing_results.h"
#include <pybind11/stl_bind.h>

using namespace dlib;
using namespace std;
namespace py = pybind11;

typedef matrix<double,0,1> sample_type; 


// ----------------------------------------------------------------------------------------

namespace dlib
{
    template <typename T>
    bool operator== (
        const ranking_pair<T>&, 
        const ranking_pair<T>& 
    )
    {
        pyassert(false, "It is illegal to compare ranking pair objects for equality.");
        return false;
    }
}

template <typename T>
void resize(T& v, unsigned long n) { v.resize(n); }

// ----------------------------------------------------------------------------------------

template <typename trainer_type>
typename trainer_type::trained_function_type train1 (
    const trainer_type& trainer,
    const ranking_pair<typename trainer_type::sample_type>& sample
)
{
    typedef ranking_pair<typename trainer_type::sample_type> st;
    pyassert(is_ranking_problem(std::vector<st>(1, sample)), "Invalid inputs");
    return trainer.train(sample);
}

template <typename trainer_type>
typename trainer_type::trained_function_type train2 (
    const trainer_type& trainer,
    const std::vector<ranking_pair<typename trainer_type::sample_type> >& samples
)
{
    pyassert(is_ranking_problem(samples), "Invalid inputs");
    return trainer.train(samples);
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
void set_c ( trainer_type& trainer, double C)
{
    pyassert(C > 0, "C must be > 0");
    trainer.set_c(C);
}

template <typename trainer_type>
double get_c (const trainer_type& trainer)
{
    return trainer.get_c();
}


template <typename trainer>
void add_ranker (
    py::module& m,
    const char* name
)
{
    py::class_<trainer>(m, name)
        .def(py::init())
        .def_property("epsilon", get_epsilon<trainer>, set_epsilon<trainer>)
        .def_property("c", get_c<trainer>, set_c<trainer>)
        .def_property("max_iterations", &trainer::get_max_iterations, &trainer::set_max_iterations)
        .def_property("force_last_weight_to_1", &trainer::forces_last_weight_to_1, &trainer::force_last_weight_to_1)
        .def_property("learns_nonnegative_weights", &trainer::learns_nonnegative_weights, &trainer::set_learns_nonnegative_weights)
        .def_property_readonly("has_prior", &trainer::has_prior)
        .def("train", train1<trainer>)
        .def("train", train2<trainer>)
        .def("set_prior", &trainer::set_prior)
        .def("be_verbose", &trainer::be_verbose)
        .def("be_quiet", &trainer::be_quiet);
}

// ----------------------------------------------------------------------------------------

template <
    typename trainer_type,
    typename T
    >
const ranking_test _cross_ranking_validate_trainer (
    const trainer_type& trainer,
    const std::vector<ranking_pair<T> >& samples,
    const unsigned long folds
)
{
    pyassert(is_ranking_problem(samples), "Training data does not make a valid training set.");
    pyassert(1 < folds && folds <= samples.size(), "Invalid number of folds given.");
    return cross_validate_ranking_trainer(trainer, samples, folds);
}

// ----------------------------------------------------------------------------------------

void bind_svm_rank_trainer(py::module& m)
{
    py::class_<ranking_pair<sample_type> >(m, "ranking_pair")
        .def(py::init())
        .def_readwrite("relevant", &ranking_pair<sample_type>::relevant)
        .def_readwrite("nonrelevant", &ranking_pair<sample_type>::nonrelevant)
        .def(py::pickle(&getstate<ranking_pair<sample_type>>, &setstate<ranking_pair<sample_type>>));

    py::class_<ranking_pair<sparse_vect> >(m, "sparse_ranking_pair")
        .def(py::init())
        .def_readwrite("relevant", &ranking_pair<sparse_vect>::relevant)
        .def_readwrite("nonrelevant", &ranking_pair<sparse_vect>::nonrelevant)
        .def(py::pickle(&getstate<ranking_pair<sparse_vect>>, &setstate<ranking_pair<sparse_vect>>));

    py::bind_vector<ranking_pairs>(m, "ranking_pairs")
        .def("clear", &ranking_pairs::clear)
        .def("resize", resize<ranking_pairs>)
        .def("extend", extend_vector_with_python_list<ranking_pair<sample_type>>)
        .def(py::pickle(&getstate<ranking_pairs>, &setstate<ranking_pairs>));

    py::bind_vector<sparse_ranking_pairs>(m, "sparse_ranking_pairs")
        .def("clear", &sparse_ranking_pairs::clear)
        .def("resize", resize<sparse_ranking_pairs>)
        .def("extend", extend_vector_with_python_list<ranking_pair<sparse_vect>>)
        .def(py::pickle(&getstate<sparse_ranking_pairs>, &setstate<sparse_ranking_pairs>));

    add_ranker<svm_rank_trainer<linear_kernel<sample_type> > >(m, "svm_rank_trainer");
    add_ranker<svm_rank_trainer<sparse_linear_kernel<sparse_vect> > >(m, "svm_rank_trainer_sparse");

    m.def("cross_validate_ranking_trainer", &_cross_ranking_validate_trainer<
                svm_rank_trainer<linear_kernel<sample_type> >,sample_type>,
                py::arg("trainer"), py::arg("samples"), py::arg("folds") );
    m.def("cross_validate_ranking_trainer", &_cross_ranking_validate_trainer<
                svm_rank_trainer<sparse_linear_kernel<sparse_vect> > ,sparse_vect>,
                py::arg("trainer"), py::arg("samples"), py::arg("folds") );
}



