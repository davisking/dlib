// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/svm.h>

using namespace dlib;
using namespace std;
namespace py = pybind11;

template <typename psi_type>
class svm_struct_prob : public structural_svm_problem<matrix<double,0,1>, psi_type>
{
    typedef structural_svm_problem<matrix<double,0,1>, psi_type> base;
    typedef typename base::feature_vector_type feature_vector_type;
    typedef typename base::matrix_type matrix_type;
    typedef typename base::scalar_type scalar_type;
public:
    svm_struct_prob (
        py::object& problem_,
        long num_dimensions_,
        long num_samples_
    ) : 
        num_dimensions(num_dimensions_),
        num_samples(num_samples_),
        problem(problem_)
    {} 

    virtual long get_num_dimensions (
    ) const { return num_dimensions; }

    virtual long get_num_samples (
    ) const { return num_samples; }

    virtual void get_truth_joint_feature_vector (
        long idx,
        feature_vector_type& psi 
    ) const 
    {
        psi = problem.attr("get_truth_joint_feature_vector")(idx).template cast<feature_vector_type&>();
    }

    virtual void separation_oracle (
        const long idx,
        const matrix_type& current_solution,
        scalar_type& loss,
        feature_vector_type& psi
    ) const 
    {
        py::object res = problem.attr("separation_oracle")(idx,std::ref(current_solution));
        pyassert(len(res) == 2, "separation_oracle() must return two objects, the loss and the psi vector");
        py::tuple t = res.cast<py::tuple>();
        // let the user supply the output arguments in any order.
        try {
            loss = t[0].cast<scalar_type>();
            psi = t[1].cast<feature_vector_type&>();
        } catch(py::cast_error&) {
            psi = t[0].cast<feature_vector_type&>();
            loss = t[1].cast<scalar_type>();
       }
    }

private:

    const long num_dimensions;
    const long num_samples;
    py::object& problem;
};

// ----------------------------------------------------------------------------------------

template <typename psi_type>
matrix<double,0,1> solve_structural_svm_problem_impl(
    py::object problem
)
{
  const double C = problem.attr("C").cast<double>();
    const bool be_verbose = py::hasattr(problem,"be_verbose") && problem.attr("be_verbose").cast<bool>();
    const bool use_sparse_feature_vectors = py::hasattr(problem,"use_sparse_feature_vectors") && 
                                            problem.attr("use_sparse_feature_vectors").cast<bool>();
    const bool learns_nonnegative_weights = py::hasattr(problem,"learns_nonnegative_weights") && 
                                            problem.attr("learns_nonnegative_weights").cast<bool>();

    double eps = 0.001;
    unsigned long max_cache_size = 10;
    if (py::hasattr(problem, "epsilon"))
        eps = problem.attr("epsilon").cast<double>();
    if (py::hasattr(problem, "max_cache_size"))
        max_cache_size = problem.attr("max_cache_size").cast<double>();

    const long num_samples = problem.attr("num_samples").cast<long>();
    const long num_dimensions = problem.attr("num_dimensions").cast<long>();

    pyassert(num_samples > 0, "You can't train a Structural-SVM if you don't have any training samples.");

    if (be_verbose)
    {
        cout << "C:              " << C << endl;
        cout << "epsilon:        " << eps << endl;
        cout << "max_cache_size: " << max_cache_size << endl;
        cout << "num_samples:    " << num_samples << endl;
        cout << "num_dimensions: " << num_dimensions << endl;
        cout << "use_sparse_feature_vectors: " << std::boolalpha << use_sparse_feature_vectors << endl;
        cout << "learns_nonnegative_weights: " << std::boolalpha << learns_nonnegative_weights << endl;
        cout << endl;
    }

    svm_struct_prob<psi_type> prob(problem, num_dimensions, num_samples);
    prob.set_c(C);
    prob.set_epsilon(eps);
    prob.set_max_cache_size(max_cache_size);
    if (be_verbose)
        prob.be_verbose();

    oca solver;
    matrix<double,0,1> w;
    if (learns_nonnegative_weights)
        solver(prob, w, prob.get_num_dimensions());
    else
        solver(prob, w);
    return w;
}

// ----------------------------------------------------------------------------------------

matrix<double,0,1> solve_structural_svm_problem(
    py::object problem
)
{
    // Check if the python code is using sparse or dense vectors to represent PSI()
    if (py::isinstance<matrix<double,0,1>>(problem.attr("get_truth_joint_feature_vector")(0)))
        return solve_structural_svm_problem_impl<matrix<double,0,1> >(problem);
    else
        return solve_structural_svm_problem_impl<std::vector<std::pair<unsigned long,double> > >(problem);
}

// ----------------------------------------------------------------------------------------

void bind_svm_struct(py::module& m)
{
    m.def("solve_structural_svm_problem",solve_structural_svm_problem, py::arg("problem"),
"This function solves a structural SVM problem and returns the weight vector    \n\
that defines the solution.  See the example program python_examples/svm_struct.py    \n\
for documentation about how to create a proper problem object.   " 
        );
}

// ----------------------------------------------------------------------------------------

