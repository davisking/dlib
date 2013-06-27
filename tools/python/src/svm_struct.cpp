// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <boost/python.hpp>
#include <dlib/matrix.h>
#include <boost/python/args.hpp>
#include "pyassert.h"
#include "boost_python_utils.h"
#include <dlib/svm.h>


using namespace dlib;
using namespace std;
using namespace boost::python;

class svm_struct_dense : public structural_svm_problem<matrix<double,0,1> >
{
public:
    svm_struct_dense (
        object& problem_,
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
        problem.attr("get_truth_joint_feature_vector")(idx,boost::ref(psi));
    }

    virtual void separation_oracle (
        const long idx,
        const matrix_type& current_solution,
        scalar_type& loss,
        feature_vector_type& psi
    ) const 
    {
        loss = extract<double>(problem.attr("separation_oracle")(idx,boost::ref(current_solution),boost::ref(psi)));
    }

private:

    const long num_dimensions;
    const long num_samples;
    object& problem;
};

// ----------------------------------------------------------------------------------------

/*
class svm_struct_sparse : public structural_svm_problem<matrix<double,0,1>, 
                                    std::vector<std::pair<unsigned long,double> >
{
};
*/

// ----------------------------------------------------------------------------------------

matrix<double,0,1> solve_structural_svm_problem(
    object problem
)
{
    const double C = extract<double>(problem.attr("C"));
    const bool be_verbose = hasattr(problem,"be_verbose") && extract<bool>(problem.attr("be_verbose"));
    const bool use_sparse_feature_vectors = hasattr(problem,"use_sparse_feature_vectors") && 
                                            extract<bool>(problem.attr("use_sparse_feature_vectors"));

    double eps = 0.001;
    unsigned long max_cache_size = 10;
    if (hasattr(problem, "epsilon"))
        eps = extract<double>(problem.attr("epsilon"));
    if (hasattr(problem, "max_cache_size"))
        eps = extract<double>(problem.attr("max_cache_size"));

    const long num_samples = extract<long>(problem.attr("num_samples"));
    const long num_dimensions = extract<long>(problem.attr("num_dimensions"));

    if (be_verbose)
    {
        cout << "C:              " << C << endl;
        cout << "epsilon:        " << eps << endl;
        cout << "max_cache_size: " << max_cache_size << endl;
        cout << "num_samples:    " << num_samples << endl;
        cout << "num_dimensions: " << num_dimensions << endl;
        cout << "use_sparse_feature_vectors: " << std::boolalpha << use_sparse_feature_vectors << endl;
        cout << endl;
    }


    svm_struct_dense prob(problem, num_dimensions, num_samples);
    prob.set_c(C);
    prob.set_epsilon(eps);
    prob.set_max_cache_size(max_cache_size);
    if (be_verbose)
        prob.be_verbose();

    oca solver;
    matrix<double,0,1> w;
    solver(prob, w);
    return w;
}

// ----------------------------------------------------------------------------------------

void bind_svm_struct()
{
    def("solve_structural_svm_problem",solve_structural_svm_problem);
}

// ----------------------------------------------------------------------------------------

