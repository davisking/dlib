// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/global_optimization.h>
#include <dlib/matrix.h>


using namespace dlib;
using namespace std;
using namespace boost::python;


// ----------------------------------------------------------------------------------------

std::vector<bool> list_to_bool_vector(
    const boost::python::list& l
)
{
    std::vector<bool> result(len(l));
    for (long i = 0; i < result.size(); ++i)
    {
        result[i] = extract<bool>(l[i]);
        cout << "bool val: " << result[i] << endl;
    }
    return result;
}

matrix<double,0,1> list_to_mat(
    const boost::python::list& l
)
{
    matrix<double,0,1> result(len(l));
    for (long i = 0; i < result.size(); ++i)
        result(i) = extract<double>(l[i]);
    return result;
}

boost::python::list mat_to_list (
    const matrix<double,0,1>& m
)
{
    boost::python::list l;
    for (long i = 0; i < m.size(); ++i)
        l.append(m(i));
    return l;
}

size_t num_function_arguments(object f)
{
    return boost::python::extract<std::size_t>(f.attr("func_code").attr("co_argcount"));
}

double call_func(object f, const matrix<double,0,1>& args)
{
    const auto num = num_function_arguments(f);
    DLIB_CASSERT(num == args.size(), 
        "The function being optimized takes a number of arguments that doesn't agree with the size of the bounds lists you provided to find_max_global()");
    DLIB_CASSERT(0 < num && num < 15, "Functions being optimized must take between 1 and 15 scalar arguments.");

#define CALL_WITH_N_ARGS(N) case N: return extract<double>(dlib::gopt_impl::_cwv(f,args,typename make_compile_time_integer_range<N>::type())); 
    switch (num)
    {
        CALL_WITH_N_ARGS(1)
        CALL_WITH_N_ARGS(2)
        CALL_WITH_N_ARGS(3)
        CALL_WITH_N_ARGS(4)
        CALL_WITH_N_ARGS(5)
        CALL_WITH_N_ARGS(6)
        CALL_WITH_N_ARGS(7)
        CALL_WITH_N_ARGS(8)
        CALL_WITH_N_ARGS(9)
        CALL_WITH_N_ARGS(10)
        CALL_WITH_N_ARGS(11)
        CALL_WITH_N_ARGS(12)
        CALL_WITH_N_ARGS(13)
        CALL_WITH_N_ARGS(14)
        CALL_WITH_N_ARGS(15)

        default:
            DLIB_CASSERT(false, "oops");
            break;
    }
}

// ----------------------------------------------------------------------------------------

boost::python::tuple py_find_max_global (
    object f,
    boost::python::list bound1,
    boost::python::list bound2,
    boost::python::list is_integer_variable,
    unsigned long num_function_calls,
    double solver_epsilon = 0
)
{
    DLIB_CASSERT(len(bound1) == len(bound2));
    DLIB_CASSERT(len(bound1) == len(is_integer_variable));

    auto func = [&](const matrix<double,0,1>& x)
    {
        return call_func(f, x);
    };

    auto result = find_max_global(func, list_to_mat(bound1), list_to_mat(bound2),
        list_to_bool_vector(is_integer_variable), max_function_calls(num_function_calls),
        solver_epsilon);

    return boost::python::make_tuple(mat_to_list(result.x),result.y);
}

boost::python::tuple py_find_max_global2 (
    object f,
    boost::python::list bound1,
    boost::python::list bound2,
    unsigned long num_function_calls,
    double solver_epsilon = 0
)
{
    DLIB_CASSERT(len(bound1) == len(bound2));

    auto func = [&](const matrix<double,0,1>& x)
    {
        return call_func(f, x);
    };

    auto result = find_max_global(func, list_to_mat(bound1), list_to_mat(bound2), max_function_calls(num_function_calls), solver_epsilon);

    return boost::python::make_tuple(mat_to_list(result.x),result.y);
}

// ----------------------------------------------------------------------------------------

void bind_global_optimization()
{
    /*!
        requires
            - len(bound1) == len(bound2) == len(is_integer_variable)
            - for all valid i: bound1[i] != bound2[i]
            - solver_epsilon >= 0
            - f() is a real valued multi-variate function.  It must take scalar real
              numbers as its arguments and the number of arguments must be len(bound1).
        ensures
            - This function performs global optimization on the given f() function.
              The goal is to maximize the following objective function:
                 f(x)
              subject to the constraints:
                min(bound1[i],bound2[i]) <= x[i] <= max(bound1[i],bound2[i])
                if (is_integer_variable[i]) then x[i] is an integer.
            - find_max_global() runs until it has called f() num_function_calls times.
              Then it returns the best x it has found along with the corresponding output
              of f().  That is, it returns (best_x_seen,f(best_x_seen)).  Here best_x_seen
              is a list containing the best arguments to f() this function has found.
            - find_max_global() uses a global optimization method based on a combination of
              non-parametric global function modeling and quadratic trust region modeling
              to efficiently find a global maximizer.  It usually does a good job with a
              relatively small number of calls to f().  For more information on how it
              works read the documentation for dlib's global_function_search object.
              However, one notable element is the solver epsilon, which you can adjust.

              The search procedure will only attempt to find a global maximizer to at most
              solver_epsilon accuracy.  Once a local maximizer is found to that accuracy
              the search will focus entirely on finding other maxima elsewhere rather than
              on further improving the current local optima found so far.  That is, once a
              local maxima is identified to about solver_epsilon accuracy, the algorithm
              will spend all its time exploring the function to find other local maxima to
              investigate.  An epsilon of 0 means it will keep solving until it reaches
              full floating point precision.  Larger values will cause it to switch to pure
              global exploration sooner and therefore might be more effective if your
              objective function has many local maxima and you don't care about a super
              high precision solution.
            - Any variables that satisfy the following conditions are optimized on a log-scale:
                - The lower bound on the variable is > 0
                - The ratio of the upper bound to lower bound is > 1000
                - The variable is not an integer variable
              We do this because it's common to optimize machine learning models that have
              parameters with bounds in a range such as [1e-5 to 1e10] (e.g. the SVM C
              parameter) and it's much more appropriate to optimize these kinds of
              variables on a log scale.  So we transform them by applying log() to
              them and then undo the transform via exp() before invoking the function
              being optimized.  Therefore, this transformation is invisible to the user
              supplied functions.  In most cases, it improves the efficiency of the
              optimizer.
    !*/
    using boost::python::arg;
    {
    def("find_max_global", &py_find_max_global, 
"requires \n\
    - len(bound1) == len(bound2) == len(is_integer_variable) \n\
    - for all valid i: bound1[i] != bound2[i] \n\
    - solver_epsilon >= 0 \n\
    - f() is a real valued multi-variate function.  It must take scalar real \n\
      numbers as its arguments and the number of arguments must be len(bound1). \n\
ensures \n\
    - This function performs global optimization on the given f() function. \n\
      The goal is to maximize the following objective function: \n\
         f(x) \n\
      subject to the constraints: \n\
        min(bound1[i],bound2[i]) <= x[i] <= max(bound1[i],bound2[i]) \n\
        if (is_integer_variable[i]) then x[i] is an integer. \n\
    - find_max_global() runs until it has called f() num_function_calls times. \n\
      Then it returns the best x it has found along with the corresponding output \n\
      of f().  That is, it returns (best_x_seen,f(best_x_seen)).  Here best_x_seen \n\
      is a list containing the best arguments to f() this function has found. \n\
    - find_max_global() uses a global optimization method based on a combination of \n\
      non-parametric global function modeling and quadratic trust region modeling \n\
      to efficiently find a global maximizer.  It usually does a good job with a \n\
      relatively small number of calls to f().  For more information on how it \n\
      works read the documentation for dlib's global_function_search object. \n\
      However, one notable element is the solver epsilon, which you can adjust. \n\
 \n\
      The search procedure will only attempt to find a global maximizer to at most \n\
      solver_epsilon accuracy.  Once a local maximizer is found to that accuracy \n\
      the search will focus entirely on finding other maxima elsewhere rather than \n\
      on further improving the current local optima found so far.  That is, once a \n\
      local maxima is identified to about solver_epsilon accuracy, the algorithm \n\
      will spend all its time exploring the function to find other local maxima to \n\
      investigate.  An epsilon of 0 means it will keep solving until it reaches \n\
      full floating point precision.  Larger values will cause it to switch to pure \n\
      global exploration sooner and therefore might be more effective if your \n\
      objective function has many local maxima and you don't care about a super \n\
      high precision solution. \n\
    - Any variables that satisfy the following conditions are optimized on a log-scale: \n\
        - The lower bound on the variable is > 0 \n\
        - The ratio of the upper bound to lower bound is > 1000 \n\
        - The variable is not an integer variable \n\
      We do this because it's common to optimize machine learning models that have \n\
      parameters with bounds in a range such as [1e-5 to 1e10] (e.g. the SVM C \n\
      parameter) and it's much more appropriate to optimize these kinds of \n\
      variables on a log scale.  So we transform them by applying log() to \n\
      them and then undo the transform via exp() before invoking the function \n\
      being optimized.  Therefore, this transformation is invisible to the user \n\
      supplied functions.  In most cases, it improves the efficiency of the \n\
      optimizer." 
        , 
	(arg("f"), arg("bound1"), arg("bound2"), arg("is_integer_variable"), arg("num_function_calls"), arg("solver_epsilon")=0)
    );
    }

    {
    def("find_max_global", &py_find_max_global2, 
        "This function simply calls the other version of find_max_global() with is_integer_variable set to False for all variables.", 
	(arg("f"), arg("bound1"), arg("bound2"), arg("num_function_calls"), arg("solver_epsilon")=0)
    );
    }
}

