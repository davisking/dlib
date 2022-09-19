// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/global_optimization.h>
#include <dlib/matrix.h>
#include <pybind11/stl.h>


using namespace dlib;
using namespace std;
namespace py = pybind11;

// ----------------------------------------------------------------------------------------

std::vector<bool> list_to_bool_vector(
    const py::list& l
)
{
    std::vector<bool> result(len(l));
    for (long i = 0; i < result.size(); ++i)
    {
        result[i] = l[i].cast<bool>();
    }
    return result;
}

matrix<double,0,1> list_to_mat(
    const py::list& l
)
{
    matrix<double,0,1> result(len(l));
    for (long i = 0; i < result.size(); ++i)
        result(i) = l[i].cast<double>();
    return result;
}

py::list mat_to_list (
    const matrix<double,0,1>& m
)
{
    py::list l;
    for (long i = 0; i < m.size(); ++i)
        l.append(m(i));
    return l;
}

size_t num_function_arguments(py::object f, size_t expected_num)
{
    const auto code_object = f.attr(hasattr(f,"func_code") ? "func_code" : "__code__");
    const auto num = code_object.attr("co_argcount").cast<std::size_t>();
    if (num < expected_num && (code_object.attr("co_flags").cast<int>() & CO_VARARGS))
        return expected_num;
    return num;
}

double call_func(py::object f, const matrix<double,0,1>& args)
{
    const auto num = num_function_arguments(f, args.size());
    DLIB_CASSERT(num == args.size(), 
        "The function being optimized takes a number of arguments that doesn't agree with the size of the bounds lists you provided to find_max_global()");
    DLIB_CASSERT(0 < num && num <= 35, "Functions being optimized must take between 1 and 35 scalar arguments.");

#define CALL_WITH_N_ARGS(N) case N: return dlib::gopt_impl::_cwv(f,args,std::make_index_sequence<N>{}).cast<double>(); 
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
        CALL_WITH_N_ARGS(16)
        CALL_WITH_N_ARGS(17)
        CALL_WITH_N_ARGS(18)
        CALL_WITH_N_ARGS(19)
        CALL_WITH_N_ARGS(20)
        CALL_WITH_N_ARGS(21)
        CALL_WITH_N_ARGS(22)
        CALL_WITH_N_ARGS(23)
        CALL_WITH_N_ARGS(24)
        CALL_WITH_N_ARGS(25)
        CALL_WITH_N_ARGS(26)
        CALL_WITH_N_ARGS(27)
        CALL_WITH_N_ARGS(28)
        CALL_WITH_N_ARGS(29)
        CALL_WITH_N_ARGS(30)
        CALL_WITH_N_ARGS(31)
        CALL_WITH_N_ARGS(32)
        CALL_WITH_N_ARGS(33)
        CALL_WITH_N_ARGS(34)
        CALL_WITH_N_ARGS(35)

        default:
            DLIB_CASSERT(false, "oops");
            break;
    }
}

// ----------------------------------------------------------------------------------------

py::tuple py_find_max_global (
    py::object f,
    py::list bound1,
    py::list bound2,
    py::list is_integer_variable,
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

    return py::make_tuple(mat_to_list(result.x),result.y);
}

py::tuple py_find_max_global2 (
    py::object f,
    py::list bound1,
    py::list bound2,
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

    return py::make_tuple(mat_to_list(result.x),result.y);
}

// ----------------------------------------------------------------------------------------

py::tuple py_find_min_global (
    py::object f,
    py::list bound1,
    py::list bound2,
    py::list is_integer_variable,
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

    auto result = find_min_global(func, list_to_mat(bound1), list_to_mat(bound2),
        list_to_bool_vector(is_integer_variable), max_function_calls(num_function_calls),
        solver_epsilon);

    return py::make_tuple(mat_to_list(result.x),result.y);
}

py::tuple py_find_min_global2 (
    py::object f,
    py::list bound1,
    py::list bound2,
    unsigned long num_function_calls,
    double solver_epsilon = 0
)
{
    DLIB_CASSERT(len(bound1) == len(bound2));

    auto func = [&](const matrix<double,0,1>& x)
    {
        return call_func(f, x);
    };

    auto result = find_min_global(func, list_to_mat(bound1), list_to_mat(bound2), max_function_calls(num_function_calls), solver_epsilon);

    return py::make_tuple(mat_to_list(result.x),result.y);
}

// ----------------------------------------------------------------------------------------

function_spec py_function_spec1 (
    py::list a,
    py::list b
)
{
    return function_spec(list_to_mat(a), list_to_mat(b));
}

function_spec py_function_spec2 (
    py::list a,
    py::list b,
    py::list c
)
{
    return function_spec(list_to_mat(a), list_to_mat(b), list_to_bool_vector(c));
}

std::shared_ptr<global_function_search> py_global_function_search1 (
    py::list functions
)
{
    std::vector<function_spec> tmp;
    for (const auto& i : functions)
        tmp.emplace_back(i.cast<function_spec>());

    return std::make_shared<global_function_search>(tmp);
}

std::shared_ptr<global_function_search> py_global_function_search2 (
    py::list functions,
    py::list initial_function_evals,
    double relative_noise_magnitude
)
{
    std::vector<function_spec> specs;
    for (const auto& i : functions)
        specs.emplace_back(i.cast<function_spec>());

    std::vector<std::vector<function_evaluation>> func_evals;
    for (const auto& i : initial_function_evals)
    {
        std::vector<function_evaluation> evals;
        for (const auto& j : i)
        {
            evals.emplace_back(j.cast<function_evaluation>());
        }
        func_evals.emplace_back(std::move(evals));
    }

    return std::make_shared<global_function_search>(specs, func_evals, relative_noise_magnitude);
}

function_evaluation py_function_evaluation(
    const py::list& x, 
    double y
)
{
    return function_evaluation(list_to_mat(x), y);
}

// ----------------------------------------------------------------------------------------

void bind_global_optimization(py::module& m)
{


    const char* docstring =
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
        if (is_integer_variable[i]) then x[i] is an integer value (but still \n\
        represented with float type). \n\
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
      optimizer.";
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
                if (is_integer_variable[i]) then x[i] is an integer value (but still
                represented with float type).
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
    m.def("find_max_global", &py_find_max_global, docstring, py::arg("f"),
        py::arg("bound1"), py::arg("bound2"), py::arg("is_integer_variable"),
        py::arg("num_function_calls"), py::arg("solver_epsilon")=0);

    m.def("find_max_global", &py_find_max_global2, 
        "This function simply calls the other version of find_max_global() with is_integer_variable set to False for all variables.", 
        py::arg("f"), py::arg("bound1"), py::arg("bound2"), py::arg("num_function_calls"),
        py::arg("solver_epsilon")=0);



    m.def("find_min_global", &py_find_min_global, 
      "This function is just like find_max_global(), except it performs minimization rather than maximization.", 
        py::arg("f"), py::arg("bound1"), py::arg("bound2"), py::arg("is_integer_variable"),
        py::arg("num_function_calls"), py::arg("solver_epsilon")=0);

    m.def("find_min_global", &py_find_min_global2, 
        "This function simply calls the other version of find_min_global() with is_integer_variable set to False for all variables.", 
        py::arg("f"), py::arg("bound1"), py::arg("bound2"), py::arg("num_function_calls"),
        py::arg("solver_epsilon")=0);

    // -------------------------------------------------
    // -------------------------------------------------


    py::class_<function_evaluation> (m, "function_evaluation",  R"RAW(  
This object records the output of a real valued function in response to
some input. 

In particular, if you have a function F(x) then the function_evaluation is
simply a struct that records x and the scalar value F(x). )RAW")
        .def(py::init<matrix<double,0,1>,double>(), py::arg("x"), py::arg("y"))
        .def(py::init<>(&py_function_evaluation), py::arg("x"), py::arg("y"))
        .def_readonly("x",       &function_evaluation::x)
        .def_readonly("y",       &function_evaluation::y);


    py::class_<function_spec> (m, "function_spec",  "See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html")
        .def(py::init<matrix<double,0,1>,matrix<double,0,1>>(), py::arg("bound1"), py::arg("bound2") )
        .def(py::init<matrix<double,0,1>,matrix<double,0,1>,std::vector<bool>>(), py::arg("bound1"), py::arg("bound2"), py::arg("is_integer") )
        .def(py::init<>(&py_function_spec1), py::arg("bound1"), py::arg("bound2"))
        .def(py::init<>(&py_function_spec2), py::arg("bound1"), py::arg("bound2"), py::arg("is_integer"))
        .def_readonly("lower",       &function_spec::lower)
        .def_readonly("upper",       &function_spec::upper)
        .def_readonly("is_integer_variable",       &function_spec::is_integer_variable);


    py::class_<function_evaluation_request> (m, "function_evaluation_request", "See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html")
        .def_property_readonly("function_idx", &function_evaluation_request::function_idx)
        .def_property_readonly("x", &function_evaluation_request::x)
        .def_property_readonly("has_been_evaluated", &function_evaluation_request::has_been_evaluated)
        .def("set", &function_evaluation_request::set);

    py::class_<global_function_search, std::shared_ptr<global_function_search>> (m, "global_function_search", "See: http://dlib.net/dlib/global_optimization/global_function_search_abstract.h.html")
        .def(py::init<function_spec>(), py::arg("function"))
        .def(py::init<>(&py_global_function_search1), py::arg("functions"))
        .def(py::init<>(&py_global_function_search2), py::arg("functions"), py::arg("initial_function_evals"), py::arg("relative_noise_magnitude"))
        .def("set_seed", &global_function_search::set_seed, py::arg("seed"))
        .def("num_functions", &global_function_search::num_functions)
        .def("get_function_evaluations", [](const global_function_search& self) { 
            std::vector<function_spec> specs;
            std::vector<std::vector<function_evaluation>> function_evals;
            self.get_function_evaluations(specs,function_evals); 
            py::list py_specs, py_func_evals;
            for (const auto& s : specs)
                py_specs.append(s);
            for (const auto& i : function_evals)
            {
                py::list tmp;
                for (const auto& j : i)
                    tmp.append(j);
                py_func_evals.append(tmp);
            }
            return py::make_tuple(py_specs,py_func_evals);})
        .def("get_best_function_eval", [](const global_function_search& self) { 
            matrix<double,0,1> x; double y; size_t idx; self.get_best_function_eval(x,y,idx); return py::make_tuple(x,y,idx);})
        .def("get_next_x", &global_function_search::get_next_x)
        .def("get_pure_random_search_probability", &global_function_search::get_pure_random_search_probability)
        .def("set_pure_random_search_probability", &global_function_search::set_pure_random_search_probability, py::arg("prob"))
        .def("get_solver_epsilon", &global_function_search::get_solver_epsilon)
        .def("set_solver_epsilon", &global_function_search::set_solver_epsilon, py::arg("eps"))
        .def("get_relative_noise_magnitude", &global_function_search::get_relative_noise_magnitude)
        .def("set_relative_noise_magnitude", &global_function_search::set_relative_noise_magnitude, py::arg("value"))
        .def("get_monte_carlo_upper_bound_sample_num", &global_function_search::get_monte_carlo_upper_bound_sample_num)
        .def("set_monte_carlo_upper_bound_sample_num", &global_function_search::set_monte_carlo_upper_bound_sample_num, py::arg("num"))
        ;

}

