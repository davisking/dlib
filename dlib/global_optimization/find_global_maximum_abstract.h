// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_FiND_GLOBAL_MAXIMUM_ABSTRACT_hH_
#ifdef DLIB_FiND_GLOBAL_MAXIMUM_ABSTRACT_hH_

#include "upper_bound_function_abstract.h"
#include "global_function_search_abstract.h"
#include "../metaprogramming.h"
#include "../matrix.h"
#include <utility>
#include <chrono>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename T
        > 
    auto call_function_and_expand_args(
        T&& f, 
        const matrix<double,0,1>& args
    ) -> decltype(f(args or args expanded out as discussed below));
    /*!
        requires
            - f is a function object with one of the following signatures:
                auto f(matrix<double,0,1>)
                auto f(double)
                auto f(double,double)
                auto f(double,double,double)
                ...
                auto f(double,double,...,double)  // up to 40 double arguments
            - if (f() explicitly expands its arguments) then 
                - args.size() == the number of arguments taken by f.
        ensures
            - This function invokes f() with the given arguments and returns the result.
              However, the signature of f() is allowed to vary.  In particular, if f()
              takes a matrix<double,0,1> as a single argument then this function simply
              calls f(args).  However, if f() takes double arguments then args is expanded
              appropriately, i.e. it calls one of the following as appropriate: 
                f(args(0))
                f(args(0),args(1))
                ...
                f(args(0),args(1),...,args(N))
              and the result of f() is returned.
    !*/

// ----------------------------------------------------------------------------------------

    struct max_function_calls
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a simple typed integer class used to strongly type the "max number
                of function calls" argument to find_global_maximum().

        !*/

        max_function_calls() = default;

        explicit max_function_calls(size_t max_calls) : max_calls(max_calls) {}

        size_t max_calls = std::numeric_limits<size_t>::max();
    };

// ----------------------------------------------------------------------------------------

    const auto FOREVER = std::chrono::hours(24*356*290); // 290 years, basically forever

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    std::pair<size_t,function_evaluation> find_global_maximum (
        std::vector<funct>& functions,
        const std::vector<function_spec>& specs,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 1e-11
    );
    /*!
        requires
            - functions.size() != 0
            - functions.size() == specs.size()
            - solver_epsilon >= 0
            - for all valid i:
                - functions[i] is a real valued multi-variate function object.  Moreover,
                  it must be callable via an expression of the form:
                  call_function_and_expand_args(functions[i], specs.lower).  This means
                  function[i] should have a signature like one of the following:
                    double f(matrix<double,0,1>)
                    double f(double)
                    double f(double,double)
                    etc.
                - The range of inputs defined by specs[i] must be valid inputs to
                  functions[i].
        ensures
            - This function performs global optimization on the set of given functions.
              The goal is to maximize the following objective function:
                 max_{i,x_i}: functions[i](x_i)
                 subject to the constraints on x_i defined by specs[i].
              Once found, the return value of find_global_maximum() is:
                make_pair(i, function_evaluation(x_i,functions[i](x_i))). 
              That is, we search for the settings of i and x that return the largest output
              and return those settings.
            - The search is performed using the global_function_search object.  See its
              documentation for details of the algorithm.
            - We set the global_function_search::get_solver_epsilon() parameter to
              solver_epsilon.  Therefore, the search will only attempt to find a global
              maximizer to at most solver_epsilon accuracy.  Once a local maximizer is
              found to that accuracy the search will focus entirely on finding other maxima
              elsewhere rather than on further improving the current local optima found so
              far.  That is, once a local maxima is identified to about solver_epsilon
              accuracy, the algorithm will spend all its time exploring the functions to
              find other local maxima to investigate.
            - find_global_maximum() runs until one of the following is true:
                - The total number of calls to the provided functions is == num.max_calls
                - More than max_runtime time has elapsed since the start of this function.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 1e-11
    );
    /*!
        requires
            - bound1.size() == bound2.size() == is_integer_variable.size()
            - for all valid i: bound1(i) != bound2(i)
            - solver_epsilon >= 0
            - f() is a real valued multi-variate function object.  Moreover, it must be
              callable via an expression of the form: call_function_and_expand_args(f,
              bound1).  This means f() should have a signature like one of the following:
                double f(matrix<double,0,1>)
                double f(double)
                double f(double,double)
                etc.
            - The range of inputs defined by function_spec(bound1,bound2,is_integer_variable) 
              must be valid inputs to f().
        ensures
            - This function performs global optimization on the given f() function.
              The goal is to maximize the following objective function:
                 f(x)
                 subject to the constraints on x defined by function_spec(bound1,bound2,is_integer_variable).
              Once found, the return value of find_global_maximum() is:
                function_evaluation(x,f(x))). 
              That is, we search for the setting of x that returns the largest output and
              return that setting.
            - The search is performed using the global_function_search object.  See its
              documentation for details of the algorithm.
            - We set the global_function_search::get_solver_epsilon() parameter to
              solver_epsilon.  Therefore, the search will only attempt to find a global
              maximizer to at most solver_epsilon accuracy.  Once a local maximizer is
              found to that accuracy the search will focus entirely on finding other maxima
              elsewhere rather than on further improving the current local optima found so
              far.  That is, once a local maxima is identified to about solver_epsilon
              accuracy, the algorithm will spend all its time exploring the function to
              find other local maxima to investigate.
            - find_global_maximum() runs until one of the following is true:
                - The total number of calls to f() is == num.max_calls
                - More than max_runtime time has elapsed since the start of this function.
    !*/

// ----------------------------------------------------------------------------------------
// The following functions are just convenient overloads for calling the above defined
// find_global_maximum() routines.
// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(std::move(f), bound1, bound2, std::vector<bool>(bound1.size(),false), num, max_runtime, solver_epsilon);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const double bound1,
        const double bound2,
        const max_function_calls num,
        const std::chrono::nanoseconds max_runtime = FOREVER,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(std::move(f), matrix<double,0,1>({bound1}), matrix<double,0,1>({bound2}), num, max_runtime, solver_epsilon);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const double bound1,
        const double bound2,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(std::move(f), bound1, bound2, max_function_calls(), max_runtime, solver_epsilon);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    function_evaluation find_global_maximum (
        funct f,
        const matrix<double,0,1>& bound1,
        const matrix<double,0,1>& bound2,
        const std::vector<bool>& is_integer_variable,
        const std::chrono::nanoseconds max_runtime,
        double solver_epsilon = 1e-11
    ) 
    {
        return find_global_maximum(std::move(f), bound1, bound2, is_integer_variable, max_function_calls(), max_runtime, solver_epsilon);
    }

// ----------------------------------------------------------------------------------------


}

#endif // DLIB_FiND_GLOBAL_MAXIMUM_ABSTRACT_hH_


