// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OPTIMIZATIOn_ABSTRACT_
#ifdef DLIB_OPTIMIZATIOn_ABSTRACT_

#include <cmath>
#include <limits>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"
#include "optimization_search_strategies_abstract.h"
#include "optimization_stop_strategies_abstract.h"
#include "optimization_line_search_abstract.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                    Functions that transform other functions  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    class central_differences;
    /*!
        This is a function object that represents the derivative of some other
        function. 

        Note that if funct is a function of a double then the derivative of 
        funct is just a double but if funct is a function of a dlib::matrix (i.e. a
        function of many variables) then its derivative is a gradient vector (a column
        vector in particular).
    !*/

    template <
        typename funct
        >
    const central_differences<funct> derivative(
        const funct& f, 
        double eps
    );
    /*!
        requires
            - f == a function that returns a scalar
            - f must take either double or a dlib::matrix that is a column vector
            - eps > 0
        ensures
            - returns a function that represents the derivative of the function f.  It
              is approximated numerically by:
                  (f(x+eps)-f(x-eps))/(2*eps)
    !*/

    template <
        typename funct
        >
    const central_differences<funct> derivative(
        const funct& f
    );
    /*!
        ensures
            - returns derivative(f, 1e-7)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename funct
        >
    class negate_function_object;
    /*!
        This is a function object that represents the negative of some other
        function.   
    !*/

    template <
        typename funct
        >
    const negate_function_object<funct> negate_function(
        const funct& f
    );
    /*!
        requires
            - f == a function that returns a scalar
        ensures
            - returns a function that represents the negation of f.  That is,
              the returned function object represents g(x) == -f(x)
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                    Functions that perform unconstrained optimization 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename search_strategy_type,
        typename stop_strategy_type,
        typename funct, 
        typename funct_der, 
        typename T
        >
    void find_min (
        search_strategy_type search_strategy,
        stop_strategy_type stop_strategy,
        const funct& f, 
        const funct_der& der, 
        T& x, 
        double min_f
    );
    /*!
        requires
            - search_strategy == an object that defines a search strategy such as one 
              of the objects from dlib/optimization/optimization_search_strategies_abstract.h
            - stop_strategy == an object that defines a stop strategy such as one of 
              the objects from dlib/optimization/optimization_stop_strategies_abstract.h
            - f(x) must be a valid expression that evaluates to a double
            - der(x) must be a valid expression that evaluates to the derivative of f() at x.
            - is_col_vector(x) == true
        ensures
            - Performs an unconstrained minimization of the function f() using the given
              search_strategy.  
            - The function is optimized until stop_strategy decides that an acceptable 
              point has been found or f(#x) < min_f.
            - #x == the value of x that was found to minimize f()
            - When this function makes calls to f() and der() it always does so by
              first calling f() and then calling der().  That is, these two functions
              are always called in pairs with f() being called first and then der()
              being called second.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename search_strategy_type,
        typename stop_strategy_type,
        typename funct, 
        typename funct_der, 
        typename T
        >
    void find_max (
        search_strategy_type search_strategy,
        stop_strategy_type stop_strategy,
        const funct& f, 
        const funct_der& der, 
        T& x, 
        double max_f
    );
    /*!
        requires
            - search_strategy == an object that defines a search strategy such as one 
              of the objects from dlib/optimization/optimization_search_strategies_abstract.h
            - stop_strategy == an object that defines a stop strategy such as one of 
              the objects from dlib/optimization/optimization_stop_strategies_abstract.h
            - f(x) must be a valid expression that evaluates to a double
            - der(x) must be a valid expression that evaluates to the derivative of f() at x.
            - is_col_vector(x) == true
        ensures
            - Performs an unconstrained maximization of the function f() using the given
              search_strategy.  
            - The function is optimized until stop_strategy decides that an acceptable 
              point has been found or f(#x) > max_f.
            - #x == the value of x that was found to maximize f()
            - When this function makes calls to f() and der() it always does so by
              first calling f() and then calling der().  That is, these two functions
              are always called in pairs with f() being called first and then der()
              being called second.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename search_strategy_type,
        typename stop_strategy_type,
        typename funct,
        typename T
        >
    void find_min_using_approximate_derivatives (
        search_strategy_type search_strategy,
        stop_strategy_type stop_strategy,
        const funct& f,
        T& x,
        double min_f,
        double derivative_eps = 1e-7
    );
    /*!
        requires
            - search_strategy == an object that defines a search strategy such as one 
              of the objects from dlib/optimization/optimization_search_strategies_abstract.h
            - stop_strategy == an object that defines a stop strategy such as one of 
              the objects from dlib/optimization/optimization_stop_strategies_abstract.h
            - f(x) must be a valid expression that evaluates to a double
            - is_col_vector(x) == true
            - derivative_eps > 0 
        ensures
            - Performs an unconstrained minimization of the function f() using the given
              search_strategy.  
            - The function is optimized until stop_strategy decides that an acceptable 
              point has been found or f(#x) < min_f.
            - #x == the value of x that was found to minimize f()
            - Uses the dlib::derivative(f,derivative_eps) function to compute gradient
              information.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename search_strategy_type,
        typename stop_strategy_type,
        typename funct,
        typename T
        >
    void find_max_using_approximate_derivatives (
        search_strategy_type search_strategy,
        stop_strategy_type stop_strategy,
        const funct& f,
        T& x,
        double max_f,
        double derivative_eps = 1e-7
    );
    /*!
        requires
            - search_strategy == an object that defines a search strategy such as one 
              of the objects from dlib/optimization/optimization_search_strategies_abstract.h
            - stop_strategy == an object that defines a stop strategy such as one of 
              the objects from dlib/optimization/optimization_stop_strategies_abstract.h
            - f(x) must be a valid expression that evaluates to a double
            - is_col_vector(x) == true
            - derivative_eps > 0 
        ensures
            - Performs an unconstrained maximization of the function f() using the given
              search_strategy.  
            - The function is optimized until stop_strategy decides that an acceptable 
              point has been found or f(#x) > max_f.
            - #x == the value of x that was found to maximize f()
            - Uses the dlib::derivative(f,derivative_eps) function to compute gradient
              information.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATIOn_ABSTRACT_



