// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_OPTIMIZATIOn_ABSTRACT_
#ifdef DLIB_OPTIMIZATIOn_ABSTRACT_

#include <cmath>
#include <limits>
#include "../matrix/matrix_abstract.h"
#include "../algs.h"


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
        typename funct, 
        typename T
        >
    class line_search_funct; 
    /*!
        This is a function object l(double x) == f(start + x*direction).  That is, it 
        represents the function l(double x) and takes f, start, and direction as arguments.
    !*/

    template <
        typename funct, 
        typename T
        >
    const line_search_funct<funct,T> make_line_search_function (
        const funct& f, 
        const T& start, 
        const T& direction
    ); 
    /*!
        requires
            - f == a function that returns a scalar
            - f must take a dlib::matrix that is a column vector
            - is_matrix<T>::value == true (i.e. T must be a dlib::matrix)
            - start.nc() == 1  
            - direction.nc() == 1
              (i.e. start and direction should be column vectors)
            - f(start + 1.5*direction) should be a valid expression that
              evaluates to a double
        ensures
            - returns a function that represents the function l(double x) 
              that is defined as:
                - l(x) == f(start + x*direction)
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                    Functions that perform unconstrained optimization 
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    inline double poly_min_extrap (
        double f0,
        double d0,
        double f1,
        double d1
    );
    /*!
        ensures
            - let c(x) be a 3rd degree polynomial such that:
                - c(0) == f0
                - c(1) == f1
                - derivative of c(x) at x==0 is d0
                - derivative of c(x) at x==1 is d1
            - returns the point in the range [0,1] that minimizes the polynomial c(x) 
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename funct, 
        typename funct_der
        >
    double line_search (
        const funct& f, 
        const funct_der& der, 
        double rho, 
        double sigma, 
        double minf,
        double& f0_out
    );
    /*!
        requires
            - 1 > sigma > rho > 0
            - f and der are scalar functions of scalars
              (e.g. line_search_funct objects)
            - der is the derivative of f
        ensures
            - returns a value alpha such that f(alpha) is
              significantly closer to the minimum of f than f(0).
            - bigger values of sigma result in a less accurate but faster line search
            - f0_out == f(0)
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct, 
        typename funct_der, 
        typename T
        >
    void find_min_quasi_newton (
        const funct& f, 
        const funct_der& der, 
        T& x, 
        double min_f, 
        double min_delta = 1e-7 
    );
    /*!
        requires
            - min_delta >= 0 
            - f(x) must be a valid expression that evaluates to a double
            - der(x) must be a valid expression that evaluates to the derivative of
              f() at x.
            - is_matrix<T>::value == true (i.e. T must be a dlib::matrix type)
            - x.nc() == 1 (i.e. x must be a column vector)
        ensures
            - Performs an unconstrained minimization of the function f() using a 
              quasi newton method.  The optimization stops when any of the following
              conditions are satisfied: 
                - the change in f() from one iteration to the next is less than min_delta
                - f(#x) <= min_f
            - #x == the value of x that was found to minimize f()
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename funct, 
        typename T
        >
    void find_min_quasi_newton2 (
        const funct& f, 
        T& x, 
        double min_f, 
        double min_delta = 1e-7, 
        const double derivative_eps = 1e-7 
    );
    /*!
        requires
            - min_delta >= 0 
            - derivative_eps > 0 
            - f(x) must be a valid expression that evaluates to a double
            - is_matrix<T>::value == true (i.e. T must be a dlib::matrix type)
            - x.nc() == 1 (i.e. x must be a column vector)
        ensures
            - Performs an unconstrained minimization of the function f() using a 
              quasi newton method.  The optimization stops when any of the following
              conditions are satisfied: 
                - the change in f() from one iteration to the next is less than min_delta
                - f(#x) <= min_f
            - Uses the dlib::derivative(f,derivative_eps) function to compute gradient
              information
            - #x == the value of x that was found to minimize f()
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename funct, 
        typename funct_der, 
        typename T
        >
    void find_min_conjugate_gradient (
        const funct& f, 
        const funct_der& der, 
        T& x, 
        double min_f, 
        double min_delta = 1e-7
    );
    /*!
        requires
            - min_delta >= 0 
            - f(x) must be a valid expression that evaluates to a double
            - der(x) must be a valid expression that evaluates to the derivative of
              f() at x.
            - is_matrix<T>::value == true (i.e. T must be a dlib::matrix type)
            - x.nc() == 1 (i.e. x must be a column vector)
        ensures
            - Performs an unconstrained minimization of the function f() using a 
              conjugate gradient method.  The optimization stops when any of the following
              conditions are satisfied: 
                - the change in f() from one iteration to the next is less than min_delta
                - f(#x) <= min_f
            - #x == the value of x that was found to minimize f()
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename funct, 
        typename T
        >
    void find_min_conjugate_gradient2 (
        const funct& f, 
        T& x, 
        double min_f, 
        double min_delta = 1e-7,
        const double derivative_eps = 1e-7 
    );
    /*!
        requires
            - min_delta >= 0 
            - derivative_eps > 0
            - f(x) must be a valid expression that evaluates to a double
            - der(x) must be a valid expression that evaluates to the derivative of
              f() at x.
            - is_matrix<T>::value == true (i.e. T must be a dlib::matrix type)
            - x.nc() == 1 (i.e. x must be a column vector)
        ensures
            - Performs an unconstrained minimization of the function f() using a 
              conjugate gradient method.  The optimization stops when any of the following
              conditions are satisfied: 
                - the change in f() from one iteration to the next is less than min_delta
                - f(#x) <= min_f
            - Uses the dlib::derivative(f,derivative_eps) function to compute gradient
              information
            - #x == the value of x that was found to minimize f()
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATIOn_ABSTRACT_



