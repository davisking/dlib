// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATIOn_LINE_SEARCH_H_
#define DLIB_OPTIMIZATIOn_LINE_SEARCH_H_

#include <cmath>
#include <limits>
#include "../matrix.h"
#include "../algs.h"
#include "optimization_line_search_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename funct, typename T>
    class line_search_funct 
    {
    public:
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        line_search_funct(const funct& f_, const T& start_, const T& direction_) 
            : f(f_),start(start_), direction(direction_), matrix_r(0), scalar_r(0)
        {}

        line_search_funct(const funct& f_, const T& start_, const T& direction_, T& r) 
            : f(f_),start(start_), direction(direction_), matrix_r(&r), scalar_r(0)
        {}

        line_search_funct(const funct& f_, const T& start_, const T& direction_, double& r) 
            : f(f_),start(start_), direction(direction_), matrix_r(0), scalar_r(&r)
        {}

        double operator()(const double& x) const
        {
            return get_value(f(start + x*direction));
        }

    private:

        double get_value (const double& r) const
        {
            // save a copy of this value for later
            if (scalar_r)
                *scalar_r = r;

            return r;
        }

        template <typename U>
        double get_value (const U& r) const
        {
            // U should be a matrix type
            COMPILE_TIME_ASSERT(is_matrix<U>::value);

            // save a copy of this value for later
            if (matrix_r)
                *matrix_r = r;

            return dot(r,direction);
        }

        const funct& f;
        const T& start;
        const T& direction;
        T* matrix_r;
        double* scalar_r;
    };

    template <typename funct, typename T>
    const line_search_funct<funct,T> make_line_search_function(const funct& f, const T& start, const T& direction) 
    { 
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(start) && is_col_vector(direction) && start.size() == direction.size(),
            "\tline_search_funct make_line_search_function(f,start,direction)"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tstart.nc():     " << start.nc()
            << "\n\tdirection.nc(): " << direction.nc()
            << "\n\tstart.nr():     " << start.nr()
            << "\n\tdirection.nr(): " << direction.nr()
        );
        return line_search_funct<funct,T>(f,start,direction); 
    }

// ----------------------------------------------------------------------------------------

    template <typename funct, typename T>
    const line_search_funct<funct,T> make_line_search_function(const funct& f, const T& start, const T& direction, double& f_out) 
    { 
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(start) && is_col_vector(direction) && start.size() == direction.size(),
            "\tline_search_funct make_line_search_function(f,start,direction)"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tstart.nc():     " << start.nc()
            << "\n\tdirection.nc(): " << direction.nc()
            << "\n\tstart.nr():     " << start.nr()
            << "\n\tdirection.nr(): " << direction.nr()
        );
        return line_search_funct<funct,T>(f,start,direction, f_out); 
    }

// ----------------------------------------------------------------------------------------

    template <typename funct, typename T>
    const line_search_funct<funct,T> make_line_search_function(const funct& f, const T& start, const T& direction, T& grad_out) 
    { 
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(start) && is_col_vector(direction) && start.size() == direction.size(),
            "\tline_search_funct make_line_search_function(f,start,direction)"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tstart.nc():     " << start.nc()
            << "\n\tdirection.nc(): " << direction.nc()
            << "\n\tstart.nr():     " << start.nr()
            << "\n\tdirection.nr(): " << direction.nr()
        );
        return line_search_funct<funct,T>(f,start,direction,grad_out); 
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    inline double poly_min_extrap (
        double f0,
        double d0,
        double f1,
        double d1
    )
    {
        const double n = 3*(f1 - f0) - 2*d0 - d1;
        const double e = d0 + d1 - 2*(f1 - f0);


        // find the minimum of the derivative of the polynomial

        double temp = std::max(n*n - 3*e*d0,0.0);

        if (temp < 0)
            return 0.5;

        temp = std::sqrt(temp);

        if (std::abs(e) <= std::numeric_limits<double>::epsilon())
            return 0.5;

        // figure out the two possible min values
        double x1 = (temp - n)/(3*e);
        double x2 = -(temp + n)/(3*e);

        // compute the value of the interpolating polynomial at these two points
        double y1 = f0 + d0*x1 + n*x1*x1 + e*x1*x1*x1;
        double y2 = f0 + d0*x2 + n*x2*x2 + e*x2*x2*x2;

        // pick the best point
        double x;
        if (y1 < y2)
            x = x1;
        else
            x = x2;

        // now make sure the minimum is within the allowed range of (0,1) 
        return put_in_range(0,1,x);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename funct, 
        typename funct_der
        >
    double line_search (
        const funct& f, 
        const double f0,
        const funct_der& der, 
        const double d0,
        double rho, 
        double sigma, 
        double min_f
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);
        COMPILE_TIME_ASSERT(is_function<funct_der>::value == false);

        DLIB_ASSERT (
            0 < rho && rho < sigma && sigma < 1,
            "\tdouble line_search()"
            << "\n\tYou have given invalid arguments to this function"
            << "\n\tsigma: " << sigma
            << "\n\trho:   " << rho 
        );

        // The bracketing phase of this function is implemented according to block 2.6.2 from
        // the book Practical Methods of Optimization by R. Fletcher.   The sectioning 
        // phase is an implementation of 2.6.4 from the same book.

        // tau1 > 1. Controls the alpha jump size during the search
        const double tau1 = 9;

        // it must be the case that 0 < tau2 < tau3 <= 1/2 for the algorithm to function
        // correctly but the specific values of tau2 and tau3 aren't super important.
        const double tau2 = 1.0/10.0;
        const double tau3 = 1.0/2.0;


        // Stop right away and return a step size of 0 if the gradient is 0 at the starting point
        if (std::abs(d0) < std::numeric_limits<double>::epsilon())
            return 0;

        // Stop right away if the current value is good enough according to min_f
        if (f0 <= min_f)
            return 0;

        // Figure out a reasonable upper bound on how large alpha can get.
        const double mu = (min_f-f0)/(rho*d0);


        double alpha = 1;
        if (mu < 0)
            alpha = -alpha;
        alpha = put_in_range(0, 0.65*mu, alpha);


        double last_alpha = 0;
        double last_val = f0;
        double last_val_der = d0;

        // the bracketing stage will find a find a range of points [a,b]
        // that contains a reasonable solution to the line search
        double a, b;

        // These variables will hold the values and derivatives of f(a) and f(b)
        double a_val, b_val, a_val_der, b_val_der;

        // This thresh value represents the Wolfe curvature condition
        const double thresh = std::abs(sigma*d0);

        // do the bracketing stage to find the bracket range [a,b]
        while (true)
        {
            const double val = f(alpha);
            const double val_der = der(alpha);

            // we are done with the line search since we found a value smaller
            // than the minimum f value
            if (val <= min_f)
                return alpha;

            if (val > f0 + rho*alpha*d0 || val >= last_val)
            {
                a_val = last_val;
                a_val_der = last_val_der;
                b_val = val;
                b_val_der = val_der;

                a = last_alpha;
                b = alpha;
                break;
            }

            if (std::abs(val_der) <= thresh)
                return alpha;

            // if we are stuck not making progress then quit with the current alpha
            if (last_alpha == alpha)
                return alpha;

            if (val_der >= 0)
            {
                a_val = val;
                a_val_der = val_der;
                b_val = last_val;
                b_val_der = last_val_der;

                a = alpha;
                b = last_alpha;
                break;
            }

            if (mu <= 2*alpha - last_alpha)
            {
                last_alpha = alpha;
                alpha = mu;
            }
            else
            {
                const double temp = alpha;

                double first = 2*alpha - last_alpha;
                double last;
                if (mu > 0)
                    last = std::min(mu, alpha + tau1*(alpha - last_alpha));
                else
                    last = std::max(mu, alpha + tau1*(alpha - last_alpha));


                // pick a point between first and last by doing some kind of interpolation
                if (last_alpha < alpha)
                    alpha = last_alpha + (alpha-last_alpha)*poly_min_extrap(last_val, last_val_der, val, val_der);
                else
                    alpha = alpha + (last_alpha-alpha)*poly_min_extrap(val, val_der, last_val, last_val_der);

                alpha = put_in_range(first,last,alpha);


                last_alpha = temp;
            }

            last_val = val;
            last_val_der = val_der;

        }


        // Now do the sectioning phase from 2.6.4
        while (true)
        {
            double first = a + tau2*(b-a);
            double last = b - tau3*(b-a);

            // use interpolation to pick alpha between first and last
            alpha = a + (b-a)*poly_min_extrap(a_val, a_val_der, b_val, b_val_der);
            alpha = put_in_range(first,last,alpha);

            const double val = f(alpha);
            const double val_der = der(alpha);

            // we are done with the line search since we found a value smaller
            // than the minimum f value
            if (val <= min_f)
                return alpha;

            // stop if the interval gets so small that it isn't shrinking any more due to rounding error 
            if (a == first || b == last)
            {
                return b;
            }


            if (val > f0 + rho*alpha*d0 || val >= a_val)
            {
                b = alpha;
                b_val = val;
                b_val_der = val_der;
            }
            else
            {
                if (std::abs(val_der) <= thresh)
                    return alpha;

                if ( (b-a)*val_der >= 0)
                {
                    b = a;
                    b_val = a_val;
                    b_val_der = a_val_der;
                }

                a = alpha;
                a_val = val;
                a_val_der = val_der;
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATIOn_LINE_SEARCH_H_

