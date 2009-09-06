// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATIOn_H_
#define DLIB_OPTIMIZATIOn_H_

#include <cmath>
#include <limits>
#include "../matrix.h"
#include "../algs.h"
#include "optimization_abstract.h"
#include "../sequence.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class objective_delta_stop_strategy
    {
    public:
        objective_delta_stop_strategy (
            double min_delta = 1e-7
        ) : _been_used(false), _min_delta(min_delta), _max_iter(0), _cur_iter(0), _prev_funct_value(0) {}

        objective_delta_stop_strategy (
            double min_delta,
            unsigned long max_iter
        ) : _been_used(false), _min_delta(min_delta), _max_iter(max_iter), _cur_iter(0), _prev_funct_value(0) {}

        template <typename T>
        bool should_continue_search (
            const T& ,
            const double funct_value,
            const T& 
        ) 
        {
            ++_cur_iter;
            if (_been_used)
            {
                // Check if we have hit the max allowable number of iterations.  (but only
                // check if _max_iter is enabled (i.e. not 0)).
                if (_max_iter != 0 && _cur_iter > _max_iter)
                    return false;

                // check if the function change was too small
                if (std::abs(funct_value - _prev_funct_value) < _min_delta)
                    return false;
            }

            _been_used = true;
            _prev_funct_value = funct_value;
            return true;
        }

    private:

        bool _been_used;
        double _min_delta;
        unsigned long _max_iter;
        unsigned long _cur_iter;
        double _prev_funct_value;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class cg_search_strategy
    {
    public:
        cg_search_strategy() : been_used(false) {}

        double get_wolfe_rho (
        ) const { return 0.001; }

        double get_wolfe_sigma (
        ) const { return 0.01; }

        template <typename T>
        const matrix<double,0,1>& get_next_direction (
            const T& ,
            const double ,
            const T& funct_derivative
        )
        /*!
            requires
                - for some function f():
                    - funct_value == f(x)
                    - funct_derivative == derivative(f)(x)
            ensures
                - returns the next search direction (from the point x).  This
                  direction is chosen using the Polak-Ribiere conjugate gradient
                  method.
        !*/
        {
            if (been_used == false)
            {
                been_used = true;
                prev_direction = -funct_derivative;
            }
            else
            {
                // Use the Polak-Ribiere (4.1.12) conjugate gradient described by Fletcher on page 83
                const double temp = trans(prev_derivative)*prev_derivative;
                // If this value hits zero then just use the direction of steepest descent.
                if (std::abs(temp) < std::numeric_limits<double>::epsilon())
                {
                    prev_derivative = funct_derivative;
                    prev_direction = -funct_derivative;
                    return prev_direction;
                }

                double b = trans(funct_derivative-prev_derivative)*funct_derivative/(temp);
                prev_direction = -funct_derivative + b*prev_direction;

            }

            prev_derivative = funct_derivative;
            return prev_direction;
        }

    private:
        bool been_used;
        matrix<double,0,1> prev_derivative;
        matrix<double,0,1> prev_direction;
    };

// ----------------------------------------------------------------------------------------

    class bfgs_search_strategy
    {
    public:
        bfgs_search_strategy() : been_used(false), been_used_twice(false) {}

        double get_wolfe_rho (
        ) const { return 0.01; }

        double get_wolfe_sigma (
        ) const { return 0.9; }

        template <typename T>
        const matrix<double,0,1>& get_next_direction (
            const T& x,
            const double ,
            const T& funct_derivative
        )
        {
            if (been_used == false)
            {
                been_used = true;
                H = identity_matrix<double>(x.size());
            }
            else
            {
                // update H with the BFGS formula from (3.2.12) on page 55 of Fletcher 
                delta = (x-prev_x); 
                gamma = funct_derivative-prev_derivative;

                double dg = dot(delta,gamma);

                // Try to set the initial value of the H matrix to something reasonable if we are still
                // in the early stages of figuring out what it is.  This formula below is what is suggested
                // in the book Numerical Optimization by Nocedal and Wright in the chapter on Quasi-Newton methods.
                if (been_used_twice == false)
                {
                    double gg = trans(gamma)*gamma;
                    if (std::abs(gg) > std::numeric_limits<double>::epsilon())
                    {
                        const double temp = put_in_range(0.01, 100, dg/gg);
                        H = diagm(uniform_matrix<double>(x.size(),1, temp));
                        been_used_twice = true;
                    }
                }

                Hg = H*gamma;
                gH = trans(trans(gamma)*H);
                double gHg = trans(gamma)*H*gamma;
                if (gHg < std::numeric_limits<double>::infinity() && dg < std::numeric_limits<double>::infinity() &&
                    dg != 0)
                {
                    H += (1 + gHg/dg)*delta*trans(delta)/(dg) - (delta*trans(gH) + Hg*trans(delta))/(dg);
                }
                else
                {
                    H = identity_matrix<double>(H.nr());
                    been_used_twice = false;
                }
            }

            prev_x = x;
            prev_direction = -H*funct_derivative;
            prev_derivative = funct_derivative;
            return prev_direction;
        }

    private:
        bool been_used;
        bool been_used_twice;
        matrix<double,0,1> prev_x;
        matrix<double,0,1> prev_derivative;
        matrix<double,0,1> prev_direction;
        matrix<double> H;
        matrix<double,0,1> delta, gamma, Hg, gH;
    };

// ----------------------------------------------------------------------------------------

    class lbfgs_search_strategy
    {
    public:
        lbfgs_search_strategy(unsigned long max_size_) : max_size(max_size_), been_used(false) {}

        lbfgs_search_strategy(const lbfgs_search_strategy& item) 
        {
            max_size = item.max_size;
            been_used = item.been_used;
            prev_x = item.prev_x;
            prev_derivative = item.prev_derivative;
            prev_direction = item.prev_direction;
            alpha = item.alpha;
            dh_temp = item.dh_temp;
        }

        double get_wolfe_rho (
        ) const { return 0.01; }

        double get_wolfe_sigma (
        ) const { return 0.9; }

        template <typename T>
        const matrix<double,0,1>& get_next_direction (
            const T& x,
            const double ,
            const T& funct_derivative
        )
        {
            if (been_used == false)
            {
                prev_direction = -funct_derivative;
                been_used = true;
            }
            else
            {
                // add an element into the stored data sequence
                dh_temp.s = x - prev_x;
                dh_temp.y = funct_derivative - prev_derivative;
                double temp = dlib::dot(dh_temp.s, dh_temp.y);
                // only accept this bit of data if temp isn't zero
                if (std::abs(temp) > std::numeric_limits<double>::epsilon())
                {
                    dh_temp.rho = 1/temp;
                    data.add(data.size(), dh_temp);
                }
                else
                {
                    data.clear();
                }

                if (data.size() > 0)
                {
                    // This block of code is from algorithm 7.4 in the Nocedal book.

                    prev_direction = -funct_derivative;
                    alpha.resize(data.size());
                    for (unsigned long i = data.size()-1; i < data.size(); --i)
                    {
                        alpha[i] = data[i].rho*dot(data[i].s, prev_direction);
                        prev_direction -= alpha[i]*data[i].y;
                    }

                    // Take a guess at what the first H matrix should be.  This formula below is what is suggested
                    // in the book Numerical Optimization by Nocedal and Wright in the chapter on Large Scale 
                    // Unconstrained Optimization (in the L-BFGS section).
                    double H_0 = 1.0/data[data.size()-1].rho/dot(data[data.size()-1].y, data[data.size()-1].y);
                    H_0 = put_in_range(0.001, 1000.0, H_0);
                    prev_direction *= H_0;

                    for (unsigned long i = 0; i < data.size(); ++i)
                    {
                        double beta = data[i].rho*dot(data[i].y, prev_direction);
                        prev_direction += data[i].s * (alpha[i] - beta);
                    }
                }
                else
                {
                    prev_derivative = -funct_derivative;
                }

            }

            if (data.size() > max_size)
            {
                // remove the oldest element in the data sequence
                data.remove(0, dh_temp);
            }

            prev_x = x;
            prev_derivative = funct_derivative;
            return prev_direction;
        }

    private:

        struct data_helper
        {
            matrix<double,0,1> s;
            matrix<double,0,1> y;
            double rho;

            friend void swap(data_helper& a, data_helper& b)
            {
                a.s.swap(b.s);
                a.y.swap(b.y);
                std::swap(a.rho, b.rho);
            }
        };
        sequence<data_helper>::kernel_2a data;

        unsigned long max_size;
        bool been_used;
        matrix<double,0,1> prev_x;
        matrix<double,0,1> prev_derivative;
        matrix<double,0,1> prev_direction;
        std::vector<double> alpha;

        data_helper dh_temp;
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                    Functions that transform other functions  
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename funct>
    class central_differences
    {
    public:
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        central_differences(const funct& f_, double eps_ = 1e-7) : f(f_), eps(eps_){}

        template <typename T>
        typename T::matrix_type operator()(const T& x) const
        {
            // T must be some sort of dlib matrix 
            COMPILE_TIME_ASSERT(is_matrix<T>::value);

            typename T::matrix_type der(x.size());
            typename T::matrix_type e(x);
            for (long i = 0; i < x.size(); ++i)
            {
                const double old_val = e(i);

                e(i) += eps;
                const double delta_plus = f(e);
                e(i) = old_val - eps;
                const double delta_minus = f(e);

                der(i) = (delta_plus - delta_minus)/(2*eps);

                // and finally restore the old value of this element
                e(i) = old_val;
            }

            return der;
        }

        double operator()(const double& x) const
        {
            return (f(x+eps)-f(x-eps))/(2*eps);
        }

    private:
        const funct& f;
        const double eps;
    };

    template <typename funct>
    const central_differences<funct> derivative(const funct& f) { return central_differences<funct>(f); }
    template <typename funct>
    const central_differences<funct> derivative(const funct& f, double eps) 
    { 
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        DLIB_ASSERT (
            eps > 0,
            "\tcentral_differences derivative(f,eps)"
            << "\n\tYou must give an epsilon > 0"
            << "\n\teps:     " << eps 
        );
        return central_differences<funct>(f,eps); 
    }

// ----------------------------------------------------------------------------------------

    template <typename funct>
    class negate_function_object 
    {
    public:
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        negate_function_object(const funct& f_) : f(f_){}

        template <typename T>
        double operator()(const T& x) const
        {
            return -f(x);
        }

    private:
        const funct& f;
    };

    template <typename funct>
    const negate_function_object<funct> negate_function(const funct& f) { return negate_function_object<funct>(f); }

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

        // TODO figure out some requirements for these two functions and add them to the abstract
        const T& get_last_scalar_eval (
        ) const { return *scalar_r; }

        const T& get_last_gradient_eval (
        ) const { return *matrix_r; }

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

            return trans(r)*direction;
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
//                    Functions that perform unconstrained optimization 
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
        double minf
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
            1 > sigma && sigma > rho && rho > 0,
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

        // Figure out a reasonable upper bound on how large alpha can get.
        const double mu = (minf-f0)/(rho*d0);


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
            if (val <= minf)
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
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);
        COMPILE_TIME_ASSERT(is_function<funct_der>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(x),
            "\tdouble find_min()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tx.nc():    " << x.nc()
        );


        T g, s;

        double f_value = f(x);
        g = der(x);

        while(stop_strategy.should_continue_search(x, f_value, g))
        {
            s = search_strategy.get_next_direction(x, f_value, g);

            double alpha = line_search(
                        make_line_search_function(f,x,s, f_value),
                        f_value,
                        make_line_search_function(der,x,s, g),
                        dot(g,s), // compute initial gradient for the line search
                        search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_f);

            x += alpha*s;
        }
    }

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
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);
        COMPILE_TIME_ASSERT(is_function<funct_der>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(x),
            "\tdouble find_max()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tx.nc():    " << x.nc()
        );

        T g, s;

        double f_value = -f(x);
        g = -der(x);

        while(stop_strategy.should_continue_search(x, f_value, g))
        {
            s = search_strategy.get_next_direction(x, f_value, g);

            double alpha = line_search(
                        negate_function(make_line_search_function(f,x,s, f_value)),
                        f_value,
                        negate_function(make_line_search_function(der,x,s, g)),
                        dot(g,s), // compute initial gradient for the line search
                        search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), -max_f);

            x += alpha*s;

            // we have to negate the outputs from the line search 
            g *= -1;
            f_value *= -1;
        }

    }

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
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(x) && derivative_eps > 0,
            "\tdouble find_min_using_approximate_derivatives()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tx.nc():         " << x.nc()
            << "\n\tderivative_eps: " << derivative_eps 
        );

        T g, s;

        double f_value = f(x);
        g = derivative(f,derivative_eps)(x);

        while(stop_strategy.should_continue_search(x, f_value, g))
        {
            s = search_strategy.get_next_direction(x, f_value, g);

            double alpha = line_search(
                        make_line_search_function(f,x,s,f_value),
                        f_value,
                        derivative(make_line_search_function(f,x,s),derivative_eps),
                        dot(g,s),  // Sometimes the following line is a better way of determining the initial gradient. 
                        //derivative(make_line_search_function(f,x,s),derivative_eps)(0),
                        search_strategy.get_wolfe_rho(), search_strategy.get_wolfe_sigma(), min_f);

            x += alpha*s;

            g = derivative(f,derivative_eps)(x);
        }

    }

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
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            is_col_vector(x) && derivative_eps > 0,
            "\tdouble find_max_using_approximate_derivatives()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tx.nc():         " << x.nc()
            << "\n\tderivative_eps: " << derivative_eps 
        );

        find_min_using_approximate_derivatives(
            search_strategy, 
            stop_strategy, 
            negate_function(f),
            x,
            -max_f,
            derivative_eps
        );
    }

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
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);
        COMPILE_TIME_ASSERT(is_function<funct_der>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            min_delta >= 0 && x.nc() == 1,
            "\tdouble find_min_quasi_newton()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tmin_delta: " << min_delta
            << "\n\tx.nc():    " << x.nc()
        );

        find_min(
            bfgs_search_strategy(), 
            objective_delta_stop_strategy(min_delta), 
            f, der, x, min_f
            );
    }

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
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);
        COMPILE_TIME_ASSERT(is_function<funct_der>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            min_delta >= 0 && x.nc() == 1,
            "\tdouble find_min_conjugate_gradient()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tmin_delta: " << min_delta
            << "\n\tx.nc():    " << x.nc()
        );

        find_min(
            cg_search_strategy(), 
            objective_delta_stop_strategy(min_delta), 
            f, der, x, min_f
            );
    }

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
        double derivative_eps = 1e-7
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            min_delta >= 0 && x.nc() == 1,
            "\tdouble find_min_quasi_newton2()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tmin_delta:      " << min_delta
            << "\n\tx.nc():         " << x.nc()
            << "\n\tderivative_eps: " << derivative_eps 
        );

        find_min_using_approximate_derivatives(
            bfgs_search_strategy(), 
            objective_delta_stop_strategy(min_delta), 
            f, x, min_f, derivative_eps
            );
    }

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
        double derivative_eps = 1e-7
    )
    {
        // You get an error on this line when you pass in a global function to this function.
        // You have to either use a function object or pass a pointer to your global function
        // by taking its address using the & operator.  (This check is here because gcc 4.0
        // has a bug that causes it to silently corrupt return values from functions that
        // invoked through a reference)
        COMPILE_TIME_ASSERT(is_function<funct>::value == false);

        COMPILE_TIME_ASSERT(is_matrix<T>::value);
        DLIB_ASSERT (
            min_delta >= 0 && x.nc() == 1 && derivative_eps > 0,
            "\tdouble find_min_conjugate_gradient2()"
            << "\n\tYou have to supply column vectors to this function"
            << "\n\tmin_delta:      " << min_delta
            << "\n\tx.nc():         " << x.nc()
            << "\n\tderivative_eps: " << derivative_eps 
        );

        find_min_using_approximate_derivatives(
            cg_search_strategy(), 
            objective_delta_stop_strategy(min_delta),
            f, x, min_f, derivative_eps
            );

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATIOn_H_

