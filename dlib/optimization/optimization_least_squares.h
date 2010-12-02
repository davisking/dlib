// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIMIZATION_LEAST_SQuARES_H___
#define DLIB_OPTIMIZATION_LEAST_SQuARES_H___

#include "../matrix.h"
#include "optimization_trust_region.h"
#include "optimization_least_squares_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename column_vector_type,
        typename funct_type,
        typename funct_der_type,
        typename vector_type
        >
    class least_squares_function_model 
    {
    public:
        least_squares_function_model (
            const funct_type& f_,
            const funct_der_type& der_,
            const vector_type& list_
        ) : f(f_), der(der_), list(list_) 
        {
            S = 0;
            last_f = 0;
            last_f2 = 0;
        }

        const funct_type& f;
        const funct_der_type& der;
        const vector_type& list;

        typedef typename column_vector_type::type type;
        typedef typename column_vector_type::mem_manager_type mem_manager_type;
        typedef typename column_vector_type::layout_type layout_type;
        const static long NR = column_vector_type::NR;

        typedef column_vector_type column_vector;
        typedef matrix<type,NR,NR,mem_manager_type,layout_type> general_matrix;


        type operator() ( 
            const column_vector& x
        ) const
        {
            type result = 0;
            for (long i = 0; i < list.size(); ++i)
            {
                const type temp = f(list(i), x);
                result += temp*temp;
            }

            last_f = 0.5*result;
            return 0.5*result;
        }

        void get_derivative_and_hessian (
            const column_vector& x,
            column_vector& d,
            general_matrix& h
        ) const
        {
            matrix<type,0,NR,mem_manager_type,layout_type> J(list.size(), x.size());
            matrix<type,0,1,mem_manager_type,layout_type> r(list.size(),1);

            for (long i = 0; i < list.size(); ++i)
            {
                r(i) = f(list(i), x);
                set_rowm(J,i) = trans(der(list(i), x));
            }


            d = trans(J)*r;
            h = trans(J)*J;

            if (S.size() == 0)
            {
                S.set_size(x.size(), x.size());
                S = 0;
            }

            if (last_r.size() != 0)
            {
                column_vector s, y, yy, temp;

                s = x - last_x;
                y = trans(J)*r - trans(last_J)*last_r;
                yy = trans(J)*r - trans(last_J)*r;

                const type ys = trans(y)*s;
                temp = yy - S*s;
                type scale = std::min<type>(1, std::abs(dot(s,yy))/std::abs(trans(s)*S*s));

                S = scale*S + (temp*trans(y) + y*trans(temp))/(ys) - dot(temp,s)/ys/ys*y*trans(y);

                // check how well both the models fit the last change in f we saw
                const type measured_delta = last_f2 - last_f;
                s = -s;
                const type h_predicted_delta = 0.5*trans(s)*h*s + trans(d)*s;
                const type s_predicted_delta = 0.5*trans(s)*(h+S)*s + trans(d)*s;

                const type h_error = std::abs((h_predicted_delta/measured_delta) - 1);
                const type s_error = std::abs((s_predicted_delta/measured_delta) - 1);

                if (s_error < h_error && h_error > 0.01)
                {
                    h += make_symmetric(S);
                }
                else if (s_error > 10)
                {
                    S = 0;
                }
            }

            last_r = r;
            last_J = J;
            last_x = x;

            last_f2 = last_f;
        }

        mutable type last_f;   // value of function we saw in last operator()
        mutable type last_f2;  // value of last_f we saw in get_derivative_and_hessian()

        mutable general_matrix S;
        mutable column_vector last_x;
        mutable matrix<type,0,1,mem_manager_type,layout_type> last_r;
        mutable matrix<type,0,NR,mem_manager_type,layout_type> last_J;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename column_vector_type,
        typename funct_type,
        typename funct_der_type,
        typename vector_type
        >
    least_squares_function_model<column_vector_type,funct_type,funct_der_type,vector_type> least_squares_model (
        const funct_type& f,
        const funct_der_type& der,
        const vector_type& list
    )
    {
        return least_squares_function_model<column_vector_type,funct_type,funct_der_type,vector_type>(f,der,list);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename stop_strategy_type,
        typename funct_type,
        typename funct_der_type,
        typename vector_type,
        typename T
        >
    double solve_least_squares (
        stop_strategy_type stop_strategy,
        const funct_type& f,
        const funct_der_type& der,
        const vector_type& list,
        T& x, 
        double radius = 1
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_vector(vector_to_matrix(list)) && list.size() > 0 && 
                    is_col_vector(x) && radius > 0,
            "\t double solve_least_squares()"
            << "\n\t invalid arguments were given to this function"
            << "\n\t is_vector(list):  " << is_vector(vector_to_matrix(list)) 
            << "\n\t list.size():      " << list.size() 
            << "\n\t is_col_vector(x): " << is_col_vector(x) 
            << "\n\t radius:           " << radius
            );

        return find_min_trust_region(stop_strategy,
                                     least_squares_model<T>(f, der, vector_to_matrix(list)), 
                                     x, 
                                     radius);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename column_vector_type,
        typename funct_type,
        typename funct_der_type,
        typename vector_type
        >
    class least_squares_lm_function_model 
    {
    public:
        least_squares_lm_function_model (
            const funct_type& f_,
            const funct_der_type& der_,
            const vector_type& list_
        ) : f(f_), der(der_), list(list_) {}

        const funct_type& f;
        const funct_der_type& der;
        const vector_type& list;

        typedef typename column_vector_type::type type;
        typedef typename column_vector_type::mem_manager_type mem_manager_type;
        typedef typename column_vector_type::layout_type layout_type;
        const static long NR = column_vector_type::NR;

        typedef column_vector_type column_vector;
        typedef matrix<type,NR,NR,mem_manager_type,layout_type> general_matrix;

        type operator() ( 
            const column_vector& x
        ) const
        {
            type result = 0;
            for (long i = 0; i < list.size(); ++i)
            {
                const type temp = f(list(i), x);
                result += temp*temp;
            }

            return 0.5*result;
        }

        void get_derivative_and_hessian (
            const column_vector& x,
            column_vector& d,
            general_matrix& h
        ) const
        {
            matrix<type,0,NR,mem_manager_type,layout_type> J(list.size(), x.size());
            matrix<type,0,1,mem_manager_type,layout_type> r(list.size(),1);

            for (long i = 0; i < list.size(); ++i)
            {
                r(i) = f(list(i), x);
                set_rowm(J,i) = trans(der(list(i), x));
            }

            d = trans(J)*r;
            h = trans(J)*J;
        }
    };

// ----------------------------------------------------------------------------------------

    template <
        typename column_vector_type,
        typename funct_type,
        typename funct_der_type,
        typename vector_type
        >
    least_squares_lm_function_model<column_vector_type,funct_type,funct_der_type,vector_type> least_squares_lm_model (
        const funct_type& f,
        const funct_der_type& der,
        const vector_type& list
    )
    {
        return least_squares_lm_function_model<column_vector_type,funct_type,funct_der_type,vector_type>(f,der,list);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename stop_strategy_type,
        typename funct_type,
        typename funct_der_type,
        typename vector_type,
        typename T
        >
    double solve_least_squares_lm (
        stop_strategy_type stop_strategy,
        const funct_type& f,
        const funct_der_type& der,
        const vector_type& list,
        T& x, 
        double radius = 1
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_vector(vector_to_matrix(list)) && list.size() > 0 && 
                    is_col_vector(x) && radius > 0,
            "\t double solve_least_squares_lm()"
            << "\n\t invalid arguments were given to this function"
            << "\n\t is_vector(list):  " << is_vector(vector_to_matrix(list)) 
            << "\n\t list.size():      " << list.size() 
            << "\n\t is_col_vector(x): " << is_col_vector(x) 
            << "\n\t radius:           " << radius
            );

        return find_min_trust_region(stop_strategy,
                                     least_squares_lm_model<T>(f, der, vector_to_matrix(list)), 
                                     x, 
                                     radius);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_OPTIMIZATION_LEAST_SQuARES_H___


