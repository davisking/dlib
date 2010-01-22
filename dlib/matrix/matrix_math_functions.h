// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_MATH_FUNCTIONS
#define DLIB_MATRIx_MATH_FUNCTIONS 

#include "matrix_math_functions_abstract.h"
#include "matrix_utilities.h"
#include "matrix.h"
#include "../algs.h"
#include <cmath>
#include <complex>
#include <limits>


namespace dlib
{

// ----------------------------------------------------------------------------------------

#define DLIB_MATRIX_SIMPLE_STD_FUNCTION(name,extra_cost) struct op_##name {     \
    template <typename EXP>                                                     \
    struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>          \
    {                                                                           \
        const static long cost = EXP::cost+(extra_cost);                        \
        typedef typename EXP::type type;                                        \
        template <typename M>                                                   \
        static type apply ( const M& m, long r, long c)                         \
        { return static_cast<type>(std::name(m(r,c))); }                        \
    };};                                                                        \
    template < typename EXP >                                                   \
    const matrix_unary_exp<EXP,op_##name> name (                                \
        const matrix_exp<EXP>& m)                                               \
    {                                                                           \
        return matrix_unary_exp<EXP,op_##name>(m.ref());                        \
    }                                                                           

// ----------------------------------------------------------------------------------------

DLIB_MATRIX_SIMPLE_STD_FUNCTION(sqrt,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(log,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(log10,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(exp,7)

DLIB_MATRIX_SIMPLE_STD_FUNCTION(conj,1)

DLIB_MATRIX_SIMPLE_STD_FUNCTION(ceil,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(floor,7)

DLIB_MATRIX_SIMPLE_STD_FUNCTION(sin,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(cos,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(tan,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(sinh,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(cosh,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(tanh,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(asin,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(acos,7)
DLIB_MATRIX_SIMPLE_STD_FUNCTION(atan,7)

// ----------------------------------------------------------------------------------------

    struct op_sigmoid 
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                return static_cast<type>(1/(1 + std::exp(-m(r,c))));
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_sigmoid> sigmoid (
        const matrix_exp<EXP>& m
    )
    {
        return matrix_unary_exp<EXP,op_sigmoid>(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_round_zeros 
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M, typename T>
            static type apply ( const M& m, const T& eps, long r, long c)
            { 
                const type temp = m(r,c);
                if (temp >= eps || temp <= -eps)
                    return temp;
                else
                    return 0;
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,typename EXP::type,op_round_zeros> round_zeros (
        const matrix_exp<EXP>& m
    )
    {
        // you can only round matrices that contain built in scalar types like double, long, float, etc...
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);
        typedef matrix_scalar_binary_exp<EXP,typename EXP::type, op_round_zeros> exp;
        return exp(m.ref(),10*std::numeric_limits<typename EXP::type>::epsilon());
    }

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,typename EXP::type,op_round_zeros> round_zeros (
        const matrix_exp<EXP>& m,
        typename EXP::type eps 
    )
    {
        // you can only round matrices that contain built in scalar types like double, long, float, etc...
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);
        return matrix_scalar_binary_exp<EXP,typename EXP::type, op_round_zeros>(m.ref(),eps);
    }

// ----------------------------------------------------------------------------------------

    struct op_cubed 
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                const type temp = m(r,c);
                return temp*temp*temp; 
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_cubed> cubed (
        const matrix_exp<EXP>& m
    )
    {
        return matrix_unary_exp<EXP,op_cubed>(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_squared
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+6;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                const type temp = m(r,c);
                return temp*temp; 
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_squared> squared (
        const matrix_exp<EXP>& m
    )
    {
        return matrix_unary_exp<EXP,op_squared>(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_pow
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M, typename S>
            static type apply ( const M& m, const S& s, long r, long c)
            { return static_cast<type>(std::pow(m(r,c),s)); }
        };
    };

    template <
        typename EXP,
        typename S
        >
    const matrix_scalar_binary_exp<EXP,typename EXP::type,op_pow> pow (
        const matrix_exp<EXP>& m,
        const S& s
    )
    {
        // you can only round matrices that contain floats, doubles or long doubles.
        COMPILE_TIME_ASSERT((
                is_same_type<typename EXP::type,float>::value == true || 
                is_same_type<typename EXP::type,double>::value == true || 
                is_same_type<typename EXP::type,long double>::value == true 
        ));
        return matrix_scalar_binary_exp<EXP,typename EXP::type,op_pow>(m.ref(),s);
    }

// ----------------------------------------------------------------------------------------

    struct op_pow2
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M, typename S>
            static type apply ( const M& m, const S& s, long r, long c)
            { return static_cast<type>(std::pow(s,m(r,c))); }
        };
    };

    template <
        typename EXP,
        typename S
        >
    const matrix_scalar_binary_exp<EXP,typename EXP::type,op_pow2> pow (
        const S& s,
        const matrix_exp<EXP>& m
    )
    {
        // you can only round matrices that contain floats, doubles or long doubles.
        COMPILE_TIME_ASSERT((
                is_same_type<typename EXP::type,float>::value == true || 
                is_same_type<typename EXP::type,double>::value == true || 
                is_same_type<typename EXP::type,long double>::value == true 
        ));
        return matrix_scalar_binary_exp<EXP,typename EXP::type,op_pow2>(m.ref(),s);
    }

// ----------------------------------------------------------------------------------------

    struct op_reciprocal
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+6;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                const type temp = m(r,c);
                if (temp != static_cast<type>(0))
                    return static_cast<type>((type)1.0/temp);
                else
                    return 0;
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_reciprocal> reciprocal (
        const matrix_exp<EXP>& m
    )
    {
        // you can only compute reciprocal matrices that contain floats, doubles or long doubles.
        COMPILE_TIME_ASSERT((
                is_same_type<typename EXP::type,float>::value == true || 
                is_same_type<typename EXP::type,double>::value == true || 
                is_same_type<typename EXP::type,long double>::value == true  ||
                is_same_type<typename EXP::type,std::complex<float> >::value == true || 
                is_same_type<typename EXP::type,std::complex<double> >::value == true || 
                is_same_type<typename EXP::type,std::complex<long double> >::value == true 
        ));
        return matrix_unary_exp<EXP,op_reciprocal>(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_reciprocal_max
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+6;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                const type temp = m(r,c);
                if (temp != static_cast<type>(0))
                    return static_cast<type>((type)1.0/temp);
                else
                    return std::numeric_limits<type>::max();
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_reciprocal_max> reciprocal_max (
        const matrix_exp<EXP>& m
    )
    {
        // you can only compute reciprocal_max matrices that contain floats, doubles or long doubles.
        COMPILE_TIME_ASSERT((
                is_same_type<typename EXP::type,float>::value == true || 
                is_same_type<typename EXP::type,double>::value == true || 
                is_same_type<typename EXP::type,long double>::value == true 
        ));
        return matrix_unary_exp<EXP,op_reciprocal_max>(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_normalize
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+5;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, const type& s, long r, long c)
            { 
                return m(r,c)*s;
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_scalar_binary_exp<EXP,typename EXP::type,op_normalize> normalize (
        const matrix_exp<EXP>& m
    )
    {
        // you can only compute normalized matrices that contain floats, doubles or long doubles.
        COMPILE_TIME_ASSERT((
                is_same_type<typename EXP::type,float>::value == true || 
                is_same_type<typename EXP::type,double>::value == true || 
                is_same_type<typename EXP::type,long double>::value == true 
        ));
        typedef matrix_scalar_binary_exp<EXP,typename EXP::type, op_normalize> exp;

        typename EXP::type temp = std::sqrt(sum(squared(m)));
        if (temp != 0.0)
            temp = 1.0/temp;

        return exp(m.ref(),temp);
    }

// ----------------------------------------------------------------------------------------

    struct op_round
    {
        template <typename EXP, typename enabled = void>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                return static_cast<type>(std::floor(m(r,c)+0.5)); 
            }
        };

        template <typename EXP>
        struct op<EXP,typename enable_if_c<std::numeric_limits<typename EXP::type>::is_integer>::type > 
                : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                return m(r,c);
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_round> round (
        const matrix_exp<EXP>& m
    )
    {
        // you can only round matrices that contain built in scalar types like double, long, float, etc...
        COMPILE_TIME_ASSERT(is_built_in_scalar_type<typename EXP::type>::value);
        typedef matrix_unary_exp<EXP,op_round> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_abs
    {
        template <typename EXP, typename return_type = typename EXP::type>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+7;
            typedef typename EXP::type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                return static_cast<type>(std::abs(m(r,c))); 
            }
        };

        template <typename EXP, typename T>
        struct op<EXP, std::complex<T> > : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost;
            typedef T type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                return static_cast<type>(std::abs(m(r,c))); 
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_abs> abs (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_abs> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_complex_matrix 
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+1;
            typedef std::complex<typename EXP::type> type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { 
                return type(m(r,c));
            }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_complex_matrix> complex_matrix (
        const matrix_exp<EXP>& m
    )
    {
        return matrix_unary_exp<EXP,op_complex_matrix>(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_complex_matrix2
    {
        template <typename EXP1, typename EXP2>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP1,EXP2>
        {
            const static long cost = EXP1::cost+EXP2::cost+1;
            typedef std::complex<typename EXP1::type> type;

            template <typename M1, typename M2>
            static type apply ( const M1& m1, const M2& m2 , long r, long c)
            { return type(m1(r,c),m2(r,c)); }
        };
    };

    template <
        typename EXP1,
        typename EXP2
        >
    const matrix_binary_exp<EXP1,EXP2,op_complex_matrix2> complex_matrix (
        const matrix_exp<EXP1>& real_part,
        const matrix_exp<EXP2>& imag_part 
    )
    {
        COMPILE_TIME_ASSERT((is_same_type<typename EXP1::type,typename EXP2::type>::value == true));
        COMPILE_TIME_ASSERT(EXP1::NR == EXP2::NR || EXP1::NR == 0 || EXP2::NR == 0);
        COMPILE_TIME_ASSERT(EXP1::NC == EXP2::NC || EXP1::NC == 0 || EXP2::NC == 0);

        DLIB_ASSERT(real_part.nr() == imag_part.nr() &&
               real_part.nc() == imag_part.nc(), 
            "\tconst matrix_exp::type complex_matrix(real_part, imag_part)"
            << "\n\tYou can only make a complex matrix from two equally sized matrices"
            << "\n\treal_part.nr(): " << real_part.nr()
            << "\n\treal_part.nc(): " << real_part.nc() 
            << "\n\timag_part.nr(): " << imag_part.nr()
            << "\n\timag_part.nc(): " << imag_part.nc() 
            );
        typedef matrix_binary_exp<EXP1,EXP2,op_complex_matrix2> exp;
        return exp(real_part.ref(),imag_part.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_norm
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost+6;
            typedef typename EXP::type::value_type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return std::norm(m(r,c)); }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_norm> norm (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_norm> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_real
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost;
            typedef typename EXP::type::value_type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return std::real(m(r,c)); }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_real> real (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_real> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

    struct op_imag
    {
        template <typename EXP>
        struct op : has_nondestructive_aliasing, preserves_dimensions<EXP>
        {
            const static long cost = EXP::cost;
            typedef typename EXP::type::value_type type;
            template <typename M>
            static type apply ( const M& m, long r, long c)
            { return std::imag(m(r,c)); }
        };
    };

    template <
        typename EXP
        >
    const matrix_unary_exp<EXP,op_imag> imag (
        const matrix_exp<EXP>& m
    )
    {
        typedef matrix_unary_exp<EXP,op_imag> exp;
        return exp(m.ref());
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_MATH_FUNCTIONS

