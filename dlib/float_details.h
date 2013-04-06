// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FLOAT_DEtAILS_H__
#define DLIB_FLOAT_DEtAILS_H__

#include <math.h>
#include "algs.h"
#include <limits> 

namespace dlib
{
    struct float_details
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
        !*/

        float_details() :
            exponent(0), mantissa(0)
        {}

        const static int16 is_inf  = 32000;
        const static int16 is_ninf = 32001;
        const static int16 is_nan  = 32002;

        float_details ( const double&      val) { *this = val; }
        float_details ( const float&       val) { *this = val; }
        float_details ( const long double& val) { *this = val; }

        float_details& operator= ( const double&      val) { convert_from_T(val); return *this; }
        float_details& operator= ( const float&       val) { convert_from_T(val); return *this; }
        float_details& operator= ( const long double& val) { convert_from_T(val); return *this; }

        operator double      () const { return convert_to_T<double>(); }
        operator float       () const { return convert_to_T<float>(); }
        operator long double () const { return convert_to_T<long double>(); }

        int16 exponent;
        int64 mantissa;

    private:

        double      _frexp(double      v, int* e) const { return frexp(v,e); }
        float       _frexp(float       v, int* e) const { return frexpf(v,e); }
        long double _frexp(long double v, int* e) const { return frexpl(v,e); }

        double      _ldexp(double      v, int e) const { return ldexp(v,e); }
        float       _ldexp(float       v, int e) const { return ldexpf(v,e); }
        long double _ldexp(long double v, int e) const { return ldexpl(v,e); }

        template <typename T>
        void convert_from_T (
            const T& val
        )
        {
            mantissa = 0;

            const int digits = dlib::tmin<std::numeric_limits<T>::digits, 63>::value;

            if (val == std::numeric_limits<T>::infinity())
            {
                exponent = is_inf;
            }
            else if (val == -std::numeric_limits<T>::infinity())
            {
                exponent = is_ninf;
            }
            else if (val < std::numeric_limits<T>::infinity())
            {
                int exp;
                mantissa = _frexp(val, &exp)*(((uint64)1)<<digits);
                exponent = exp - digits;

                // Compact the representation a bit by shifting off any low order bytes 
                // which are zero in the mantissa.  This makes the numbers in mantissa and
                // exponent generally smaller which can make serialization and other things
                // more efficient in some cases.
                for (int i = 0; i < 8 && ((mantissa&0xFF)==0); ++i)
                {
                    mantissa >>= 8;
                    exponent += 8;
                }
            }
            else
            {
                exponent = is_nan;
            }
        }

        template <typename T>
        T convert_to_T (
        ) const
        {
            if (exponent < is_inf)
                return _ldexp((T)mantissa, exponent);
            else if (exponent == is_inf)
                return std::numeric_limits<T>::infinity();
            else if (exponent == is_ninf)
                return -std::numeric_limits<T>::infinity();
            else
                return std::numeric_limits<T>::quiet_NaN();
        }

    };

}

#endif // DLIB_FLOAT_DEtAILS_H__

