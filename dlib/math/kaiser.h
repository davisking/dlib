// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_KAISER
#define DLIB_MATH_KAISER

#include <type_traits>
#include "bessel.h"

namespace dlib
{
    template<typename R>
    inline R attenuation_to_beta(R attenuation_db)
    /*!
        This function converts a desired attenuation value (in dB) to a beta value,
        which can be passed to either kaiser_i() or kaiser_r().
        This function is useful in filter design.
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");

        R beta{0};
        if (attenuation_db > 50.0)
            beta = 0.1102*(attenuation_db - 8.7);
        else if (attenuation_db >= 21.0)
            beta = 0.5842*std::pow(attenuation_db - 21, 0.4) + 0.07886*(attenuation_db - 21);
        return beta;
    }

    template<typename R>
    inline R kaiser_r(R x, R L, R beta)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window

        ensures
            - returns the kaiser window function when |x| <= L/2 where L is the window length
            - returns 0 otherwise
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");

        if (std::abs(x) <= L/R{2})
        {
            const R r = 2*x/L;
            const R a = dlib::cyl_bessel_i(0, beta*sqrt(1-r*r));
            const R b = dlib::cyl_bessel_i(0, beta);
            return a / b;
        }
        else
        {
            return R{0};
        }
    }

    template<typename R>
    inline R kaiser_i(std::size_t i, std::size_t N, R beta)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window
        This variant is a short-cut for computing a window function and storing it
        in an array of size N where 0 <= i < N is the array index.

        ensures
            - returns kaiser_r(i - (N-1)/2, N-1, beta)
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        return kaiser_r(R(i) - R(N-1) / R(2), R(N-1), beta);
    }
}

#endif //DLIB_MATH_KAISER
