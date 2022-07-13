// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_KAISER
#define DLIB_MATH_KAISER

#include <type_traits>
#include "bessel.h"

namespace dlib
{
    /*! Strong types !*/

    struct attenuation_t
    /*!
        This object represents a desired attenuation in dB.
        This is automatically converted into a beta_t value suitable
        for constructing a kaiser window.
        See https://www.mathworks.com/help/signal/ug/kaiser-window.html on
        filter design
    !*/
    {
        explicit attenuation_t(double attenuation_db) : v{attenuation_db} {}
        double v = 0.0;
    };

    struct beta_t
    /*!
        This value determines the shape of the kaiser window.
        See https://en.wikipedia.org/wiki/Kaiser_window#Definition for more details
    !*/
    {
        explicit beta_t(double beta) : v{beta} {}
        beta_t(attenuation_t attenuation_db)
        {
            if (attenuation_db.v > 50.0)
                v = 0.1102*(attenuation_db.v - 8.7);
            else if (attenuation_db.v >= 21.0)
                v = 0.5842*std::pow(attenuation_db.v - 21, 0.4) + 0.07886*(attenuation_db.v - 21);
        }
        double v = 0.0;
    };

    struct index_t
    /*!
        This object is a strong type representing an array index.
        It is suitable for distinguishing which overload of the kaiser()
        function should be used.
    !*/
    {
        explicit index_t(std::size_t i_) : i{i_} {}
        std::size_t i = 0;
    };

    struct window_duration
    /*!
        This ojbect is a strong type representing the window duration of a kaiser window.
        See https://en.wikipedia.org/wiki/Kaiser_window.
    !*/
    {
        explicit window_duration(double L_) : L{L_} {}
        double L = 0.0;
    };

    struct window_length
    /*!
        This ojbect is a strong type representing the window length of a kaiser window.
        See https://en.wikipedia.org/wiki/Kaiser_window.
    !*/
    {
        explicit window_length(std::size_t N_) : N{N_} {}
        std::size_t N = 0;
    };

    template<typename R>
    inline R kaiser(R x, window_duration L, beta_t beta)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window

        ensures
            - returns the kaiser window function when |x| <= L/2 where L is the window duration
            - returns 0 otherwise
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");

        if (std::abs(x) <= L.L/R{2})
        {
            const R r = 2*x/L.L;
            const R a = dlib::cyl_bessel_i(0, beta.v*sqrt(1-r*r));
            const R b = dlib::cyl_bessel_i(0, beta.v);
            return a / b;
        }
        else
        {
            return R{0};
        }
    }

    template<typename R>
    inline R kaiser(index_t i, window_length N, beta_t beta)
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
        return kaiser(R(i.i) - R(N.N-1) / R(2), window_duration{R(N.N-1)}, beta);
    }
}

#endif //DLIB_MATH_KAISER
