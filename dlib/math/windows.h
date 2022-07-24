// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_WINDOWS
#define DLIB_MATH_WINDOWS

#include <type_traits>
#include "bessel.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

    /*! Strong types !*/

    struct attenuation_t
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a desired attenuation in dB.
                This is automatically converted into a beta_t value suitable
                for constructing a kaiser window.
                See https://www.mathworks.com/help/signal/ug/kaiser-window.html on
                filter design.
        !*/
        attenuation_t() = default;
        explicit attenuation_t(double attenuation_db) : v{attenuation_db} {}
        double v = 0.0;
    };

    struct beta_t
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This value determines the shape of the kaiser window.
                See https://en.wikipedia.org/wiki/Kaiser_window#Definition for more details.
        !*/
        beta_t() = default;
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

    enum WindowSymmetry
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This enum controls whether the window is a symmetric or periodic window.
                See https://en.wikipedia.org/wiki/Window_function#Symmetry for a discussion on
                symmetric vs periodic windows. This is using the same nomenclature as Matlab and Scipy
                when describing windows as either symmetric or periodic.
        !*/
        SYMMETRIC,
        PERIODIC
    };

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R kaiser(R x, R L, beta_t beta)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window.

        requires
            - R is float, double, or long double
        ensures
            - returns the kaiser window function when |x| <= L/2 where L is the window duration
            - returns 0 otherwise
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");

        if (std::abs(x) <= L/R{2})
        {
            const R r = 2*x/L;
            const R a = dlib::cyl_bessel_i(0, beta.v*std::sqrt(1-r*r));
            const R b = dlib::cyl_bessel_i(0, beta.v);
            return a / b;
        }
        else
        {
            return R{0};
        }
    }

    template<typename R>
    R kaiser(std::size_t i, std::size_t N, beta_t beta, WindowSymmetry type)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window
        This variant is a short-cut for computing a window function and storing it
        in an array of size N where 0 <= i < N is the array index.

        requires
            - R is float, double, or long double
            - 0 <= i < N
        ensures
            - returns kaiser(i - (N-1)/2, window_duration{N-1}, beta)
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        return kaiser(R(i) - R(size) / R(2), R(size), beta);
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R hann(std::size_t i, std::size_t N, WindowSymmetry type)
    /*!
        This computes the hann window function.
        See https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const R phi = (2.0 * pi * i) / size;
        return 0.5 - 0.5 * std::cos(phi);
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman(std::size_t i, std::size_t N, WindowSymmetry type)
    /*!
        This computes the Blackman window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman_window and
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const R phi = (2.0 * pi * i) / size;
        return 0.42 -
               0.5 * std::cos(phi) +
               0.08 * std::cos(2.0 * phi);
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman_nuttall(std::size_t i, std::size_t N, WindowSymmetry type)
    /*!
        This computes the Blackman-Nuttall window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Nuttall_window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const R phi = (2.0 * pi * i) / size;
        return 0.3635819 -
               0.4891775 * std::cos(phi) +
               0.1365995 * std::cos(2*phi) -
               0.0106411 * std::cos(3*phi);
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman_harris(std::size_t i, std::size_t N, WindowSymmetry type)
    /*!
        This computes the Blackman-Harris window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const R phi = (2.0 * pi * i) / size;
        return 0.35875 -
               0.48829 * std::cos(phi) +
               0.14128 * std::cos(2*phi) -
               0.01168 * std::cos(3*phi);
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman_harris7(std::size_t i, std::size_t N, WindowSymmetry type)
    /*!
        This computes the 7-order Blackman-Harris window function.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const R phi = (2.0 * pi * i) / size;
        return 0.27105 -
               0.43329 * std::cos(phi) +
               0.21812 * std::cos(2*phi) -
               0.06592 * std::cos(3*phi) +
               0.01081 * std::cos(4*phi) -
               0.00077 * std::cos(5*phi) +
               0.00001 * std::cos(6*phi);
    }

    // ----------------------------------------------------------------------------------------

    enum WindowType
    {
        HANN,
        BLACKMAN,
        BLACKMAN_NUTTALL,
        BLACKMAN_HARRIS,
        BLACKMAN_HARRIS7,
        KAISER
    };

    struct window_args
    {
        beta_t beta;
    };

    template<typename R>
    R window(std::size_t i, std::size_t N, WindowType w, WindowSymmetry type, window_args args)
    {
        switch(w)
        {
            case HANN:              return hann<R>(i, N, type);
            case BLACKMAN:          return blackman<R>(i, N, type);
            case BLACKMAN_NUTTALL:  return blackman_nuttall<R>(i, N, type);
            case BLACKMAN_HARRIS:   return blackman_harris<R>(i, N, type);
            case BLACKMAN_HARRIS7:  return blackman_harris7<R>(i, N, type);
            case KAISER:            return kaiser<R>(i, N, args.beta, type);
        }
        DLIB_CASSERT(false, "This should never happen");
        return R{};
    }

    // ----------------------------------------------------------------------------------------
}

#endif //DLIB_MATH_WINDOWS
