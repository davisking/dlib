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
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a strong type representing an array index.
                It is suitable for distinguishing which overload of the kaiser()
                function should be used.
        !*/
        explicit index_t(std::size_t i_) : i{i_} {}
        std::size_t i = 0;
    };

    struct window_duration
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This ojbect is a strong type representing the window duration of a kaiser window.
                See https://en.wikipedia.org/wiki/Kaiser_window.
        !*/
        explicit window_duration(double L_) : L{L_} {}
        double L = 0.0;
    };

    struct window_length
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This ojbect is a strong type representing the window length of a kaiser window.
                See https://en.wikipedia.org/wiki/Kaiser_window.
        !*/
        explicit window_length(std::size_t N_) : N{N_} {}
        std::size_t N = 0;
    };

    struct symmetric_t
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a strong type that signifies that the window is a symmetric window.
                See https://en.wikipedia.org/wiki/Window_function#Symmetry for a discussion on
                symmetric vs periodic windows. This is using the same nomenclature as Matlab and Scipy
                when describing windows as either symmetric or periodic.
        !*/
    };

    struct periodic_t
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a strong type that signifies that the window is a periodic window.
                See https://en.wikipedia.org/wiki/Window_function#Symmetry for a discussion on
                symmetric vs periodic windows. This is using the same nomenclature as Matlab and Scipy
                when describing windows as either symmetric or periodic.
        !*/
    };

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R kaiser(R x, window_duration L, beta_t beta)
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

        if (std::abs(x) <= L.L/R{2})
        {
            const R r = 2*x/L.L;
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
    R kaiser(index_t i, window_length N, beta_t beta, symmetric_t)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window
        This variant is a short-cut for computing a window function and storing it
        in an array of size N where 0 <= i < N is the array index.
        This is the symmetric version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
        ensures
            - returns kaiser(i - (N-1)/2, window_duration{N.N-1}, beta)
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i.i < N.N, "index out of range");
        return kaiser(R(i.i) - R(N.N-1) / R(2), window_duration{R(N.N-1)}, beta);
    }

    template<typename R>
    R kaiser(index_t i, window_length N, beta_t beta, periodic_t)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window
        This variant is a short-cut for computing a window function and storing it
        in an array of size N where 0 <= i < N is the array index.
        This is the periodic version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        return kaiser<R>(i, window_length{N.N+1}, beta, symmetric_t{});
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R hann(index_t i, window_length N, symmetric_t)
    /*!
        This computes the hann window function.
        See https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows.
        This variant computes a symmetric window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i.i < N.N, "index out of range");
        const R phi = (2.0 * pi * i.i) / (N.N - 1);
        return 0.5 - 0.5 * std::cos(phi);
    }

    template<typename R>
    R hann(index_t i, window_length N, periodic_t)
    /*!
        This computes the hann window function.
        See https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows.
        This variant computes a periodic window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        return hann<R>(i, window_length{N.N+1}, symmetric_t{});
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman(index_t i, window_length N, symmetric_t)
    /*!
        This computes the Blackman window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman_window and
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html.
        This variant computes a symmetric window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i.i < N.N, "index out of range");
        const R phi = (2.0 * pi * i.i) / (N.N - 1);
        return 0.42 -
               0.5 * std::cos(phi) +
               0.08 * std::cos(2.0 * phi);
    }

    template<typename R>
    R blackman(index_t i, window_length N, periodic_t)
    /*!
        This computes the Blackman window function.
        This variant computes a periodic window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        return blackman<R>(i, window_length{N.N+1}, symmetric_t{});
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman_nuttall(index_t i, window_length N, symmetric_t)
    /*!
        This computes the Blackman-Nuttall window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Nuttall_window.
        This is the symmetric version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i.i < N.N, "index out of range");
        const R phi = (2.0 * pi * i.i) / (N.N - 1);
        return 0.3635819 -
               0.4891775 * std::cos(phi) +
               0.1365995 * std::cos(2*phi) -
               0.0106411 * std::cos(3*phi);
    }

    template<typename R>
    R blackman_nuttall(index_t i, window_length N, periodic_t)
    /*!
        This computes the Blackman-Nuttall window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Nuttall_window.
        This is the periodic version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        return blackman_nuttall<R>(i, window_length{N.N+1}, symmetric_t{});
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman_harris(index_t i, window_length N, symmetric_t)
    /*!
        This computes the Blackman-Harris window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window.
        This is the symmetric version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i.i < N.N, "index out of range");
        const R phi = (2.0 * pi * i.i) / (N.N - 1);
        return 0.35875 -
               0.48829 * std::cos(phi) +
               0.14128 * std::cos(2*phi) -
               0.01168 * std::cos(3*phi);
    }

    template<typename R>
    R blackman_harris(index_t i, window_length N, periodic_t)
    /*!
        This computes the Blackman-Harris window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window.
        This is the periodic version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        return blackman_harris<R>(i, window_length{N.N+1}, symmetric_t{});
    }

    // ----------------------------------------------------------------------------------------

    template<typename R>
    R blackman_harris7(index_t i, window_length N, symmetric_t)
    /*!
        This computes the 7-order Blackman-Harris window function.
        This is the symmetric version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
        DLIB_ASSERT(i.i < N.N, "index out of range");
        const R phi = (2.0 * pi * i.i) / (N.N - 1);
        return 0.27105 -
               0.43329 * std::cos(phi) +
               0.21812 * std::cos(2*phi) -
               0.06592 * std::cos(3*phi) +
               0.01081 * std::cos(4*phi) -
               0.00077 * std::cos(5*phi) +
               0.00001 * std::cos(6*phi);
    }

    template<typename R>
    R blackman_harris7(index_t i, window_length N, periodic_t)
    /*!
        This computes the 7-order Blackman-Harris window function.
        This is the periodic version.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        return blackman_harris7<R>(i, window_length{N.N+1}, symmetric_t{});
    }

    // ----------------------------------------------------------------------------------------
}

#endif //DLIB_MATH_WINDOWS
