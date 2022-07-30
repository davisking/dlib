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

    enum window_symmetry
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

    inline double kaiser(double x, double L, beta_t beta)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window.

        ensures
            - returns the kaiser window function when |x| <= L/2 where L is the window duration
            - returns 0 otherwise
    !*/
    {
        if (std::abs(x) <= L/2.0)
        {
            const double r = 2*x/L;
            const double a = dlib::cyl_bessel_i(0, beta.v*std::sqrt(1-r*r));
            const double b = dlib::cyl_bessel_i(0, beta.v);
            return a / b;
        }
        else
        {
            return 0.0;
        }
    }

    inline double kaiser(std::size_t i, std::size_t N, beta_t beta, window_symmetry type)
    /*!
        This computes the kaiser window function or kaiser-bessel window function.
        See https://en.wikipedia.org/wiki/Kaiser_window
        This variant is a short-cut for computing a window function and storing it
        in an array of size N where 0 <= i < N is the array index.

        requires
            - 0 <= i < N
        ensures
            - returns kaiser(i - (N-1)/2, window_duration{N-1}, beta)
    !*/
    {
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t L = type == SYMMETRIC ? N-1 : N;
        return kaiser(i - L / 2.0, (double)L, beta);
    }

    // ----------------------------------------------------------------------------------------

    inline double hann(std::size_t i, std::size_t N, window_symmetry type)
    /*!
        This computes the hann window function.
        See https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows.

        requires
            - 0 <= i < N
    !*/
    {
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const double phi = (2.0 * pi * i) / size;
        return 0.5 - 0.5 * std::cos(phi);
    }

    // ----------------------------------------------------------------------------------------

    inline double blackman(std::size_t i, std::size_t N, window_symmetry type)
    /*!
        This computes the Blackman window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman_window and
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.windows.blackman.html.

        requires
            - 0 <= i < N
    !*/
    {
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const double phi = (2.0 * pi * i) / size;
        return 0.42 -
               0.5 * std::cos(phi) +
               0.08 * std::cos(2.0 * phi);
    }

    // ----------------------------------------------------------------------------------------

    inline double blackman_nuttall(std::size_t i, std::size_t N, window_symmetry type)
    /*!
        This computes the Blackman-Nuttall window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Nuttall_window.

        requires
            - 0 <= i < N
    !*/
    {
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const double phi = (2.0 * pi * i) / size;
        return 0.3635819 -
               0.4891775 * std::cos(phi) +
               0.1365995 * std::cos(2*phi) -
               0.0106411 * std::cos(3*phi);
    }

    // ----------------------------------------------------------------------------------------

    inline double blackman_harris(std::size_t i, std::size_t N, window_symmetry type)
    /*!
        This computes the Blackman-Harris window function.
        See https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const double phi = (2.0 * pi * i) / size;
        return 0.35875 -
               0.48829 * std::cos(phi) +
               0.14128 * std::cos(2*phi) -
               0.01168 * std::cos(3*phi);
    }

    // ----------------------------------------------------------------------------------------

    inline double blackman_harris7(std::size_t i, std::size_t N, window_symmetry type)
    /*!
        This computes the 7-order Blackman-Harris window function.

        requires
            - R is float, double, or long double
            - 0 <= i < N
    !*/
    {
        DLIB_ASSERT(i < N, "index out of range");
        const std::size_t size = type == SYMMETRIC ? N-1 : N;
        const double phi = (2.0 * pi * i) / size;
        return 0.27105 -
               0.43329 * std::cos(phi) +
               0.21812 * std::cos(2*phi) -
               0.06592 * std::cos(3*phi) +
               0.01081 * std::cos(4*phi) -
               0.00077 * std::cos(5*phi) +
               0.00001 * std::cos(6*phi);
    }

    // ----------------------------------------------------------------------------------------
}

#endif //DLIB_MATH_WINDOWS
