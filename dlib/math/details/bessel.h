// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#ifndef DLIB_MATH_DETAIL_BESSEL
#define DLIB_MATH_DETAIL_BESSEL

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include "../../numeric_constants.h"
#include "../../assert.h"

namespace dlib
{
    namespace detail
    {
#if __cpp_lib_math_special_functions
        using std::cyl_bessel_i;
        using std::cyl_bessel_j;
#else
        constexpr unsigned int BESSEL_NITERATIONS = 250;

        template<typename R>
        R cyl_bessel_i(R nu, R x)
        {
            static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
            DLIB_ASSERT(nu >= R{0} && x >= R{0}, "bad arguments. Contract preconditions are : nu >= 0 and x >= 0");

            if (std::isnan(nu) || std::isnan(x))
                return std::numeric_limits<R>::quiet_NaN();

            if(x == 0)
            {
                return (nu == R{0}) ? R{1} : R{0};
            }
            else if(nu == R(0.5))
            {
                // common special case
                return std::sqrt(2 / (x * pi)) * std::sinh(x);
            }
            else
            {
                // Compute sum in log-domain to avoid overflow issues
                const R fact = nu == R{0} ? R{1} : std::pow(R(0.5)*x, nu); //factorize (x/2)^nu
                R a{};
                R b{};
                R c{};
                R sum{0};

                for (unsigned int k=0; k < BESSEL_NITERATIONS; ++k)
                {
                    a = 2 * k * std::log(R(0.5)*x); // log((x/2)^(2k))
                    b = std::lgamma(R(k) + R{1});   // log(k!) = log(gamma(k+1)). Recall gamma(k) = (k-1)!
                    c = std::lgamma(nu + k + 1);
                    sum += std::exp( a - b - c );
                }

                return fact * sum;
            }
        }

        template<typename R>
        R cyl_bessel_j(R nu, R x)
        {
            static_assert(std::is_floating_point<R>::value, "template parameter must be a floating point type");
            DLIB_ASSERT(nu >= R{0} && x >= R{0}, "bad arguments. Contract preconditions are : nu >= 0 and x >= 0");

            if (std::isnan(nu) || std::isnan(x))
                return std::numeric_limits<R>::quiet_NaN();

            /*! Special case !*/
            else if (x == R{0})
            {
                return (nu == R{0}) ? R{1} : R{0};
            }
            else
            {
                // Compute sum in log-domain to avoid overflow issues
                const R fact = nu == R{0} ? R{1} : std::pow(R(0.5)*x, nu); //factorize (x/2)^nu
                R a{};
                R b{};
                R c{};
                R sum{0};

                for (unsigned int k=0; k < BESSEL_NITERATIONS; ++k)
                {
                    a = 2 * k * std::log(R(0.5)*x); // log((x/2)^(2k))
                    b = std::lgamma(R(k) + R{1});   // log(k!) = log(gamma(k+1)). Recall gamma(k) = (k-1)!
                    c = std::lgamma(nu + k + 1);
                    if (k&1)
                        sum -= std::exp(a-b-c);
                    else
                        sum += std::exp(a-b-c);
                }

                return fact * sum;
            }
        }
#endif
    }
}

#endif //DLIB_MATH_DETAIL_BESSEL
