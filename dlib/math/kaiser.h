// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_KAISER
#define DLIB_MATH_KAISER

#include "bessel.h"

namespace dlib
{
    template<typename R>
    inline R attenuation_to_beta(R attenuation)
    {
        R beta{0};
        if (attenuation > 50.0)
            beta = 0.1102*(attenuation - 8.7);
        else if (attenuation >= 21.0)
            beta = 0.5842*std::pow(attenuation - 21, 0.4) + 0.07886*(attenuation - 21);
        return beta;
    }

    template<typename R>
    inline R kaiser_r(R x, R L, R beta)
    {
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
    {
        return kaiser_r(R(i) - R(N-1) / R(2), R(N-1), beta);
    }
}

#endif //DLIB_MATH_KAISER
