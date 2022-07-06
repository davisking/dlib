// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_BESSEL
#define DLIB_MATH_BESSEL

#ifndef __cpp_lib_math_special_functions
#include "details/bessel.h"
#endif

namespace dlib
{
#if __cpp_lib_math_special_functions
    using std::cyl_bessel_i;
    using std::cyl_bessel_j;
#else

    template<typename R>
    R cyl_bessel_i(R nu, R x)
    {
        return detail::cyl_bessel_i(nu, x);
    }

    template<typename R>
    R cyl_bessel_j(R nu, R x)
    {
        return detail::cyl_bessel_j(nu, x);
    }
#endif
}

#endif //DLIB_MATH_BESSEL
