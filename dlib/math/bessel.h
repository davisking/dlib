// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_BESSEL
#define DLIB_MATH_BESSEL

#include "details/bessel.h"

namespace dlib
{
    template<typename R1, typename R2>
    inline typename std::common_type<R1,R2>::type cyl_bessel_i(R1 nu, R2 x)
    /*!
        This is the the regular modified cylindrical Bessel function

        ensures
            - identical to std::cyl_bessel_i()
            - works with C++11 onwards
    !*/
    {
        using T = typename std::common_type<R1,R2>::type;
        return detail::cyl_bessel_i<T>(nu, x);
    }

    template<typename R1, typename R2>
    inline typename std::common_type<R1,R2>::type cyl_bessel_j(R1 nu, R2 x)
    /*!
        This is the cylindrical Bessel functions (of the first kind)

        ensures
            - identical to std::cyl_bessel_j()
            - works with C++11 onwards
    !*/
    {
        using T = typename std::common_type<R1,R2>::type;
        return detail::cyl_bessel_j<T>(nu, x);
    }
}

#endif //DLIB_MATH_BESSEL
