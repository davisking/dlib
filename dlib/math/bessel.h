// Copyright (C) 2006  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATH_BESSEL
#define DLIB_MATH_BESSEL

namespace dlib
{
#if __cpp_lib_math_special_functions
    using std::cyl_bessel_i;
    using std::cyl_bessel_j;
#else
#endif
}

#endif //DLIB_MATH_BESSEL
