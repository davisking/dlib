// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMd_Hh_
#define DLIB_SIMd_Hh_

#include "simd/simd_check.h"


#ifndef DLIB_HAVE_ALTIVEC
#include "simd/simd4f.h"
#include "simd/simd4i.h"
#include "simd/simd8f.h"
#include "simd/simd8i.h"
#else
#include "simd/simd4f_vec.h"
#include "simd/simd4i_vec.h"
#include "simd/simd8f_vec.h"
#include "simd/simd8i_vec.h"
#endif

#endif // DLIB_SIMd_Hh_

