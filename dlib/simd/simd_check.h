// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMd_CHECK_H__
#define DLIB_SIMd_CHECK_H__

//#define DLIB_DO_NOT_USE_SIMD

// figure out which SIMD instructions we can use.
#ifndef DLIB_DO_NOT_USE_SIMD
    #if defined(_MSC_VER) 
        #ifdef __AVX__
            #define DLIB_HAVE_SSE2
            #define DLIB_HAVE_SSE3
            #define DLIB_HAVE_SSE41
            #define DLIB_HAVE_AVX
        #endif
        #if defined(_M_IX86_FP) && _M_IX86_FP >= 2 && !defined(DLIB_HAVE_SSE2)
            #define DLIB_HAVE_SSE2
        #endif
    #else
        #ifdef __SSE2__
            #define DLIB_HAVE_SSE2
        #endif
        #ifdef __SSSE3__
            #define DLIB_HAVE_SSE3
        #endif
        #ifdef __SSE4_1__
            #define DLIB_HAVE_SSE41
        #endif
        #ifdef __AVX__
            #define DLIB_HAVE_AVX
        #endif
        #ifdef __AVX2__
            #define DLIB_HAVE_AVX2
        #endif
    #endif
#endif

 
// ----------------------------------------------------------------------------------------

#ifdef DLIB_HAVE_SSE2
    #include <xmmintrin.h>
    #include <emmintrin.h>
    #include <mmintrin.h>
#endif
#ifdef DLIB_HAVE_SSE3
    #include <pmmintrin.h> // SSE3
    #include <tmmintrin.h>
#endif
#ifdef DLIB_HAVE_SSE41
    #include <smmintrin.h> // SSE4
#endif
#ifdef DLIB_HAVE_AVX
    #include <immintrin.h> // AVX
#endif
#ifdef DLIB_HAVE_AVX2
    #include <avx2intrin.h>
#endif

#endif // DLIB_SIMd_CHECK_H__


