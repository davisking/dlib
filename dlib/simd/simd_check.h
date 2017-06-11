// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMd_CHECK_Hh_
#define DLIB_SIMd_CHECK_Hh_

//#define DLIB_DO_NOT_USE_SIMD

// figure out which SIMD instructions we can use.
#ifndef DLIB_DO_NOT_USE_SIMD
    #if defined(_MSC_VER) 
        #ifdef __AVX__
            #ifndef DLIB_HAVE_SSE2
                #define DLIB_HAVE_SSE2
            #endif 
            #ifndef DLIB_HAVE_SSE3
                #define DLIB_HAVE_SSE3
            #endif
            #ifndef DLIB_HAVE_SSE41
                #define DLIB_HAVE_SSE41
            #endif
            #ifndef DLIB_HAVE_AVX
                #define DLIB_HAVE_AVX
            #endif
        #endif
        #if (defined( _M_X64) || defined(_M_IX86_FP) && _M_IX86_FP >= 2) && !defined(DLIB_HAVE_SSE2)
            #define DLIB_HAVE_SSE2
        #endif
    #else
        #ifdef __SSE2__
            #ifndef DLIB_HAVE_SSE2
                #define DLIB_HAVE_SSE2
            #endif 
        #endif
        #ifdef __SSSE3__
            #ifndef DLIB_HAVE_SSE3
                #define DLIB_HAVE_SSE3
            #endif
        #endif
        #ifdef __SSE4_1__
            #ifndef DLIB_HAVE_SSE41
                #define DLIB_HAVE_SSE41
            #endif
        #endif
        #ifdef __AVX__
            #ifndef DLIB_HAVE_AVX
                #define DLIB_HAVE_AVX
            #endif
        #endif
        #ifdef __AVX2__
            #ifndef DLIB_HAVE_AVX2
                #define DLIB_HAVE_AVX2
            #endif
        #endif
        #ifdef __ALTIVEC__
            #ifndef DLIB_HAVE_ALTIVEC
                #define DLIB_HAVE_ALTIVEC
            #endif
        #endif
        #ifdef __VSX__
            #ifndef DLIB_HAVE_VSX
                #define DLIB_HAVE_VSX
            #endif
        #endif
        #ifdef __VEC__ // __VEC__ = 10206
            #ifndef DLIB_HAVE_POWER_VEC	// vector and vec_ intrinsics
                #define DLIB_HAVE_POWER_VEC
            #endif
        #endif
        #ifdef __ARM_NEON
            #ifndef DLIB_HAVE_NEON
                #define DLIB_HAVE_NEON
            #endif
        #endif
    #endif
#endif

 
// ----------------------------------------------------------------------------------------


#ifdef DLIB_HAVE_ALTIVEC
#include <altivec.h>
#endif

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
    #include <immintrin.h> // AVX
//    #include <avx2intrin.h>
#endif
#ifdef DLIB_HAVE_NEON
    #include <arm_neon.h> // ARM NEON
#endif


#endif // DLIB_SIMd_CHECK_Hh_


