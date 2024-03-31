// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMd_CHECK_Hh_
#define DLIB_SIMd_CHECK_Hh_

#include <array>
#include <iostream>

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
#undef bool
#undef vector
#undef pixel
#endif

#ifdef DLIB_HAVE_SSE2
    #include <xmmintrin.h>
    #include <emmintrin.h>
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

// ----------------------------------------------------------------------------------------
// Define functions to check, at runtime, what instructions are available

#if defined(_MSC_VER) && (defined(_M_I86) || defined(_M_IX86) || defined(_M_X64) || defined(_M_AMD64) )
    #include <intrin.h>

    inline std::array<unsigned int,4> cpuid(int function_id) 
    { 
        std::array<unsigned int,4> info;
        // Load EAX, EBX, ECX, EDX into info
        __cpuid((int*)info.data(), function_id);
        return info;
    }

#elif (defined(__GNUC__) || defined(__clang__)) && (defined(__i386__) || defined(__i686__) || defined(__amd64__) || defined(__x86_64__))
    #include <cpuid.h>

    inline std::array<unsigned int,4> cpuid(int function_id) 
    { 
        std::array<unsigned int,4> info;
        // Load EAX, EBX, ECX, EDX into info
        __cpuid_count(function_id, 0, info[0], info[1], info[2], info[3]);
        return info;
    }

#else

    inline std::array<unsigned int,4> cpuid(int) 
    {
        std::array<unsigned int,4> info = { 0 };
        return info;
    }

#endif

    inline bool cpu_has_sse2_instructions()   { return 0!=(cpuid(1)[3]&(1<<26)); }
    inline bool cpu_has_sse3_instructions()   { return 0!=(cpuid(1)[2]&(1<<0));  }
    inline bool cpu_has_sse41_instructions()  { return 0!=(cpuid(1)[2]&(1<<19)); }
    inline bool cpu_has_sse42_instructions()  { return 0!=(cpuid(1)[2]&(1<<20)); }
    inline bool cpu_has_avx_instructions()    { return 0!=(cpuid(1)[2]&(1<<28)); }
    inline bool cpu_has_avx2_instructions()   { return 0!=(cpuid(7)[1]&(1<<5));  }
    inline bool cpu_has_avx512_instructions() { return 0!=(cpuid(7)[1]&(1<<16)); }

    inline void warn_about_unavailable_but_used_cpu_instructions()
    {
#if defined(DLIB_HAVE_AVX2)
        if (!cpu_has_avx2_instructions())
            std::cerr << "Dlib was compiled to use AVX2 instructions, but these aren't available on your machine." << std::endl;
#elif defined(DLIB_HAVE_AVX)
        if (!cpu_has_avx_instructions())
            std::cerr << "Dlib was compiled to use AVX instructions, but these aren't available on your machine." << std::endl;
#elif defined(DLIB_HAVE_SSE41)
        if (!cpu_has_sse41_instructions())
            std::cerr << "Dlib was compiled to use SSE41 instructions, but these aren't available on your machine." << std::endl;
#elif defined(DLIB_HAVE_SSE3)
        if (!cpu_has_sse3_instructions())
            std::cerr << "Dlib was compiled to use SSE3 instructions, but these aren't available on your machine." << std::endl;
#elif defined(DLIB_HAVE_SSE2)
        if (!cpu_has_sse2_instructions())
            std::cerr << "Dlib was compiled to use SSE2 instructions, but these aren't available on your machine." << std::endl;
#endif
    }

// ----------------------------------------------------------------------------------------

#endif // DLIB_SIMd_CHECK_Hh_


