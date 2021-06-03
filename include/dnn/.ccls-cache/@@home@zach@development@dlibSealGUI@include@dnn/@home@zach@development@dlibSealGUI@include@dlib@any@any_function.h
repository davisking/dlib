// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_FUNCTION_Hh_
#define DLIB_AnY_FUNCTION_Hh_

#include "any.h"

#include "any_function_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct sig_traits {};

    template <
        typename T
        >
    struct sig_traits<T ()> 
    { 
        typedef T result_type; 
        typedef void arg1_type; 
        typedef void arg2_type; 
        typedef void arg3_type; 
        typedef void arg4_type; 
        typedef void arg5_type; 
        typedef void arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 0;
    };

    template <
        typename T,
        typename A1
        >
    struct sig_traits<T (A1)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef void arg2_type; 
        typedef void arg3_type; 
        typedef void arg4_type; 
        typedef void arg5_type; 
        typedef void arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 1;
    };

    template <
        typename T,
        typename A1, typename A2
        >
    struct sig_traits<T (A1,A2)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef void arg3_type; 
        typedef void arg4_type; 
        typedef void arg5_type; 
        typedef void arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 2;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3
        >
    struct sig_traits<T (A1,A2,A3)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef void arg4_type; 
        typedef void arg5_type; 
        typedef void arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 3;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4
        >
    struct sig_traits<T (A1,A2,A3,A4)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef void arg5_type; 
        typedef void arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 4;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5
        >
    struct sig_traits<T (A1,A2,A3,A4,A5)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef void arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 5;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef void arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 6;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef void arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 7;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef void arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 8;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef void arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 9;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef void arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 10;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef void arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 11;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef void arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 12;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef void arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 13;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef void arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 14;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef A15 arg15_type; 
        typedef void arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 15;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15,
        typename A16
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef A15 arg15_type; 
        typedef A16 arg16_type; 
        typedef void arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 16;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15,
        typename A16,
        typename A17
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef A15 arg15_type; 
        typedef A16 arg16_type; 
        typedef A17 arg17_type; 
        typedef void arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 17;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15,
        typename A16,
        typename A17,
        typename A18
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef A15 arg15_type; 
        typedef A16 arg16_type; 
        typedef A17 arg17_type; 
        typedef A18 arg18_type; 
        typedef void arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 18;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15,
        typename A16,
        typename A17,
        typename A18,
        typename A19
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef A15 arg15_type; 
        typedef A16 arg16_type; 
        typedef A17 arg17_type; 
        typedef A18 arg18_type; 
        typedef A19 arg19_type; 
        typedef void arg20_type; 

        const static unsigned long num_args = 19;
    };

    template <
        typename T,
        typename A1, typename A2, typename A3,
        typename A4, typename A5, typename A6,
        typename A7, typename A8, typename A9,
        typename A10,
        typename A11,
        typename A12,
        typename A13,
        typename A14,
        typename A15,
        typename A16,
        typename A17,
        typename A18,
        typename A19,
        typename A20
        >
    struct sig_traits<T (A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,A12,A13,A14,A15,A16,A17,A18,A19,A20)> 
    { 
        typedef T result_type; 
        typedef A1 arg1_type; 
        typedef A2 arg2_type; 
        typedef A3 arg3_type; 
        typedef A4 arg4_type; 
        typedef A5 arg5_type; 
        typedef A6 arg6_type; 
        typedef A7 arg7_type; 
        typedef A8 arg8_type; 
        typedef A9 arg9_type; 
        typedef A10 arg10_type; 
        typedef A11 arg11_type; 
        typedef A12 arg12_type; 
        typedef A13 arg13_type; 
        typedef A14 arg14_type; 
        typedef A15 arg15_type; 
        typedef A16 arg16_type; 
        typedef A17 arg17_type; 
        typedef A18 arg18_type; 
        typedef A19 arg19_type; 
        typedef A20 arg20_type; 

        const static unsigned long num_args = 20;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename function_type, 
        // These arguments are used to control the overloading.  A user should
        // not mess with them.  
        typename Enabled = void,
        unsigned long Num_args = sig_traits<function_type>::num_args
        >
    class any_function
    {
    private:
        any_function() {}
    /* !!!!!!!!    ERRORS ON THE ABOVE LINE    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        An error on this line means you are trying to use a function signature
        with more than the supported number of arguments.  The current version
        of dlib only supports up to 10 arguments.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/
    };


    // The following preprocessor commands build the various overloaded versions
    // of any_function for different numbers of commands and void vs. non-void return
    // types.

//  0 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST
#define DLIB_ANY_FUNCTION_ARGS 
#define DLIB_ANY_FUNCTION_NUM_ARGS 0
#include "any_function_impl2.h"

//  1 argument
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1
#define DLIB_ANY_FUNCTION_ARGS a1
#define DLIB_ANY_FUNCTION_NUM_ARGS 1
#include "any_function_impl2.h"

//  2 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2
#define DLIB_ANY_FUNCTION_ARGS a1,a2
#define DLIB_ANY_FUNCTION_NUM_ARGS 2
#include "any_function_impl2.h"

//  3 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3
#define DLIB_ANY_FUNCTION_NUM_ARGS 3
#include "any_function_impl2.h"

//  4 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4
#define DLIB_ANY_FUNCTION_NUM_ARGS 4
#include "any_function_impl2.h"

//  5 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4, \
                                   arg5_type a5
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4,a5
#define DLIB_ANY_FUNCTION_NUM_ARGS 5
#include "any_function_impl2.h"

//  6 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4, \
                                   arg5_type a5, arg6_type a6
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4,a5,a6
#define DLIB_ANY_FUNCTION_NUM_ARGS 6
#include "any_function_impl2.h"

//  7 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4, \
                                   arg5_type a5, arg6_type a6, arg7_type a7
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4,a5,a6,a7
#define DLIB_ANY_FUNCTION_NUM_ARGS 7
#include "any_function_impl2.h"

//  8 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4, \
                                   arg5_type a5, arg6_type a6, arg7_type a7, arg8_type a8
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4,a5,a6,a7,a8
#define DLIB_ANY_FUNCTION_NUM_ARGS 8
#include "any_function_impl2.h"

//  9 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4, \
                                   arg5_type a5, arg6_type a6, arg7_type a7, arg8_type a8, \
                                   arg9_type a9
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4,a5,a6,a7,a8,a9
#define DLIB_ANY_FUNCTION_NUM_ARGS 9
#include "any_function_impl2.h"

//  10 arguments
#define DLIB_ANY_FUNCTION_ARG_LIST arg1_type a1, arg2_type a2, arg3_type a3, arg4_type a4, \
                                   arg5_type a5, arg6_type a6, arg7_type a7, arg8_type a8, \
                                   arg9_type a9, arg10_type a10
#define DLIB_ANY_FUNCTION_ARGS a1,a2,a3,a4,a5,a6,a7,a8,a9,a10
#define DLIB_ANY_FUNCTION_NUM_ARGS 10 
#include "any_function_impl2.h"

// ----------------------------------------------------------------------------------------

    template <typename function_type>
    inline void swap (
        any_function<function_type>& a,
        any_function<function_type>& b
    ) { a.swap(b); }

// ----------------------------------------------------------------------------------------

    template <typename T, typename function_type> 
    T& any_cast(any_function<function_type>& a) { return a.template cast_to<T>(); }

    template <typename T, typename function_type> 
    const T& any_cast(const any_function<function_type>& a) { return a.template cast_to<T>(); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_AnY_FUNCTION_Hh_

