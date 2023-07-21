// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SiG_TRAITS_Hh_
#define DLIB_SiG_TRAITS_Hh_

namespace dlib
{

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

}

#endif // DLIB_AnY_FUNCTION_Hh_
