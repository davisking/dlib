/*
 *  Name:        wx/cpp.h
 *  Purpose:     Various preprocessor helpers
 *  Author:      Vadim Zeitlin
 *  Created:     2006-09-30
 *  Copyright:   (c) 2006 Vadim Zeitlin <vadim@wxwidgets.org>
 *  Licence:     wxWindows licence
 */

/* THIS IS A C FILE, DON'T USE C++ FEATURES (IN PARTICULAR COMMENTS) IN IT */

#ifndef _WX_CPP_H_
#define _WX_CPP_H_

#include "wx/compiler.h"    /* wxCHECK_XXX_VERSION() macros */

/* wxCONCAT works like preprocessor ## operator but also works with macros */
#define wxCONCAT_HELPER(text, line) text ## line

#define wxCONCAT(x1, x2) \
    wxCONCAT_HELPER(x1, x2)
#define wxCONCAT3(x1, x2, x3) \
    wxCONCAT(wxCONCAT(x1, x2), x3)
#define wxCONCAT4(x1, x2, x3, x4) \
    wxCONCAT(wxCONCAT3(x1, x2, x3), x4)
#define wxCONCAT5(x1, x2, x3, x4, x5) \
    wxCONCAT(wxCONCAT4(x1, x2, x3, x4), x5)
#define wxCONCAT6(x1, x2, x3, x4, x5, x6) \
    wxCONCAT(wxCONCAT5(x1, x2, x3, x4, x5), x6)
#define wxCONCAT7(x1, x2, x3, x4, x5, x6, x7) \
    wxCONCAT(wxCONCAT6(x1, x2, x3, x4, x5, x6), x7)
#define wxCONCAT8(x1, x2, x3, x4, x5, x6, x7, x8) \
    wxCONCAT(wxCONCAT7(x1, x2, x3, x4, x5, x6, x7), x8)
#define wxCONCAT9(x1, x2, x3, x4, x5, x6, x7, x8, x9) \
    wxCONCAT(wxCONCAT8(x1, x2, x3, x4, x5, x6, x7, x8), x9)

/* wxSTRINGIZE works as the preprocessor # operator but also works with macros */
#define wxSTRINGIZE_HELPER(x)       #x
#define wxSTRINGIZE(x)              wxSTRINGIZE_HELPER(x)

/* a Unicode-friendly version of wxSTRINGIZE_T */
#define wxSTRINGIZE_T(x)            wxAPPLY_T(wxSTRINGIZE(x))

/*
   Helper macros for wxMAKE_UNIQUE_NAME: normally this works by appending the
   current line number to the given identifier to reduce the probability of the
   conflict (it may still happen if this is used in the headers, hence you
   should avoid doing it or provide unique prefixes then) but we have to do it
   differently for VC++
  */
#if defined(__VISUALC__)
    /*
       __LINE__ handling is completely broken in VC++ when using "Edit and
       Continue" (/ZI option) and results in preprocessor errors if we use it
       inside the macros. Luckily VC7 has another standard macro which can be
       used like this and is even better than __LINE__ because it is globally
       unique.
     */
#   define wxCONCAT_LINE(text)         wxCONCAT(text, __COUNTER__)
#else /* normal compilers */
#   define wxCONCAT_LINE(text)         wxCONCAT(text, __LINE__)
#endif

/* Create a "unique" name with the given prefix */
#define wxMAKE_UNIQUE_NAME(text)    wxCONCAT_LINE(text)

/*
   This macro can be passed as argument to another macro when you don't have
   anything to pass in fact.
 */
#define wxEMPTY_PARAMETER_VALUE /* Fake macro parameter value */

/*
    Helpers for defining macros that expand into a single statement.

    The standard solution is to use "do { ... } while (0)" statement but MSVC
    generates a C4127 "condition expression is constant" warning for it so we
    use something which is just complicated enough to not be recognized as a
    constant but still simple enough to be optimized away.

    Another solution would be to use __pragma() to temporarily disable C4127.

    Notice that wxASSERT_ARG_TYPE in wx/strvargarg.h relies on these macros
    creating some kind of a loop because it uses "break".
 */
#define wxSTATEMENT_MACRO_BEGIN  do {
#define wxSTATEMENT_MACRO_END } while ( (void)0, 0 )

/*
    Helper for executing the following statement conditionally without using
    conditional statements.

    This strange macro is needed in the first place to avoid the problems due
    to nested if/else inside macros. E.g. if some MACRO started with "if", then

        if ( cond )
            MACRO();
        else
            ...

    would be broken because "..." would bind to the wrong "if" inside the macro
    rather than the visible one. So we use wxDO_IF() inside the macro instead
    to avoid this problem.
 */
#define wxDO_IF_HELPER(loopvar, condition)                                    \
    for ( bool loopvar = false; !loopvar && condition; loopvar = true )

#define wxDO_IF(condition) wxDO_IF_HELPER(wxMAKE_UNIQUE_NAME(wxdoif), condition)

/*
    Define __WXFUNCTION__ which is like standard __FUNCTION__ but defined as
    NULL for the compilers which don't support the latter.
 */
#ifndef __WXFUNCTION__
    #if defined(__GNUC__) || \
          defined(__VISUALC__) || \
          defined(__FUNCTION__)
        #define __WXFUNCTION__ __FUNCTION__
    #else
        /* still define __WXFUNCTION__ to avoid #ifdefs elsewhere */
        #define __WXFUNCTION__ (NULL)
    #endif
#endif /* __WXFUNCTION__ already defined */


/* Auto-detect variadic macros support unless explicitly disabled. */
#if !defined(HAVE_VARIADIC_MACROS) && !defined(wxNO_VARIADIC_MACROS)
    /* Any C99 or C++11 compiler should have them. */
    #if (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) || \
        (defined(__cplusplus) && __cplusplus >= 201103L)
        #define HAVE_VARIADIC_MACROS 1
    #elif defined(__GNUC__)
        #define HAVE_VARIADIC_MACROS 1
    #elif wxCHECK_VISUALC_VERSION(8)
        #define HAVE_VARIADIC_MACROS 1
    #endif
#endif /* !HAVE_VARIADIC_MACROS */



#ifdef HAVE_VARIADIC_MACROS

/*
   wxCALL_FOR_EACH(what, ...) calls the macro from its first argument, what(pos, x),
   for every remaining argument 'x', with 'pos' being its 1-based index in
   *reverse* order (with the last argument being numbered 1).

   For example, wxCALL_FOR_EACH(test, a, b, c) expands into this:

       test(3, a) \
       test(2, b) \
       test(1, c)

   Up to eight arguments are supported.

   (With thanks to https://groups.google.com/d/topic/comp.std.c/d-6Mj5Lko_s/discussion
   and https://stackoverflow.com/questions/1872220/is-it-possible-to-iterate-over-arguments-in-variadic-macros)
*/
#define wxCALL_FOR_EACH_NARG(...)   wxCALL_FOR_EACH_NARG_((__VA_ARGS__, wxCALL_FOR_EACH_RSEQ_N()))
#define wxCALL_FOR_EACH_NARG_(args) wxCALL_FOR_EACH_ARG_N args
#define wxCALL_FOR_EACH_ARG_N(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define wxCALL_FOR_EACH_RSEQ_N() 8, 7, 6, 5, 4, 3, 2, 1, 0

#define wxCALL_FOR_EACH_1_(args)   wxCALL_FOR_EACH_1 args
#define wxCALL_FOR_EACH_2_(args)   wxCALL_FOR_EACH_2 args
#define wxCALL_FOR_EACH_3_(args)   wxCALL_FOR_EACH_3 args
#define wxCALL_FOR_EACH_4_(args)   wxCALL_FOR_EACH_4 args
#define wxCALL_FOR_EACH_5_(args)   wxCALL_FOR_EACH_5 args
#define wxCALL_FOR_EACH_6_(args)   wxCALL_FOR_EACH_6 args
#define wxCALL_FOR_EACH_7_(args)   wxCALL_FOR_EACH_7 args
#define wxCALL_FOR_EACH_8_(args)   wxCALL_FOR_EACH_8 args

#define wxCALL_FOR_EACH_1(what, x)        what(1, x)
#define wxCALL_FOR_EACH_2(what, x, ...)   what(2, x)  wxCALL_FOR_EACH_1_((what, __VA_ARGS__))
#define wxCALL_FOR_EACH_3(what, x, ...)   what(3, x)  wxCALL_FOR_EACH_2_((what, __VA_ARGS__))
#define wxCALL_FOR_EACH_4(what, x, ...)   what(4, x)  wxCALL_FOR_EACH_3_((what, __VA_ARGS__))
#define wxCALL_FOR_EACH_5(what, x, ...)   what(5, x)  wxCALL_FOR_EACH_4_((what, __VA_ARGS__))
#define wxCALL_FOR_EACH_6(what, x, ...)   what(6, x)  wxCALL_FOR_EACH_5_((what, __VA_ARGS__))
#define wxCALL_FOR_EACH_7(what, x, ...)   what(7, x)  wxCALL_FOR_EACH_6_((what, __VA_ARGS__))
#define wxCALL_FOR_EACH_8(what, x, ...)   what(8, x)  wxCALL_FOR_EACH_7_((what, __VA_ARGS__))

#define wxCALL_FOR_EACH_(N, args) \
    wxCONCAT(wxCALL_FOR_EACH_, N) args

#define wxCALL_FOR_EACH(what, ...) \
    wxCALL_FOR_EACH_(wxCALL_FOR_EACH_NARG(__VA_ARGS__), (what, __VA_ARGS__))

#else
    #define wxCALL_FOR_EACH  Error_wx_CALL_FOR_EACH_requires_variadic_macros_support
#endif /* HAVE_VARIADIC_MACROS */

#endif /* _WX_CPP_H_ */

