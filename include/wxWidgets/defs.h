/*
 *  Name:        wx/defs.h
 *  Purpose:     Declarations/definitions common to all wx source files
 *  Author:      Julian Smart and others
 *  Modified by: Ryan Norton (Converted to C)
 *  Created:     01/02/97
 *  Copyright:   (c) Julian Smart
 *  Licence:     wxWindows licence
 */

/* THIS IS A C FILE, DON'T USE C++ FEATURES (IN PARTICULAR COMMENTS) IN IT */

/*
    We want to avoid compilation and, even more perniciously, link errors if
    the user code includes <windows.h> before include wxWidgets headers. These
    error happen because <windows.h> #define's many common symbols, such as
    Yield or GetClassInfo, which are also used in wxWidgets API. Including our
    "cleanup" header below un-#defines them to fix this.

    Moreover, notice that it is also possible for the user code to include some
    wx header (this including wx/defs.h), then include <windows.h> and then
    include another wx header. To avoid the problem for the second header
    inclusion, we must include wx/msw/winundef.h from here always and not just
    during the first inclusion, so it has to be outside of _WX_DEFS_H_ guard
    check below.
 */
#ifdef __cplusplus
    /*
        Test for _WINDOWS_, used as header guard by windows.h itself, not our
        own __WINDOWS__, which is not defined yet.
     */
#   ifdef _WINDOWS_
#       include "wx/msw/winundef.h"
#   endif /* WIN32 */
#endif /* __cplusplus */


#ifndef _WX_DEFS_H_
#define _WX_DEFS_H_

/*  ---------------------------------------------------------------------------- */
/*  compiler and OS identification */
/*  ---------------------------------------------------------------------------- */

#include "wx/platform.h"

#ifdef __cplusplus
/*  Make sure the environment is set correctly */
#   if defined(__WXMSW__) && defined(__X__)
#       error "Target can't be both X and MSW"
#   elif !defined(__WXMOTIF__) && \
         !defined(__WXMSW__)   && \
         !defined(__WXGTK__)   && \
         !defined(__WXOSX_COCOA__)   && \
         !defined(__WXOSX_IPHONE__)   && \
         !defined(__X__)       && \
         !defined(__WXDFB__)   && \
         !defined(__WXX11__)   && \
         !defined(__WXQT__)    && \
          wxUSE_GUI
#       ifdef __UNIX__
#           error "No Target! You should use wx-config program for compilation flags!"
#       else /*  !Unix */
#           error "No Target! You should use supplied makefiles for compilation!"
#       endif /*  Unix/!Unix */
#   endif
#endif /*__cplusplus*/

#ifndef __WXWINDOWS__
    #define __WXWINDOWS__ 1
#endif

#ifndef wxUSE_BASE
    /*  by default consider that this is a monolithic build */
    #define wxUSE_BASE 1
#endif

#if !wxUSE_GUI && !defined(__WXBASE__)
    #define __WXBASE__
#endif

/*  suppress some Visual C++ warnings */
#ifdef __VISUALC__
    /*  the only "real" warning here is 4244 but there are just too many of them */
    /*  in our code... one day someone should go and fix them but until then... */
#   pragma warning(disable:4097)    /*  typedef used as class */
#   pragma warning(disable:4201)    /*  nonstandard extension used: nameless struct/union */
#   pragma warning(disable:4244)    /*  conversion from double to float */
#   pragma warning(disable:4355)    /* 'this' used in base member initializer list */
#   pragma warning(disable:4511)    /*  copy ctor couldn't be generated */
#   pragma warning(disable:4512)    /*  operator=() couldn't be generated */
#   pragma warning(disable:4514)   /*  unreferenced inline func has been removed */
#   pragma warning(disable:4710)    /*  function not inlined */

    /*
        TODO: this warning should really be enabled as it can be genuinely
              useful, check where does it occur in wxWidgets
     */
    #pragma warning(disable: 4127) /*  conditional expression is constant */

    /* There are too many false positives for this one, particularly when
       using templates like wxVector<T> */
    /* class 'foo' needs to have dll-interface to be used by clients of
       class 'bar'" */
#   pragma warning(disable:4251)

    /*
        This is a similar warning which occurs when deriving from standard
        containers. MSDN even mentions that it can be ignored in this case
        (albeit only in debug build while the warning is the same in release
        too and seems equally harmless).
     */
#if wxUSE_STD_CONTAINERS
#   pragma warning(disable:4275)
#endif /* wxUSE_STD_CONTAINERS */

#   ifdef __VISUALC5__
    /* For VC++ 5.0 for release mode, the warning 'C4702: unreachable code */
    /* is buggy, and occurs for code that does actually get executed */
#   ifndef __WXDEBUG__
#       pragma warning(disable:4702)    /* unreachable code */
#   endif

    /* The VC++ 5.0 warning 'C4003: not enough actual parameters for macro'
     * is incompatible with the wxWidgets headers since it is given when
     * parameters are empty but not missing. */
#       pragma warning(disable:4003)    /* not enough actual parameters for macro */
#   endif

    /*
       When compiling with VC++ 7 /Wp64 option we get thousands of warnings for
       conversion from size_t to int or long. Some precious few of them might
       be worth looking into but unfortunately it seems infeasible to fix all
       the other, harmless ones (e.g. inserting static_cast<int>(s.length())
       everywhere this method is used though we are quite sure that using >4GB
       strings is a bad idea anyhow) so just disable it globally for now.
     */
    #if wxCHECK_VISUALC_VERSION(7)
        /* conversion from 'size_t' to 'unsigned long', possible loss of data */
        #pragma warning(disable:4267)
    #endif /* VC++ 7 or later */

    /*
       VC++ 8 gives a warning when using standard functions such as sprintf,
       localtime, ... -- stop this madness, unless the user had already done it
     */
    #if wxCHECK_VISUALC_VERSION(8)
        #ifndef _CRT_SECURE_NO_DEPRECATE
            #define _CRT_SECURE_NO_DEPRECATE 1
        #endif
        #ifndef _CRT_NON_CONFORMING_SWPRINTFS
            #define _CRT_NON_CONFORMING_SWPRINTFS 1
        #endif
        #ifndef _SCL_SECURE_NO_WARNINGS
            #define _SCL_SECURE_NO_WARNINGS 1
        #endif
    #endif /* VC++ 8 */
#endif /*  __VISUALC__ */

/*
   g++ gives a warning when a class has private dtor if it has no friends but
   this is a perfectly valid situation for a ref-counted class which destroys
   itself when its ref count drops to 0, so provide a macro to suppress this
   warning
 */
#ifdef __GNUG__
#   define wxSUPPRESS_GCC_PRIVATE_DTOR_WARNING(name) \
        friend class wxDummyFriendFor ## name;
#else /* !g++ */
#   define wxSUPPRESS_GCC_PRIVATE_DTOR_WARNING(name)
#endif

/*
   Clang Support
 */

#ifndef WX_HAS_CLANG_FEATURE
#   ifndef __has_feature
#       define WX_HAS_CLANG_FEATURE(x) 0
#   else
#       define WX_HAS_CLANG_FEATURE(x) __has_feature(x)
#   endif
#endif

/*  ---------------------------------------------------------------------------- */
/*  wxWidgets version and compatibility defines */
/*  ---------------------------------------------------------------------------- */

#include "wx/version.h"

/*  ============================================================================ */
/*  non portable C++ features */
/*  ============================================================================ */

/*  ---------------------------------------------------------------------------- */
/*  compiler defects workarounds */
/*  ---------------------------------------------------------------------------- */

/*
   Digital Unix C++ compiler only defines this symbol for .cxx and .hxx files,
   so define it ourselves (newer versions do it for all files, though, and
   don't allow it to be redefined)
 */
#if defined(__DECCXX) && !defined(__VMS) && !defined(__cplusplus)
#define __cplusplus
#endif /* __DECCXX */

/*  Resolves linking problems under HP-UX when compiling with gcc/g++ */
#if defined(__HPUX__) && defined(__GNUG__)
#define va_list __gnuc_va_list
#endif /*  HP-UX */

/* Prevents conflicts between sys/types.h and winsock.h with Cygwin, */
/* when using Windows sockets. */
#if defined(__CYGWIN__) && defined(__WINDOWS__)
#define __USE_W32_SOCKETS
#endif

/*  ---------------------------------------------------------------------------- */
/*  check for native bool type and TRUE/FALSE constants */
/*  ---------------------------------------------------------------------------- */

/*  for backwards compatibility, also define TRUE and FALSE */
/*  */
/*  note that these definitions should work both in C++ and C code, so don't */
/*  use true/false below */
#ifndef TRUE
    #define TRUE 1
#endif

#ifndef FALSE
    #define FALSE 0
#endif

typedef short int WXTYPE;


/*  ---------------------------------------------------------------------------- */
/*  other feature tests */
/*  ---------------------------------------------------------------------------- */

#ifdef __cplusplus

/*  Every ride down a slippery slope begins with a single step.. */
/*  */
/*  Yes, using nested classes is indeed against our coding standards in */
/*  general, but there are places where you can use them to advantage */
/*  without totally breaking ports that cannot use them.  If you do, then */
/*  wrap it in this guard, but such cases should still be relatively rare. */
#define wxUSE_NESTED_CLASSES    1

/* This macro is obsolete, use the 'explicit' keyword in the new code. */
#define wxEXPLICIT explicit

/* check for override keyword support */
#ifndef HAVE_OVERRIDE
    #if __cplusplus >= 201103L
        /* All C++11 compilers should have it. */
        #define HAVE_OVERRIDE
    #elif wxCHECK_VISUALC_VERSION(11)
        /*
           VC++ supports override keyword since version 8 but doesn't define
           __cplusplus as indicating C++11 support (at least up to and
           including 12), so handle its case specially.

           Also note that while the keyword is supported, using it with
           versions 8, 9 and 10 results in C4481 compiler warning ("nonstandard
           extension used") and so we avoid using it there, you could disable
           this warning and predefine HAVE_OVERRIDE if you don't care about it.
         */
        #define HAVE_OVERRIDE
    #elif WX_HAS_CLANG_FEATURE(cxx_override_control)
        #define HAVE_OVERRIDE
    #endif
#endif /* !HAVE_OVERRIDE */

#ifdef HAVE_OVERRIDE
    #define wxOVERRIDE override
#else /*  !HAVE_OVERRIDE */
    #define wxOVERRIDE
#endif /*  HAVE_OVERRIDE */

/* same for more C++11 keywords which don't have such historic baggage as
   override and so can be detected by just testing for C++11 support (which
   still requires handling MSVS specially, unfortunately) */
#if __cplusplus >= 201103L || wxCHECK_VISUALC_VERSION(14)
    #define wxHAS_MEMBER_DEFAULT

    #define wxHAS_NOEXCEPT
    #define wxNOEXCEPT noexcept
#else
    #define wxNOEXCEPT
#endif

/*
    Support for nullptr is available since MSVS 2010, even though it doesn't
    define __cplusplus as a C++11 compiler.
 */
#if __cplusplus >= 201103 || wxCHECK_VISUALC_VERSION(10)
    #define wxHAS_NULLPTR_T
#endif

/* wxFALLTHROUGH is used to notate explicit fallthroughs in switch statements */

#if __cplusplus >= 201703L
    #define wxFALLTHROUGH [[fallthrough]]
#elif __cplusplus >= 201103L && defined(__has_warning) && WX_HAS_CLANG_FEATURE(cxx_attributes)
    #define wxFALLTHROUGH [[clang::fallthrough]]
#elif wxCHECK_GCC_VERSION(7, 0)
    #define wxFALLTHROUGH __attribute__ ((fallthrough))
#endif

#ifndef wxFALLTHROUGH
    #define wxFALLTHROUGH ((void)0)
#endif

/* these macros are obsolete, use the standard C++ casts directly now */
#define wx_static_cast(t, x) static_cast<t>(x)
#define wx_const_cast(t, x) const_cast<t>(x)
#define wx_reinterpret_cast(t, x) reinterpret_cast<t>(x)

/*
   This one is a wx invention: like static cast but used when we intentionally
   truncate from a larger to smaller type, static_cast<> can't be used for it
   as it results in warnings when using some compilers (SGI mipspro for example)
 */
#if defined(__INTELC__)
    template <typename T, typename X>
    inline T wx_truncate_cast_impl(X x)
    {
        #pragma warning(push)
        /* implicit conversion of a 64-bit integral type to a smaller integral type */
        #pragma warning(disable: 1682)
        /* conversion from "X" to "T" may lose significant bits */
        #pragma warning(disable: 810)
        /* non-pointer conversion from "foo" to "bar" may lose significant bits */
        #pragma warning(disable: 2259)

        return x;

        #pragma warning(pop)
    }

    #define wx_truncate_cast(t, x) wx_truncate_cast_impl<t>(x)

#elif defined(__clang__)
    #define wx_truncate_cast(t, x) static_cast<t>(x)

#elif defined(__VISUALC__) && __VISUALC__ >= 1310
    template <typename T, typename X>
    inline T wx_truncate_cast_impl(X x)
    {
        #pragma warning(push)
        /* conversion from 'size_t' to 'type', possible loss of data */
        #pragma warning(disable: 4267)
        /* conversion from 'type1' to 'type2', possible loss of data */
        #pragma warning(disable: 4242)

        return x;

        #pragma warning(pop)
    }

    #define wx_truncate_cast(t, x) wx_truncate_cast_impl<t>(x)
#else
    #define wx_truncate_cast(t, x) ((t)(x))
#endif

/* for consistency with wxStatic/DynamicCast defined in wx/object.h */
#define wxConstCast(obj, className) const_cast<className *>(obj)

#ifndef HAVE_STD_WSTRING
    #if __cplusplus >= 201103L
        #define HAVE_STD_WSTRING
    #elif defined(__VISUALC__)
        #define HAVE_STD_WSTRING
    #elif defined(__MINGW32__)
        #define HAVE_STD_WSTRING
    #endif
#endif

#ifndef HAVE_STD_STRING_COMPARE
    #if __cplusplus >= 201103L
        #define HAVE_STD_STRING_COMPARE
    #elif defined(__VISUALC__)
        #define HAVE_STD_STRING_COMPARE
    #elif defined(__MINGW32__) || defined(__CYGWIN32__)
        #define HAVE_STD_STRING_COMPARE
    #endif
#endif

#ifndef HAVE_TR1_TYPE_TRAITS
    #if defined(__VISUALC__) && (_MSC_FULL_VER >= 150030729)
        #define HAVE_TR1_TYPE_TRAITS
    #endif
#endif

/*
    If using configure, stick to the options detected by it even if different
    compiler options could result in detecting something different here, as it
    would cause ABI issues otherwise (see #18034).
*/
#ifndef __WX_SETUP_H__
    /*
        Check for C++11 compilers, it is important to do it before the
        __has_include() checks because at least g++ 4.9.2+ __has_include() returns
        true for C++11 headers which can't be compiled in non-C++11 mode.
     */
    #if __cplusplus >= 201103L || wxCHECK_VISUALC_VERSION(10)
        #ifndef HAVE_TYPE_TRAITS
            #define HAVE_TYPE_TRAITS
        #endif
        #ifndef HAVE_STD_UNORDERED_MAP
            #define HAVE_STD_UNORDERED_MAP
        #endif
        #ifndef HAVE_STD_UNORDERED_SET
            #define HAVE_STD_UNORDERED_SET
        #endif
    #elif defined(__has_include)
        /*
            We're in non-C++11 mode here, so only test for pre-C++11 headers. As
            mentioned above, using __has_include() to test for C++11 would wrongly
            detect them even though they can't be used in this case, don't do it.
         */
        #if !defined(HAVE_TR1_TYPE_TRAITS) && __has_include(<tr1/type_traits>)
            #define HAVE_TR1_TYPE_TRAITS
        #endif

        #if !defined(HAVE_TR1_UNORDERED_MAP) && __has_include(<tr1/unordered_map>)
            #define HAVE_TR1_UNORDERED_MAP
        #endif

        #if !defined(HAVE_TR1_UNORDERED_SET) && __has_include(<tr1/unordered_set>)
            #define HAVE_TR1_UNORDERED_SET
        #endif
    #endif /* defined(__has_include) */

    #endif /* __cplusplus */
#endif /* __WX_SETUP_H__ */

/* provide replacement for C99 va_copy() if the compiler doesn't have it */

/* could be already defined by configure or the user */
#ifndef wxVaCopy
    /* if va_copy is a macro or configure detected that we have it, use it */
    #if defined(va_copy) || defined(HAVE_VA_COPY)
        #define wxVaCopy va_copy
    #else /* no va_copy, try to provide a replacement */
        /*
           configure tries to determine whether va_list is an array or struct
           type, but it may not be used under Windows, so deal with a few
           special cases.
         */

        #if defined(__PPC__) && (defined(_CALL_SYSV) || defined (_WIN32))
            /*
                PPC using SysV ABI and NT/PPC are special in that they use an
                extra level of indirection.
             */
            #define VA_LIST_IS_POINTER
        #endif /* SysV or Win32 on __PPC__ */

        /*
            note that we use memmove(), not memcpy(), in case anybody tries
            to do wxVaCopy(ap, ap)
         */
        #if defined(VA_LIST_IS_POINTER)
            #define wxVaCopy(d, s)  memmove(*(d), *(s), sizeof(va_list))
        #elif defined(VA_LIST_IS_ARRAY)
            #define wxVaCopy(d, s) memmove((d), (s), sizeof(va_list))
        #else /* we can only hope that va_lists are simple lvalues */
            #define wxVaCopy(d, s) ((d) = (s))
        #endif
    #endif /* va_copy/!va_copy */
#endif /* wxVaCopy */

#ifndef HAVE_WOSTREAM
    /*
        Cygwin is the only platform which doesn't have std::wostream
     */
    #if !defined(__CYGWIN__)
        #define HAVE_WOSTREAM
    #endif
#endif /* HAVE_WOSTREAM */

/*  ---------------------------------------------------------------------------- */
/*  portable calling conventions macros */
/*  ---------------------------------------------------------------------------- */

/*  stdcall is used for all functions called by Windows under Windows */
#if defined(__WINDOWS__)
    #if defined(__GNUWIN32__)
        #define wxSTDCALL __attribute__((stdcall))
    #else
        #define wxSTDCALL _stdcall
    #endif

#else /*  Win */
    /*  no such stupidness under Unix */
    #define wxSTDCALL
#endif /*  platform */

/*  LINKAGEMODE mode is most likely empty everywhere */
#ifndef LINKAGEMODE
    #define LINKAGEMODE
#endif /*  LINKAGEMODE */

/*  wxCALLBACK should be used for the functions which are called back by */
/*  Windows (such as compare function for wxListCtrl) */
#if defined(__WIN32__)
    #define wxCALLBACK wxSTDCALL
#else
    /*  no stdcall under Unix nor Win16 */
    #define wxCALLBACK
#endif /*  platform */

/*  generic calling convention for the extern "C" functions */

#if defined(__VISUALC__)
  #define   wxC_CALLING_CONV    _cdecl
#else   /*  !Visual C++ */
  #define   wxC_CALLING_CONV
#endif  /*  compiler */

/*  callling convention for the qsort(3) callback */
#define wxCMPFUNC_CONV wxC_CALLING_CONV

/*  compatibility :-( */
#define CMPFUNC_CONV wxCMPFUNC_CONV

/*  DLL import/export declarations */
#include "wx/dlimpexp.h"

/*  ---------------------------------------------------------------------------- */
/*  Very common macros */
/*  ---------------------------------------------------------------------------- */

/*  Printf-like attribute definitions to obtain warnings with GNU C/C++ */
#if defined(__GNUC__) && !wxUSE_UNICODE
#    define WX_ATTRIBUTE_FORMAT(like, m, n) __attribute__ ((__format__ (like, m, n)))
#else
#    define WX_ATTRIBUTE_FORMAT(like, m, n)
#endif

#ifndef WX_ATTRIBUTE_PRINTF
#   define WX_ATTRIBUTE_PRINTF(m, n) WX_ATTRIBUTE_FORMAT(__printf__, m, n)

#   define WX_ATTRIBUTE_PRINTF_1 WX_ATTRIBUTE_PRINTF(1, 2)
#   define WX_ATTRIBUTE_PRINTF_2 WX_ATTRIBUTE_PRINTF(2, 3)
#   define WX_ATTRIBUTE_PRINTF_3 WX_ATTRIBUTE_PRINTF(3, 4)
#   define WX_ATTRIBUTE_PRINTF_4 WX_ATTRIBUTE_PRINTF(4, 5)
#   define WX_ATTRIBUTE_PRINTF_5 WX_ATTRIBUTE_PRINTF(5, 6)
#endif /* !defined(WX_ATTRIBUTE_PRINTF) */

#ifndef WX_ATTRIBUTE_NORETURN
#   if WX_HAS_CLANG_FEATURE(attribute_analyzer_noreturn)
#       define WX_ATTRIBUTE_NORETURN __attribute__((analyzer_noreturn))
#   elif defined( __GNUC__ )
#       define WX_ATTRIBUTE_NORETURN __attribute__ ((noreturn))
#   elif defined(__VISUALC__)
#       define WX_ATTRIBUTE_NORETURN __declspec(noreturn)
#   else
#       define WX_ATTRIBUTE_NORETURN
#   endif
#endif

#if defined(__GNUC__)
    #define WX_ATTRIBUTE_UNUSED __attribute__ ((unused))
#else
    #define WX_ATTRIBUTE_UNUSED
#endif

/*
    Macros for marking functions as being deprecated.

    The preferred macro in the new code is wxDEPRECATED_MSG() which allows to
    explain why is the function deprecated. Almost all the existing code uses
    the older wxDEPRECATED() or its variants currently, but this will hopefully
    change in the future.
 */

/* The basic compiler-specific construct to generate a deprecation warning. */
#ifdef __clang__
    #define wxDEPRECATED_DECL __attribute__((deprecated))
#elif defined(__GNUC__)
    #define wxDEPRECATED_DECL __attribute__((deprecated))
#elif defined(__VISUALC__)
    #define wxDEPRECATED_DECL __declspec(deprecated)
#else
    #define wxDEPRECATED_DECL
#endif

/*
    Macro taking the deprecation message. It applies to the next declaration.

    If the compiler doesn't support showing the message, this degrades to a
    simple wxDEPRECATED(), i.e. at least gives a warning, if possible.
 */
#if defined(__clang__) && defined(__has_extension)
    #if __has_extension(attribute_deprecated_with_message)
        #define wxDEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
    #else
        #define wxDEPRECATED_MSG(msg) __attribute__((deprecated))
    #endif
#elif wxCHECK_GCC_VERSION(4, 5)
    #define wxDEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#elif wxCHECK_VISUALC_VERSION(8)
    #define wxDEPRECATED_MSG(msg) __declspec(deprecated("deprecated: " msg))
#else
    #define wxDEPRECATED_MSG(msg) wxDEPRECATED_DECL
#endif

/*
    Macro taking the declaration that it deprecates. Prefer to use
    wxDEPRECATED_MSG() instead as it's simpler (wrapping the entire declaration
    makes the code unclear) and allows to specify the explanation.
 */
#define wxDEPRECATED(x) wxDEPRECATED_DECL x

/*
    This macro used to be defined differently for gcc < 3.4, but we don't
    support it any more, so it's just the same thing as wxDEPRECATED now.
 */
#define wxDEPRECATED_CONSTRUCTOR(x) wxDEPRECATED(x)

/*
   Macro which marks the function as being deprecated but also defines it
   inline.

   Currently it's defined in the same trivial way in all cases but it could
   need a special definition with some other compilers in the future which
   explains why do we have it.
 */
#define wxDEPRECATED_INLINE(func, body) wxDEPRECATED(func) { body }

/*
    A macro to define a simple deprecated accessor.
 */
#define wxDEPRECATED_ACCESSOR(func, what) wxDEPRECATED_INLINE(func, return what;)

/*
   Special variant of the macro above which should be used for the functions
   which are deprecated but called by wx itself: this often happens with
   deprecated virtual functions which are called by the library.
 */
#ifdef WXBUILDING
#   define wxDEPRECATED_BUT_USED_INTERNALLY(x) x
#else
#   define wxDEPRECATED_BUT_USED_INTERNALLY(x) wxDEPRECATED(x)
#endif

/*
   Macros to suppress and restore gcc warnings, requires g++ >= 4.6 and don't
   do anything otherwise.

   Example of use:

        wxGCC_WARNING_SUPPRESS(float-equal)
        inline bool wxIsSameDouble(double x, double y) { return x == y; }
        wxGCC_WARNING_RESTORE(float-equal)

   Note that these macros apply to both gcc and clang, even though they only
   have "GCC" in their names.
 */
#if defined(__clang__) || wxCHECK_GCC_VERSION(4, 6)
#   define wxGCC_WARNING_SUPPRESS(x) \
        _Pragma (wxSTRINGIZE(GCC diagnostic push)) \
        _Pragma (wxSTRINGIZE(GCC diagnostic ignored wxSTRINGIZE(wxCONCAT(-W,x))))
#   define wxGCC_WARNING_RESTORE(x) \
       _Pragma (wxSTRINGIZE(GCC diagnostic pop))
#else /* gcc < 4.6 or not gcc and not clang at all */
#   define wxGCC_WARNING_SUPPRESS(x)
#   define wxGCC_WARNING_RESTORE(x)
#endif

/*
    Similar macros but for gcc-specific warnings.
 */
#ifdef __GNUC__
#   define wxGCC_ONLY_WARNING_SUPPRESS(x) wxGCC_WARNING_SUPPRESS(x)
#   define wxGCC_ONLY_WARNING_RESTORE(x) wxGCC_WARNING_RESTORE(x)
#else
#   define wxGCC_ONLY_WARNING_SUPPRESS(x)
#   define wxGCC_ONLY_WARNING_RESTORE(x)
#endif

/* Specific macros for -Wcast-function-type warning new in gcc 8. */
#if wxCHECK_GCC_VERSION(8, 0)
    #define wxGCC_WARNING_SUPPRESS_CAST_FUNCTION_TYPE() \
        wxGCC_WARNING_SUPPRESS(cast-function-type)
    #define wxGCC_WARNING_RESTORE_CAST_FUNCTION_TYPE() \
        wxGCC_WARNING_RESTORE(cast-function-type)
#else
    #define wxGCC_WARNING_SUPPRESS_CAST_FUNCTION_TYPE()
    #define wxGCC_WARNING_RESTORE_CAST_FUNCTION_TYPE()
#endif

/*
   Macros to suppress and restore clang warning only when it is valid.

   Example:
        wxCLANG_WARNING_SUPPRESS(inconsistent-missing-override)
        virtual wxClassInfo *GetClassInfo() const
        wxCLANG_WARNING_RESTORE(inconsistent-missing-override)
*/
#if defined(__clang__) && defined(__has_warning)
#    define wxCLANG_HAS_WARNING(x) __has_warning(x) /* allow macro expansion for the warning name */
#    define wxCLANG_IF_VALID_WARNING(x,y) \
         wxCONCAT(wxCLANG_IF_VALID_WARNING_,wxCLANG_HAS_WARNING(wxSTRINGIZE(wxCONCAT(-W,x))))(y)
#    define wxCLANG_IF_VALID_WARNING_0(x)
#    define wxCLANG_IF_VALID_WARNING_1(x) x
#    define wxCLANG_WARNING_SUPPRESS(x) \
         wxCLANG_IF_VALID_WARNING(x,wxGCC_WARNING_SUPPRESS(x))
#    define wxCLANG_WARNING_RESTORE(x) \
         wxCLANG_IF_VALID_WARNING(x,wxGCC_WARNING_RESTORE(x))
#else
#    define wxCLANG_WARNING_SUPPRESS(x)
#    define wxCLANG_WARNING_RESTORE(x)
#endif

/*
    Specific macro for disabling warnings related to not using override: this
    has to be done differently for gcc and clang and is only supported since
    gcc 5.1.
 */
#if defined(__clang__)
#   define wxWARNING_SUPPRESS_MISSING_OVERRIDE() \
        wxCLANG_WARNING_SUPPRESS(suggest-override) \
        wxCLANG_WARNING_SUPPRESS(inconsistent-missing-override)
#   define wxWARNING_RESTORE_MISSING_OVERRIDE() \
        wxCLANG_WARNING_RESTORE(inconsistent-missing-override) \
        wxCLANG_WARNING_RESTORE(suggest-override)
#elif wxCHECK_GCC_VERSION(5, 1)
#   define wxWARNING_SUPPRESS_MISSING_OVERRIDE() \
        wxGCC_WARNING_SUPPRESS(suggest-override)
#   define wxWARNING_RESTORE_MISSING_OVERRIDE() \
        wxGCC_WARNING_RESTORE(suggest-override)
#else
#   define wxWARNING_SUPPRESS_MISSING_OVERRIDE()
#   define wxWARNING_RESTORE_MISSING_OVERRIDE()
#endif

/*
    Combination of the two variants above: should be used for deprecated
    functions which are defined inline and are used by wxWidgets itself.
 */
#ifdef WXBUILDING
#   define wxDEPRECATED_BUT_USED_INTERNALLY_INLINE(func, body) func { body }
#else
#   define wxDEPRECATED_BUT_USED_INTERNALLY_INLINE(func, body) \
        wxDEPRECATED(func) { body }
#endif

/*  NULL declaration: it must be defined as 0 for C++ programs (in particular, */
/*  it must not be defined as "(void *)0" which is standard for C but completely */
/*  breaks C++ code) */
#include <stddef.h>

/*  size of statically declared array */
#define WXSIZEOF(array)   (sizeof(array)/sizeof(array[0]))

/*  symbolic constant used by all Find()-like functions returning positive */
/*  integer on success as failure indicator */
#define wxNOT_FOUND       (-1)

/* the default value for some length parameters meaning that the string is */
/* NUL-terminated */
#define wxNO_LEN ((size_t)-1)

/*  ---------------------------------------------------------------------------- */
/*  macros dealing with comparison operators */
/*  ---------------------------------------------------------------------------- */

/*
    Expands into m(op, args...) for each op in the set { ==, !=, <, <=, >, >= }.
 */
#define wxFOR_ALL_COMPARISONS(m) \
    m(==) m(!=) m(>=) m(<=) m(>) m(<)

#define wxFOR_ALL_COMPARISONS_1(m, x) \
    m(==,x) m(!=,x) m(>=,x) m(<=,x) m(>,x) m(<,x)

#define wxFOR_ALL_COMPARISONS_2(m, x, y) \
    m(==,x,y) m(!=,x,y) m(>=,x,y) m(<=,x,y) m(>,x,y) m(<,x,y)

#define wxFOR_ALL_COMPARISONS_3(m, x, y, z) \
    m(==,x,y,z) m(!=,x,y,z) m(>=,x,y,z) m(<=,x,y,z) m(>,x,y,z) m(<,x,y,z)

/*
    These are only used with wxDEFINE_COMPARISON_[BY_]REV: they pass both the
    normal and the reversed comparison operators to the macro.
 */
#define wxFOR_ALL_COMPARISONS_2_REV(m, x, y) \
    m(==,x,y,==) m(!=,x,y,!=) m(>=,x,y,<=) \
    m(<=,x,y,>=) m(>,x,y,<) m(<,x,y,>)

#define wxFOR_ALL_COMPARISONS_3_REV(m, x, y, z) \
    m(==,x,y,z,==) m(!=,x,y,z,!=) m(>=,x,y,z,<=) \
    m(<=,x,y,z,>=) m(>,x,y,z,<) m(<,x,y,z,>)


#define wxDEFINE_COMPARISON(op, T1, T2, cmp) \
    inline bool operator op(T1 x, T2 y) { return cmp(x, y, op); }

#define wxDEFINE_COMPARISON_REV(op, T1, T2, cmp, oprev) \
    inline bool operator op(T2 y, T1 x) { return cmp(x, y, oprev); }

#define wxDEFINE_COMPARISON_BY_REV(op, T1, T2, oprev) \
    inline bool operator op(T1 x, T2 y) { return y oprev x; }

/*
    Define all 6 comparison operators (==, !=, <, <=, >, >=) for the given
    types in the specified order. The implementation is provided by the cmp
    macro. Normally wxDEFINE_ALL_COMPARISONS should be used as comparison
    operators are usually symmetric.
 */
#define wxDEFINE_COMPARISONS(T1, T2, cmp) \
    wxFOR_ALL_COMPARISONS_3(wxDEFINE_COMPARISON, T1, T2, cmp)

/*
    Define all 6 comparison operators (==, !=, <, <=, >, >=) for the given
    types in the specified order, implemented in terms of existing operators
    for the reverse order.
 */
#define wxDEFINE_COMPARISONS_BY_REV(T1, T2) \
    wxFOR_ALL_COMPARISONS_2_REV(wxDEFINE_COMPARISON_BY_REV, T1, T2)

/*
    This macro allows to define all 12 comparison operators (6 operators for
    both orders of arguments) for the given types using the provided "cmp"
    macro to implement the actual comparison: the macro is called with the 2
    arguments names, the first of type T1 and the second of type T2, and the
    comparison operator being implemented.
 */
#define wxDEFINE_ALL_COMPARISONS(T1, T2, cmp) \
    wxFOR_ALL_COMPARISONS_3(wxDEFINE_COMPARISON, T1, T2, cmp) \
    wxFOR_ALL_COMPARISONS_3_REV(wxDEFINE_COMPARISON_REV, T1, T2, cmp)

/*  ---------------------------------------------------------------------------- */
/*  macros to avoid compiler warnings */
/*  ---------------------------------------------------------------------------- */

/*  Macro to cut down on compiler warnings. */
#if 1 /*  there should be no more any compilers needing the "#else" version */
    #define WXUNUSED(identifier) /* identifier */
#else  /*  stupid, broken compiler */
    #define WXUNUSED(identifier) identifier
#endif

/*  some arguments are not used in unicode mode */
#if wxUSE_UNICODE
    #define WXUNUSED_IN_UNICODE(param)  WXUNUSED(param)
#else
    #define WXUNUSED_IN_UNICODE(param)  param
#endif

/*  unused parameters in non stream builds */
#if wxUSE_STREAMS
    #define WXUNUSED_UNLESS_STREAMS(param)  param
#else
    #define WXUNUSED_UNLESS_STREAMS(param)  WXUNUSED(param)
#endif

/*  some compilers give warning about a possibly unused variable if it is */
/*  initialized in both branches of if/else and shut up if it is initialized */
/*  when declared, but other compilers then give warnings about unused variable */
/*  value -- this should satisfy both of them */
#if defined(__VISUALC__)
    #define wxDUMMY_INITIALIZE(val) = val
#else
    #define wxDUMMY_INITIALIZE(val)
#endif

/*  sometimes the value of a variable is *really* not used, to suppress  the */
/*  resulting warning you may pass it to this function */
#ifdef __cplusplus
    template <class T>
        inline void wxUnusedVar(const T& WXUNUSED(t)) { }
#endif

/*  ---------------------------------------------------------------------------- */
/*  compiler specific settings */
/*  ---------------------------------------------------------------------------- */

/*  where should i put this? we need to make sure of this as it breaks */
/*  the <iostream> code. */
#if defined(__WXDEBUG__)
#    undef wxUSE_DEBUG_NEW_ALWAYS
#    define wxUSE_DEBUG_NEW_ALWAYS 0
#endif

#include "wx/types.h"

#ifdef __cplusplus

// everybody gets the assert and other debug macros
#include "wx/debug.h"

    // delete pointer if it is not NULL and NULL it afterwards
    template <typename T>
    inline void wxDELETE(T*& ptr)
    {
        typedef char TypeIsCompleteCheck[sizeof(T)] WX_ATTRIBUTE_UNUSED;

        if ( ptr != NULL )
        {
            delete ptr;
            ptr = NULL;
        }
    }

    // delete an array and NULL it (see comments above)
    template <typename T>
    inline void wxDELETEA(T*& ptr)
    {
        typedef char TypeIsCompleteCheck[sizeof(T)] WX_ATTRIBUTE_UNUSED;

        if ( ptr != NULL )
        {
            delete [] ptr;
            ptr = NULL;
        }
    }

    // trivial implementation of std::swap() for primitive types
    template <typename T>
    inline void wxSwap(T& first, T& second)
    {
        T tmp(first);
        first = second;
        second = tmp;
    }

/* And also define a couple of simple functions to cast pointer to/from it. */
inline wxUIntPtr wxPtrToUInt(const void *p)
{
    /*
       VC++ 7.1 gives warnings about casts such as below even when they're
       explicit with /Wp64 option, suppress them as we really know what we're
       doing here. Same thing with icc with -Wall.
     */
#ifdef __VISUALC__
    #pragma warning(push)
    /* pointer truncation from '' to '' */
    #pragma warning(disable: 4311)
#elif defined(__INTELC__)
    #pragma warning(push)
    /* conversion from pointer to same-sized integral type */
    #pragma warning(disable: 1684)
#endif

    return reinterpret_cast<wxUIntPtr>(p);

#if defined(__VISUALC__) || defined(__INTELC__)
    #pragma warning(pop)
#endif
}

inline void *wxUIntToPtr(wxUIntPtr p)
{
#ifdef __VISUALC__
    #pragma warning(push)
    /* conversion to type of greater size */
    #pragma warning(disable: 4312)
#elif defined(__INTELC__)
    #pragma warning(push)
    /* invalid type conversion: "wxUIntPtr={unsigned long}" to "void *" */
    #pragma warning(disable: 171)
#endif

    return reinterpret_cast<void *>(p);

#if defined(__VISUALC__) || defined(__INTELC__)
    #pragma warning(pop)
#endif
}
#endif /*__cplusplus*/



/*  base floating point types */
/*  wxFloat32: 32 bit IEEE float ( 1 sign, 8 exponent bits, 23 fraction bits ) */
/*  wxFloat64: 64 bit IEEE float ( 1 sign, 11 exponent bits, 52 fraction bits ) */
/*  wxDouble: native fastest representation that has at least wxFloat64 */
/*            precision, so use the IEEE types for storage, and this for */
/*            calculations */

typedef float wxFloat32;
typedef double wxFloat64;

typedef double wxDouble;

/*
    Some (non standard) compilers typedef wchar_t as an existing type instead
    of treating it as a real fundamental type, set wxWCHAR_T_IS_REAL_TYPE to 0
    for them and to 1 for all the others.
 */
#ifndef wxWCHAR_T_IS_REAL_TYPE
    /*
        VC++ typedefs wchar_t as unsigned short by default until VC8, that is
        unless /Za or /Zc:wchar_t option is used in which case _WCHAR_T_DEFINED
        is defined.
     */
#   if defined(__VISUALC__) && !defined(_NATIVE_WCHAR_T_DEFINED)
#       define wxWCHAR_T_IS_REAL_TYPE 0
#   else /* compiler having standard-conforming wchar_t */
#       define wxWCHAR_T_IS_REAL_TYPE 1
#   endif
#endif /* !defined(wxWCHAR_T_IS_REAL_TYPE) */

/* Helper macro for doing something dependent on whether wchar_t is or isn't a
   typedef inside another macro. */
#if wxWCHAR_T_IS_REAL_TYPE
    #define wxIF_WCHAR_T_TYPE(x) x
#else /* !wxWCHAR_T_IS_REAL_TYPE */
    #define wxIF_WCHAR_T_TYPE(x)
#endif /* wxWCHAR_T_IS_REAL_TYPE/!wxWCHAR_T_IS_REAL_TYPE */

/*
   This constant should be used instead of NULL in vararg functions taking
   wxChar* arguments: passing NULL (which is the same as 0, unless the compiler
   defines it specially, e.g. like gcc does with its __null built-in) doesn't
   work in this case as va_arg() wouldn't interpret the integer 0 correctly
   when trying to convert it to a pointer on architectures where sizeof(int) is
   strictly less than sizeof(void *).

   Examples of places where this must be used include wxFileTypeInfo ctor.
 */
#define wxNullPtr ((void *)NULL)


/* Define wxChar16 and wxChar32                                              */

#if SIZEOF_WCHAR_T == 2
    #define wxWCHAR_T_IS_WXCHAR16
    typedef wchar_t wxChar16;
#else
    typedef wxUint16 wxChar16;
#endif

#if SIZEOF_WCHAR_T == 4
    #define wxWCHAR_T_IS_WXCHAR32
    typedef wchar_t wxChar32;
#else
    typedef wxUint32 wxChar32;
#endif


/*
    Helper macro expanding into the given "m" macro invoked with each of the
    integer types as parameter (notice that this does not include char/unsigned
    char and bool but does include wchar_t).
 */
#define wxDO_FOR_INT_TYPES(m) \
    m(short) \
    m(unsigned short) \
    m(int) \
    m(unsigned int) \
    m(long) \
    m(unsigned long) \
    wxIF_LONG_LONG_TYPE( m(wxLongLong_t) ) \
    wxIF_LONG_LONG_TYPE( m(wxULongLong_t) ) \
    wxIF_WCHAR_T_TYPE( m(wchar_t) )

/*
    Same as wxDO_FOR_INT_TYPES() but does include char and unsigned char.

    Notice that we use "char" and "unsigned char" here but not "signed char"
    which would be more correct as "char" could be unsigned by default. But
    wxWidgets code currently supposes that char is signed and we'd need to
    clean up assumptions about it, notably in wx/unichar.h, to be able to use
    "signed char" here.
 */
#define wxDO_FOR_CHAR_INT_TYPES(m) \
    m(char) \
    m(unsigned char) \
    wxDO_FOR_INT_TYPES(m)

/*
    Same as wxDO_FOR_INT_TYPES() above except that m macro takes the
    type as the first argument and some extra argument, passed from this macro
    itself, as the second one.
 */
#define wxDO_FOR_INT_TYPES_1(m, arg) \
    m(short, arg) \
    m(unsigned short, arg) \
    m(int, arg) \
    m(unsigned int, arg) \
    m(long, arg) \
    m(unsigned long, arg) \
    wxIF_LONG_LONG_TYPE( m(wxLongLong_t, arg) ) \
    wxIF_LONG_LONG_TYPE( m(wxULongLong_t, arg) ) \
    wxIF_WCHAR_T_TYPE( m(wchar_t, arg) )

/*
    Combination of wxDO_FOR_CHAR_INT_TYPES() and wxDO_FOR_INT_TYPES_1():
    invokes the given macro with the specified argument as its second parameter
    for all char and int types.
 */
#define wxDO_FOR_CHAR_INT_TYPES_1(m, arg) \
    m(char, arg) \
    m(unsigned char, arg) \
    wxDO_FOR_INT_TYPES_1(m, arg)


/*  ---------------------------------------------------------------------------- */
/*  byte ordering related definition and macros */
/*  ---------------------------------------------------------------------------- */

/*  byte sex */

#define  wxBIG_ENDIAN     4321
#define  wxLITTLE_ENDIAN  1234
#define  wxPDP_ENDIAN     3412

#ifdef WORDS_BIGENDIAN
#define  wxBYTE_ORDER  wxBIG_ENDIAN
#else
#define  wxBYTE_ORDER  wxLITTLE_ENDIAN
#endif

/*  byte swapping */

#define wxUINT16_SWAP_ALWAYS(val) \
   ((wxUint16) ( \
    (((wxUint16) (val) & (wxUint16) 0x00ffU) << 8) | \
    (((wxUint16) (val) & (wxUint16) 0xff00U) >> 8)))

#define wxINT16_SWAP_ALWAYS(val) \
   ((wxInt16) ( \
    (((wxUint16) (val) & (wxUint16) 0x00ffU) << 8) | \
    (((wxUint16) (val) & (wxUint16) 0xff00U) >> 8)))

#define wxUINT32_SWAP_ALWAYS(val) \
   ((wxUint32) ( \
    (((wxUint32) (val) & (wxUint32) 0x000000ffU) << 24) | \
    (((wxUint32) (val) & (wxUint32) 0x0000ff00U) <<  8) | \
    (((wxUint32) (val) & (wxUint32) 0x00ff0000U) >>  8) | \
    (((wxUint32) (val) & (wxUint32) 0xff000000U) >> 24)))

#define wxINT32_SWAP_ALWAYS(val) \
   ((wxInt32) ( \
    (((wxUint32) (val) & (wxUint32) 0x000000ffU) << 24) | \
    (((wxUint32) (val) & (wxUint32) 0x0000ff00U) <<  8) | \
    (((wxUint32) (val) & (wxUint32) 0x00ff0000U) >>  8) | \
    (((wxUint32) (val) & (wxUint32) 0xff000000U) >> 24)))
/*  machine specific byte swapping */

#ifdef wxLongLong_t
    #define wxUINT64_SWAP_ALWAYS(val) \
       ((wxUint64) ( \
        (((wxUint64) (val) & (wxUint64) wxULL(0x00000000000000ff)) << 56) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x000000000000ff00)) << 40) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x0000000000ff0000)) << 24) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x00000000ff000000)) <<  8) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x000000ff00000000)) >>  8) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x0000ff0000000000)) >> 24) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x00ff000000000000)) >> 40) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0xff00000000000000)) >> 56)))

    #define wxINT64_SWAP_ALWAYS(val) \
       ((wxInt64) ( \
        (((wxUint64) (val) & (wxUint64) wxULL(0x00000000000000ff)) << 56) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x000000000000ff00)) << 40) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x0000000000ff0000)) << 24) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x00000000ff000000)) <<  8) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x000000ff00000000)) >>  8) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x0000ff0000000000)) >> 24) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0x00ff000000000000)) >> 40) | \
        (((wxUint64) (val) & (wxUint64) wxULL(0xff00000000000000)) >> 56)))
#elif wxUSE_LONGLONG /*  !wxLongLong_t */
    #define wxUINT64_SWAP_ALWAYS(val) \
       ((wxUint64) ( \
        ((wxULongLong(val) & wxULongLong(0L, 0x000000ffU)) << 56) | \
        ((wxULongLong(val) & wxULongLong(0L, 0x0000ff00U)) << 40) | \
        ((wxULongLong(val) & wxULongLong(0L, 0x00ff0000U)) << 24) | \
        ((wxULongLong(val) & wxULongLong(0L, 0xff000000U)) <<  8) | \
        ((wxULongLong(val) & wxULongLong(0x000000ffL, 0U)) >>  8) | \
        ((wxULongLong(val) & wxULongLong(0x0000ff00L, 0U)) >> 24) | \
        ((wxULongLong(val) & wxULongLong(0x00ff0000L, 0U)) >> 40) | \
        ((wxULongLong(val) & wxULongLong(0xff000000L, 0U)) >> 56)))

    #define wxINT64_SWAP_ALWAYS(val) \
       ((wxInt64) ( \
        ((wxLongLong(val) & wxLongLong(0L, 0x000000ffU)) << 56) | \
        ((wxLongLong(val) & wxLongLong(0L, 0x0000ff00U)) << 40) | \
        ((wxLongLong(val) & wxLongLong(0L, 0x00ff0000U)) << 24) | \
        ((wxLongLong(val) & wxLongLong(0L, 0xff000000U)) <<  8) | \
        ((wxLongLong(val) & wxLongLong(0x000000ffL, 0U)) >>  8) | \
        ((wxLongLong(val) & wxLongLong(0x0000ff00L, 0U)) >> 24) | \
        ((wxLongLong(val) & wxLongLong(0x00ff0000L, 0U)) >> 40) | \
        ((wxLongLong(val) & wxLongLong(0xff000000L, 0U)) >> 56)))
#endif /*  wxLongLong_t/!wxLongLong_t */

#ifdef WORDS_BIGENDIAN
    #define wxUINT16_SWAP_ON_BE(val)  wxUINT16_SWAP_ALWAYS(val)
    #define wxINT16_SWAP_ON_BE(val)   wxINT16_SWAP_ALWAYS(val)
    #define wxUINT16_SWAP_ON_LE(val)  (val)
    #define wxINT16_SWAP_ON_LE(val)   (val)
    #define wxUINT32_SWAP_ON_BE(val)  wxUINT32_SWAP_ALWAYS(val)
    #define wxINT32_SWAP_ON_BE(val)   wxINT32_SWAP_ALWAYS(val)
    #define wxUINT32_SWAP_ON_LE(val)  (val)
    #define wxINT32_SWAP_ON_LE(val)   (val)
    #if wxHAS_INT64
        #define wxUINT64_SWAP_ON_BE(val)  wxUINT64_SWAP_ALWAYS(val)
        #define wxUINT64_SWAP_ON_LE(val)  (val)
        #define wxINT64_SWAP_ON_BE(val)  wxINT64_SWAP_ALWAYS(val)
        #define wxINT64_SWAP_ON_LE(val)  (val)

        #define wxUINT64_SWAP_ON_BE_IN_PLACE(val)   val = wxUINT64_SWAP_ALWAYS(val)
        #define wxINT64_SWAP_ON_BE_IN_PLACE(val)   val = wxINT64_SWAP_ALWAYS(val)
        #define wxUINT64_SWAP_ON_LE_IN_PLACE(val)
        #define wxINT64_SWAP_ON_LE_IN_PLACE(val)
    #endif

    #define wxUINT16_SWAP_ON_BE_IN_PLACE(val)   val = wxUINT16_SWAP_ALWAYS(val)
    #define wxINT16_SWAP_ON_BE_IN_PLACE(val)   val = wxINT16_SWAP_ALWAYS(val)
    #define wxUINT16_SWAP_ON_LE_IN_PLACE(val)
    #define wxINT16_SWAP_ON_LE_IN_PLACE(val)
    #define wxUINT32_SWAP_ON_BE_IN_PLACE(val)   val = wxUINT32_SWAP_ALWAYS(val)
    #define wxINT32_SWAP_ON_BE_IN_PLACE(val)   val = wxINT32_SWAP_ALWAYS(val)
    #define wxUINT32_SWAP_ON_LE_IN_PLACE(val)
    #define wxINT32_SWAP_ON_LE_IN_PLACE(val)
#else
    #define wxUINT16_SWAP_ON_LE(val)  wxUINT16_SWAP_ALWAYS(val)
    #define wxINT16_SWAP_ON_LE(val)   wxINT16_SWAP_ALWAYS(val)
    #define wxUINT16_SWAP_ON_BE(val)  (val)
    #define wxINT16_SWAP_ON_BE(val)   (val)
    #define wxUINT32_SWAP_ON_LE(val)  wxUINT32_SWAP_ALWAYS(val)
    #define wxINT32_SWAP_ON_LE(val)   wxINT32_SWAP_ALWAYS(val)
    #define wxUINT32_SWAP_ON_BE(val)  (val)
    #define wxINT32_SWAP_ON_BE(val)   (val)
    #if wxHAS_INT64
        #define wxUINT64_SWAP_ON_LE(val)  wxUINT64_SWAP_ALWAYS(val)
        #define wxUINT64_SWAP_ON_BE(val)  (val)
        #define wxINT64_SWAP_ON_LE(val)  wxINT64_SWAP_ALWAYS(val)
        #define wxINT64_SWAP_ON_BE(val)  (val)
        #define wxUINT64_SWAP_ON_BE_IN_PLACE(val)
        #define wxINT64_SWAP_ON_BE_IN_PLACE(val)
        #define wxUINT64_SWAP_ON_LE_IN_PLACE(val)   val = wxUINT64_SWAP_ALWAYS(val)
        #define wxINT64_SWAP_ON_LE_IN_PLACE(val)   val = wxINT64_SWAP_ALWAYS(val)
    #endif

    #define wxUINT16_SWAP_ON_BE_IN_PLACE(val)
    #define wxINT16_SWAP_ON_BE_IN_PLACE(val)
    #define wxUINT16_SWAP_ON_LE_IN_PLACE(val)   val = wxUINT16_SWAP_ALWAYS(val)
    #define wxINT16_SWAP_ON_LE_IN_PLACE(val)   val = wxINT16_SWAP_ALWAYS(val)
    #define wxUINT32_SWAP_ON_BE_IN_PLACE(val)
    #define wxINT32_SWAP_ON_BE_IN_PLACE(val)
    #define wxUINT32_SWAP_ON_LE_IN_PLACE(val)   val = wxUINT32_SWAP_ALWAYS(val)
    #define wxINT32_SWAP_ON_LE_IN_PLACE(val)   val = wxINT32_SWAP_ALWAYS(val)
#endif

/*  ---------------------------------------------------------------------------- */
/*  Geometric flags */
/*  ---------------------------------------------------------------------------- */

enum wxGeometryCentre
{
    wxCENTRE                  = 0x0001,
    wxCENTER                  = wxCENTRE
};

/*  centering into frame rather than screen (obsolete) */
#define wxCENTER_FRAME          0x0000
/*  centre on screen rather than parent */
#define wxCENTRE_ON_SCREEN      0x0002
#define wxCENTER_ON_SCREEN      wxCENTRE_ON_SCREEN

enum wxOrientation
{
    /* don't change the values of these elements, they are used elsewhere */
    wxHORIZONTAL              = 0x0004,
    wxVERTICAL                = 0x0008,

    wxBOTH                    = wxVERTICAL | wxHORIZONTAL,

    /*  a mask to extract orientation from the combination of flags */
    wxORIENTATION_MASK        = wxBOTH
};

enum wxDirection
{
    wxLEFT                    = 0x0010,
    wxRIGHT                   = 0x0020,
    wxUP                      = 0x0040,
    wxDOWN                    = 0x0080,

    wxTOP                     = wxUP,
    wxBOTTOM                  = wxDOWN,

    wxNORTH                   = wxUP,
    wxSOUTH                   = wxDOWN,
    wxWEST                    = wxLEFT,
    wxEAST                    = wxRIGHT,

    wxALL                     = (wxUP | wxDOWN | wxRIGHT | wxLEFT),

    /*  a mask to extract direction from the combination of flags */
    wxDIRECTION_MASK           = wxALL
};

enum wxAlignment
{
    /*
        0 is a valid wxAlignment value (both wxALIGN_LEFT and wxALIGN_TOP
        use it) so define a symbolic name for an invalid alignment value
        which can be assumed to be different from anything else
     */
    wxALIGN_INVALID           = -1,

    wxALIGN_NOT               = 0x0000,
    wxALIGN_CENTER_HORIZONTAL = 0x0100,
    wxALIGN_CENTRE_HORIZONTAL = wxALIGN_CENTER_HORIZONTAL,
    wxALIGN_LEFT              = wxALIGN_NOT,
    wxALIGN_TOP               = wxALIGN_NOT,
    wxALIGN_RIGHT             = 0x0200,
    wxALIGN_BOTTOM            = 0x0400,
    wxALIGN_CENTER_VERTICAL   = 0x0800,
    wxALIGN_CENTRE_VERTICAL   = wxALIGN_CENTER_VERTICAL,

    wxALIGN_CENTER            = (wxALIGN_CENTER_HORIZONTAL | wxALIGN_CENTER_VERTICAL),
    wxALIGN_CENTRE            = wxALIGN_CENTER,

    /*  a mask to extract alignment from the combination of flags */
    wxALIGN_MASK              = 0x0f00
};

/* misc. flags for wxSizer items */
enum wxSizerFlagBits
{
    /*
        wxADJUST_MINSIZE doesn't do anything any more but we still define
        it for compatibility. Notice that it may be also predefined (as 0,
        hopefully) in the user code in order to use it even in
        !WXWIN_COMPATIBILITY_2_8 builds so don't redefine it in such case.
     */
#if WXWIN_COMPATIBILITY_2_8 && !defined(wxADJUST_MINSIZE)
    wxADJUST_MINSIZE               = 0,
#endif
    wxFIXED_MINSIZE                = 0x8000,
    wxRESERVE_SPACE_EVEN_IF_HIDDEN = 0x0002,

    /*  a mask to extract wxSizerFlagBits from combination of flags */
    wxSIZER_FLAG_BITS_MASK         = 0x8002
};

enum wxStretch
{
    wxSTRETCH_NOT             = 0x0000,
    wxSHRINK                  = 0x1000,
    wxGROW                    = 0x2000,
    wxEXPAND                  = wxGROW,
    wxSHAPED                  = 0x4000,
    wxTILE                    = wxSHAPED | wxFIXED_MINSIZE,

    /*  a mask to extract stretch from the combination of flags */
    wxSTRETCH_MASK            = 0x7000 /* sans wxTILE */
};

/*  border flags: the values are chosen for backwards compatibility */
enum wxBorder
{
    /*  this is different from wxBORDER_NONE as by default the controls do have */
    /*  border */
    wxBORDER_DEFAULT = 0,

    wxBORDER_NONE   = 0x00200000,
    wxBORDER_STATIC = 0x01000000,
    wxBORDER_SIMPLE = 0x02000000,
    wxBORDER_RAISED = 0x04000000,
    wxBORDER_SUNKEN = 0x08000000,
    wxBORDER_DOUBLE = 0x10000000, /* deprecated */
    wxBORDER_THEME  = wxBORDER_DOUBLE,

    /*  a mask to extract border style from the combination of flags */
    wxBORDER_MASK   = 0x1f200000
};

/* This makes it easier to specify a 'normal' border for a control */
#define wxDEFAULT_CONTROL_BORDER    wxBORDER_SUNKEN

/*  ---------------------------------------------------------------------------- */
/*  Window style flags */
/*  ---------------------------------------------------------------------------- */

/*
 * Values are chosen so they can be |'ed in a bit list.
 * Some styles are used across more than one group,
 * so the values mustn't clash with others in the group.
 * Otherwise, numbers can be reused across groups.
 */

/*
    Summary of the bits used by various styles.

    High word, containing styles which can be used with many windows:

    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
    |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
      |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
      |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  \_ wxFULL_REPAINT_ON_RESIZE
      |  |  |  |  |  |  |  |  |  |  |  |  |  |  \____ wxPOPUP_WINDOW
      |  |  |  |  |  |  |  |  |  |  |  |  |  \_______ wxWANTS_CHARS
      |  |  |  |  |  |  |  |  |  |  |  |  \__________ wxTAB_TRAVERSAL
      |  |  |  |  |  |  |  |  |  |  |  \_____________ wxTRANSPARENT_WINDOW
      |  |  |  |  |  |  |  |  |  |  \________________ wxBORDER_NONE
      |  |  |  |  |  |  |  |  |  \___________________ wxCLIP_CHILDREN
      |  |  |  |  |  |  |  |  \______________________ wxALWAYS_SHOW_SB
      |  |  |  |  |  |  |  \_________________________ wxBORDER_STATIC
      |  |  |  |  |  |  \____________________________ wxBORDER_SIMPLE
      |  |  |  |  |  \_______________________________ wxBORDER_RAISED
      |  |  |  |  \__________________________________ wxBORDER_SUNKEN
      |  |  |  \_____________________________________ wxBORDER_{DOUBLE,THEME}
      |  |  \________________________________________ wxCAPTION/wxCLIP_SIBLINGS
      |  \___________________________________________ wxHSCROLL
      \______________________________________________ wxVSCROLL


    Low word style bits is class-specific meaning that the same bit can have
    different meanings for different controls (e.g. 0x10 is wxCB_READONLY
    meaning that the control can't be modified for wxComboBox but wxLB_SORT
    meaning that the control should be kept sorted for wxListBox, while
    wxLB_SORT has a different value -- and this is just fine).
 */

/*
 * Window (Frame/dialog/subwindow/panel item) style flags
 */

/* The cast is needed to avoid g++ -Wnarrowing warnings when initializing
 * values of int type with wxVSCROLL on 32 bit platforms, where its value is
 * greater than INT_MAX.
 */
#define wxVSCROLL               ((int)0x80000000)
#define wxHSCROLL               0x40000000
#define wxCAPTION               0x20000000

/*  New styles (border styles are now in their own enum) */
#define wxDOUBLE_BORDER         wxBORDER_DOUBLE
#define wxSUNKEN_BORDER         wxBORDER_SUNKEN
#define wxRAISED_BORDER         wxBORDER_RAISED
#define wxBORDER                wxBORDER_SIMPLE
#define wxSIMPLE_BORDER         wxBORDER_SIMPLE
#define wxSTATIC_BORDER         wxBORDER_STATIC
#define wxNO_BORDER             wxBORDER_NONE

/*  wxALWAYS_SHOW_SB: instead of hiding the scrollbar when it is not needed, */
/*  disable it - but still show (see also wxLB_ALWAYS_SB style) */
/*  */
/*  NB: as this style is only supported by wxUniversal and wxMSW so far */
#define wxALWAYS_SHOW_SB        0x00800000

/*  Clip children when painting, which reduces flicker in e.g. frames and */
/*  splitter windows, but can't be used in a panel where a static box must be */
/*  'transparent' (panel paints the background for it) */
#define wxCLIP_CHILDREN         0x00400000

/*  Note we're reusing the wxCAPTION style because we won't need captions */
/*  for subwindows/controls */
#define wxCLIP_SIBLINGS         0x20000000

#define wxTRANSPARENT_WINDOW    0x00100000

/*  Add this style to a panel to get tab traversal working outside of dialogs */
/*  (on by default for wxPanel, wxDialog, wxScrolledWindow) */
#define wxTAB_TRAVERSAL         0x00080000

/*  Add this style if the control wants to get all keyboard messages (under */
/*  Windows, it won't normally get the dialog navigation key events) */
#define wxWANTS_CHARS           0x00040000

/*  Make window retained (Motif only, see src/generic/scrolwing.cpp)
 *  This is non-zero only under wxMotif, to avoid a clash with wxPOPUP_WINDOW
 *  on other platforms
 */

#ifdef __WXMOTIF__
#define wxRETAINED              0x00020000
#else
#define wxRETAINED              0x00000000
#endif
#define wxBACKINGSTORE          wxRETAINED

/*  set this flag to create a special popup window: it will be always shown on */
/*  top of other windows, will capture the mouse and will be dismissed when the */
/*  mouse is clicked outside of it or if it loses focus in any other way */
#define wxPOPUP_WINDOW          0x00020000

/*  force a full repaint when the window is resized (instead of repainting just */
/*  the invalidated area) */
#define wxFULL_REPAINT_ON_RESIZE 0x00010000

/*  obsolete: now this is the default behaviour */
/*  */
/*  don't invalidate the whole window (resulting in a PAINT event) when the */
/*  window is resized (currently, makes sense for wxMSW only) */
#define wxNO_FULL_REPAINT_ON_RESIZE 0

/* A mask which can be used to filter (out) all wxWindow-specific styles.
 */
#define wxWINDOW_STYLE_MASK     \
    (wxVSCROLL|wxHSCROLL|wxBORDER_MASK|wxALWAYS_SHOW_SB|wxCLIP_CHILDREN| \
     wxCLIP_SIBLINGS|wxTRANSPARENT_WINDOW|wxTAB_TRAVERSAL|wxWANTS_CHARS| \
     wxRETAINED|wxPOPUP_WINDOW|wxFULL_REPAINT_ON_RESIZE)

/*
 * Extra window style flags (use wxWS_EX prefix to make it clear that they
 * should be passed to wxWindow::SetExtraStyle(), not SetWindowStyle())
 */

/* This flag is obsolete as recursive validation is now the default (and only
 * possible) behaviour. Simply don't use it any more in the new code. */
#define wxWS_EX_VALIDATE_RECURSIVELY    0x00000000 /* used to be 1 */

/*  wxCommandEvents and the objects of the derived classes are forwarded to the */
/*  parent window and so on recursively by default. Using this flag for the */
/*  given window allows to block this propagation at this window, i.e. prevent */
/*  the events from being propagated further upwards. The dialogs have this */
/*  flag on by default. */
#define wxWS_EX_BLOCK_EVENTS            0x00000002

/*  don't use this window as an implicit parent for the other windows: this must */
/*  be used with transient windows as otherwise there is the risk of creating a */
/*  dialog/frame with this window as a parent which would lead to a crash if the */
/*  parent is destroyed before the child */
#define wxWS_EX_TRANSIENT               0x00000004

/*  don't paint the window background, we'll assume it will */
/*  be done by a theming engine. This is not yet used but could */
/*  possibly be made to work in the future, at least on Windows */
#define wxWS_EX_THEMED_BACKGROUND       0x00000008

/*  this window should always process idle events */
#define wxWS_EX_PROCESS_IDLE            0x00000010

/*  this window should always process UI update events */
#define wxWS_EX_PROCESS_UI_UPDATES      0x00000020

/*  Draw the window in a metal theme on Mac */
#define wxFRAME_EX_METAL                0x00000040
#define wxDIALOG_EX_METAL               0x00000040

/*  Use this style to add a context-sensitive help to the window (currently for */
/*  Win32 only and it doesn't work if wxMINIMIZE_BOX or wxMAXIMIZE_BOX are used) */
#define wxWS_EX_CONTEXTHELP             0x00000080

/* synonyms for wxWS_EX_CONTEXTHELP for compatibility */
#define wxFRAME_EX_CONTEXTHELP          wxWS_EX_CONTEXTHELP
#define wxDIALOG_EX_CONTEXTHELP         wxWS_EX_CONTEXTHELP

/*  Create a window which is attachable to another top level window */
#define wxFRAME_DRAWER          0x0020

/*
 * MDI parent frame style flags
 * Can overlap with some of the above.
 */

#define wxFRAME_NO_WINDOW_MENU  0x0100

/*
 * wxMenuBar style flags
 */
/*  use native docking */
#define wxMB_DOCKABLE       0x0001

/*
 * wxMenu style flags
 */
#define wxMENU_TEAROFF      0x0001

/*
 * Apply to all panel items
 */
#define wxCOLOURED          0x0800
#define wxFIXED_LENGTH      0x0400

/*
 * Styles for wxListBox
 */
#define wxLB_SORT           0x0010
#define wxLB_SINGLE         0x0020
#define wxLB_MULTIPLE       0x0040
#define wxLB_EXTENDED       0x0080
/*  wxLB_OWNERDRAW is Windows-only */
#define wxLB_NEEDED_SB      0x0000
#define wxLB_OWNERDRAW      0x0100
#define wxLB_ALWAYS_SB      0x0200
#define wxLB_NO_SB          0x0400
#define wxLB_HSCROLL        wxHSCROLL
/*  always show an entire number of rows */
#define wxLB_INT_HEIGHT     0x0800

/*
 * wxComboBox style flags
 */
#define wxCB_SIMPLE         0x0004
#define wxCB_SORT           0x0008
#define wxCB_READONLY       0x0010
#define wxCB_DROPDOWN       0x0020

/*
 * wxRadioBox style flags
 * These styles are not used in any port.
 */
#define wxRA_LEFTTORIGHT    0x0001
#define wxRA_TOPTOBOTTOM    0x0002

/*  New, more intuitive names to specify majorDim argument */
#define wxRA_SPECIFY_COLS   wxHORIZONTAL
#define wxRA_SPECIFY_ROWS   wxVERTICAL

/*  Old names for compatibility */
#define wxRA_HORIZONTAL     wxHORIZONTAL
#define wxRA_VERTICAL       wxVERTICAL

/*
 * wxRadioButton style flag
 */
#define wxRB_GROUP          0x0004
#define wxRB_SINGLE         0x0008

/*
 * wxScrollBar flags
 */
#define wxSB_HORIZONTAL      wxHORIZONTAL
#define wxSB_VERTICAL        wxVERTICAL

/*
 * wxSpinButton flags.
 * Note that a wxSpinCtrl is sometimes defined as a wxTextCtrl, and so the
 * flags shouldn't overlap with wxTextCtrl flags that can be used for a single
 * line controls (currently we reuse wxTE_CHARWRAP and wxTE_RICH2 neither of
 * which makes sense for them).
 */
#define wxSP_HORIZONTAL       wxHORIZONTAL /*  4 */
#define wxSP_VERTICAL         wxVERTICAL   /*  8 */
#define wxSP_ARROW_KEYS       0x4000
#define wxSP_WRAP             0x8000

/*
 * wxTabCtrl flags
 */
#define wxTC_RIGHTJUSTIFY     0x0010
#define wxTC_FIXEDWIDTH       0x0020
#define wxTC_TOP              0x0000    /*  default */
#define wxTC_LEFT             0x0020
#define wxTC_RIGHT            0x0040
#define wxTC_BOTTOM           0x0080
#define wxTC_MULTILINE        0x0200    /* == wxNB_MULTILINE */
#define wxTC_OWNERDRAW        0x0400

/*
 * wxStaticBitmap flags
 */
#define wxBI_EXPAND           wxEXPAND

/*
 * wxStaticLine flags
 */
#define wxLI_HORIZONTAL         wxHORIZONTAL
#define wxLI_VERTICAL           wxVERTICAL


/*
 * extended dialog specifiers. these values are stored in a different
 * flag and thus do not overlap with other style flags. note that these
 * values do not correspond to the return values of the dialogs (for
 * those values, look at the wxID_XXX defines).
 */

/*  wxCENTRE already defined as  0x00000001 */
#define wxYES                   0x00000002
#define wxOK                    0x00000004
#define wxNO                    0x00000008
#define wxYES_NO                (wxYES | wxNO)
#define wxCANCEL                0x00000010
#define wxAPPLY                 0x00000020
#define wxCLOSE                 0x00000040

#define wxOK_DEFAULT            0x00000000  /* has no effect (default) */
#define wxYES_DEFAULT           0x00000000  /* has no effect (default) */
#define wxNO_DEFAULT            0x00000080  /* only valid with wxYES_NO */
#define wxCANCEL_DEFAULT        0x80000000  /* only valid with wxCANCEL */

#define wxICON_WARNING          0x00000100
#define wxICON_ERROR            0x00000200
#define wxICON_QUESTION         0x00000400
#define wxICON_INFORMATION      0x00000800
#define wxICON_EXCLAMATION      wxICON_WARNING
#define wxICON_HAND             wxICON_ERROR
#define wxICON_STOP             wxICON_ERROR
#define wxICON_ASTERISK         wxICON_INFORMATION

#define wxHELP                  0x00001000
#define wxFORWARD               0x00002000
#define wxBACKWARD              0x00004000
#define wxRESET                 0x00008000
#define wxMORE                  0x00010000
#define wxSETUP                 0x00020000
#define wxICON_NONE             0x00040000
#define wxICON_AUTH_NEEDED      0x00080000

#define wxICON_MASK \
    (wxICON_EXCLAMATION|wxICON_HAND|wxICON_QUESTION|wxICON_INFORMATION|wxICON_NONE|wxICON_AUTH_NEEDED)

/*
 * Background styles. See wxWindow::SetBackgroundStyle
 */
enum wxBackgroundStyle
{
    /*
        background is erased in the EVT_ERASE_BACKGROUND handler or using
        the system default background if no such handler is defined (this
        is the default style)
     */
    wxBG_STYLE_ERASE,

    /*
        background is erased by the system, no EVT_ERASE_BACKGROUND event
        is generated at all
     */
    wxBG_STYLE_SYSTEM,

    /*
        background is erased in EVT_PAINT handler and not erased at all
        before it, this should be used if the paint handler paints over
        the entire window to avoid flicker
     */
    wxBG_STYLE_PAINT,

    /*
        Indicates that the window background is not erased, letting the parent
        window show through.
     */
    wxBG_STYLE_TRANSPARENT,

    /* this style is deprecated and doesn't do anything, don't use */
    wxBG_STYLE_COLOUR,

    /*
        this style is deprecated and is synonymous with
        wxBG_STYLE_PAINT, use the new name
     */
    wxBG_STYLE_CUSTOM = wxBG_STYLE_PAINT
};

/*
 * Key types used by (old style) lists and hashes.
 */
enum wxKeyType
{
    wxKEY_NONE,
    wxKEY_INTEGER,
    wxKEY_STRING
};

/*  ---------------------------------------------------------------------------- */
/*  standard IDs */
/*  ---------------------------------------------------------------------------- */

/*  Standard menu IDs */
enum wxStandardID
{
    /*
       These ids delimit the range used by automatically-generated ids
       (i.e. those used when wxID_ANY is specified during construction).
     */
#if defined(__WXMSW__) || wxUSE_AUTOID_MANAGEMENT
    /*
       On MSW the range is always restricted no matter if id management
       is used or not because the native window ids are limited to short
       range.  On other platforms the range is only restricted if id
       management is used so the reference count buffer won't be so big.
     */
    wxID_AUTO_LOWEST = -32000,
    wxID_AUTO_HIGHEST = -2000,
#else
    wxID_AUTO_LOWEST = -1000000,
    wxID_AUTO_HIGHEST = -2000,
#endif

    /* no id matches this one when compared to it */
    wxID_NONE = -3,

    /*  id for a separator line in the menu (invalid for normal item) */
    wxID_SEPARATOR = -2,

    /* any id: means that we don't care about the id, whether when installing
     * an event handler or when creating a new window */
    wxID_ANY = -1,


    /* all predefined ids are between wxID_LOWEST and wxID_HIGHEST */
    wxID_LOWEST = 4999,

    wxID_OPEN,
    wxID_CLOSE,
    wxID_NEW,
    wxID_SAVE,
    wxID_SAVEAS,
    wxID_REVERT,
    wxID_EXIT,
    wxID_UNDO,
    wxID_REDO,
    wxID_HELP,
    wxID_PRINT,
    wxID_PRINT_SETUP,
    wxID_PAGE_SETUP,
    wxID_PREVIEW,
    wxID_ABOUT,
    wxID_HELP_CONTENTS,
    wxID_HELP_INDEX,
    wxID_HELP_SEARCH,
    wxID_HELP_COMMANDS,
    wxID_HELP_PROCEDURES,
    wxID_HELP_CONTEXT,
    wxID_CLOSE_ALL,
    wxID_PREFERENCES,

    wxID_EDIT = 5030,
    wxID_CUT,
    wxID_COPY,
    wxID_PASTE,
    wxID_CLEAR,
    wxID_FIND,
    wxID_DUPLICATE,
    wxID_SELECTALL,
    wxID_DELETE,
    wxID_REPLACE,
    wxID_REPLACE_ALL,
    wxID_PROPERTIES,

    wxID_VIEW_DETAILS,
    wxID_VIEW_LARGEICONS,
    wxID_VIEW_SMALLICONS,
    wxID_VIEW_LIST,
    wxID_VIEW_SORTDATE,
    wxID_VIEW_SORTNAME,
    wxID_VIEW_SORTSIZE,
    wxID_VIEW_SORTTYPE,

    wxID_FILE = 5050,
    wxID_FILE1,
    wxID_FILE2,
    wxID_FILE3,
    wxID_FILE4,
    wxID_FILE5,
    wxID_FILE6,
    wxID_FILE7,
    wxID_FILE8,
    wxID_FILE9,

    /*  Standard button and menu IDs */
    wxID_OK = 5100,
    wxID_CANCEL,
    wxID_APPLY,
    wxID_YES,
    wxID_NO,
    wxID_STATIC,
    wxID_FORWARD,
    wxID_BACKWARD,
    wxID_DEFAULT,
    wxID_MORE,
    wxID_SETUP,
    wxID_RESET,
    wxID_CONTEXT_HELP,
    wxID_YESTOALL,
    wxID_NOTOALL,
    wxID_ABORT,
    wxID_RETRY,
    wxID_IGNORE,
    wxID_ADD,
    wxID_REMOVE,

    wxID_UP,
    wxID_DOWN,
    wxID_HOME,
    wxID_REFRESH,
    wxID_STOP,
    wxID_INDEX,

    wxID_BOLD,
    wxID_ITALIC,
    wxID_JUSTIFY_CENTER,
    wxID_JUSTIFY_FILL,
    wxID_JUSTIFY_RIGHT,
    wxID_JUSTIFY_LEFT,
    wxID_UNDERLINE,
    wxID_INDENT,
    wxID_UNINDENT,
    wxID_ZOOM_100,
    wxID_ZOOM_FIT,
    wxID_ZOOM_IN,
    wxID_ZOOM_OUT,
    wxID_UNDELETE,
    wxID_REVERT_TO_SAVED,
    wxID_CDROM,
    wxID_CONVERT,
    wxID_EXECUTE,
    wxID_FLOPPY,
    wxID_HARDDISK,
    wxID_BOTTOM,
    wxID_FIRST,
    wxID_LAST,
    wxID_TOP,
    wxID_INFO,
    wxID_JUMP_TO,
    wxID_NETWORK,
    wxID_SELECT_COLOR,
    wxID_SELECT_FONT,
    wxID_SORT_ASCENDING,
    wxID_SORT_DESCENDING,
    wxID_SPELL_CHECK,
    wxID_STRIKETHROUGH,

    /*  System menu IDs (used by wxUniv): */
    wxID_SYSTEM_MENU = 5200,
    wxID_CLOSE_FRAME,
    wxID_MOVE_FRAME,
    wxID_RESIZE_FRAME,
    wxID_MAXIMIZE_FRAME,
    wxID_ICONIZE_FRAME,
    wxID_RESTORE_FRAME,

    /* MDI window menu ids */
    wxID_MDI_WINDOW_FIRST = 5230,
    wxID_MDI_WINDOW_CASCADE = wxID_MDI_WINDOW_FIRST,
    wxID_MDI_WINDOW_TILE_HORZ,
    wxID_MDI_WINDOW_TILE_VERT,
    wxID_MDI_WINDOW_ARRANGE_ICONS,
    wxID_MDI_WINDOW_PREV,
    wxID_MDI_WINDOW_NEXT,
    wxID_MDI_WINDOW_LAST = wxID_MDI_WINDOW_NEXT,

    /* OS X system menu ids */
    wxID_OSX_MENU_FIRST = 5250,
    wxID_OSX_HIDE = wxID_OSX_MENU_FIRST,
    wxID_OSX_HIDEOTHERS,
    wxID_OSX_SHOWALL,
#if wxABI_VERSION >= 30001
    wxID_OSX_SERVICES,
    wxID_OSX_MENU_LAST = wxID_OSX_SERVICES,
#else
    wxID_OSX_MENU_LAST = wxID_OSX_SHOWALL,
#endif

    /*  IDs used by generic file dialog (13 consecutive starting from this value) */
    wxID_FILEDLGG = 5900,

    /*  IDs used by generic file ctrl (4 consecutive starting from this value) */
    wxID_FILECTRL = 5950,

    wxID_HIGHEST = 5999
};

/*  ---------------------------------------------------------------------------- */
/*  wxWindowID type                                                              */
/*  ---------------------------------------------------------------------------- */

/* Note that this is defined even in non-GUI code as the same type is also used
   for e.g. timer IDs. */
typedef int wxWindowID;

/*  ---------------------------------------------------------------------------- */
/*  other constants */
/*  ---------------------------------------------------------------------------- */

/*  menu and toolbar item kinds */
enum wxItemKind
{
    wxITEM_SEPARATOR = -1,
    wxITEM_NORMAL,
    wxITEM_CHECK,
    wxITEM_RADIO,
    wxITEM_DROPDOWN,
    wxITEM_MAX
};

/*
 * The possible states of a 3-state checkbox (Compatible
 * with the 2-state checkbox).
 */
enum wxCheckBoxState
{
    wxCHK_UNCHECKED,
    wxCHK_CHECKED,
    wxCHK_UNDETERMINED /* 3-state checkbox only */
};


/*  hit test results */
enum wxHitTest
{
    wxHT_NOWHERE,

    /*  scrollbar */
    wxHT_SCROLLBAR_FIRST = wxHT_NOWHERE,
    wxHT_SCROLLBAR_ARROW_LINE_1,    /*  left or upper arrow to scroll by line */
    wxHT_SCROLLBAR_ARROW_LINE_2,    /*  right or down */
    wxHT_SCROLLBAR_ARROW_PAGE_1,    /*  left or upper arrow to scroll by page */
    wxHT_SCROLLBAR_ARROW_PAGE_2,    /*  right or down */
    wxHT_SCROLLBAR_THUMB,           /*  on the thumb */
    wxHT_SCROLLBAR_BAR_1,           /*  bar to the left/above the thumb */
    wxHT_SCROLLBAR_BAR_2,           /*  bar to the right/below the thumb */
    wxHT_SCROLLBAR_LAST,

    /*  window */
    wxHT_WINDOW_OUTSIDE,            /*  not in this window at all */
    wxHT_WINDOW_INSIDE,             /*  in the client area */
    wxHT_WINDOW_VERT_SCROLLBAR,     /*  on the vertical scrollbar */
    wxHT_WINDOW_HORZ_SCROLLBAR,     /*  on the horizontal scrollbar */
    wxHT_WINDOW_CORNER,             /*  on the corner between 2 scrollbars */

    wxHT_MAX
};

/*  ---------------------------------------------------------------------------- */
/*  Possible SetSize flags */
/*  ---------------------------------------------------------------------------- */

/*  Use internally-calculated width if -1 */
#define wxSIZE_AUTO_WIDTH       0x0001
/*  Use internally-calculated height if -1 */
#define wxSIZE_AUTO_HEIGHT      0x0002
/*  Use internally-calculated width and height if each is -1 */
#define wxSIZE_AUTO             (wxSIZE_AUTO_WIDTH|wxSIZE_AUTO_HEIGHT)
/*  Ignore missing (-1) dimensions (use existing). */
/*  For readability only: test for wxSIZE_AUTO_WIDTH/HEIGHT in code. */
#define wxSIZE_USE_EXISTING     0x0000
/*  Allow -1 as a valid position */
#define wxSIZE_ALLOW_MINUS_ONE  0x0004
/*  Don't do parent client adjustments (for implementation only) */
#define wxSIZE_NO_ADJUSTMENTS   0x0008
/*  Change the window position even if it seems to be already correct */
#define wxSIZE_FORCE            0x0010
/*  Emit size event even if size didn't change */
#define wxSIZE_FORCE_EVENT      0x0020

/*  ---------------------------------------------------------------------------- */
/*  GDI descriptions */
/*  ---------------------------------------------------------------------------- */

// Hatch styles used by both pen and brush styles.
//
// NB: Do not use these constants directly, they're for internal use only, use
//     wxBRUSHSTYLE_XXX_HATCH and wxPENSTYLE_XXX_HATCH instead.
enum wxHatchStyle
{
    wxHATCHSTYLE_INVALID = -1,

    /*
        The value of the first style is chosen to fit with
        wxDeprecatedGUIConstants values below, don't change it.
     */
    wxHATCHSTYLE_FIRST = 111,
    wxHATCHSTYLE_BDIAGONAL = wxHATCHSTYLE_FIRST,
    wxHATCHSTYLE_CROSSDIAG,
    wxHATCHSTYLE_FDIAGONAL,
    wxHATCHSTYLE_CROSS,
    wxHATCHSTYLE_HORIZONTAL,
    wxHATCHSTYLE_VERTICAL,
    wxHATCHSTYLE_LAST = wxHATCHSTYLE_VERTICAL
};

/*
    WARNING: the following styles are deprecated; use the
             wxFontFamily, wxFontStyle, wxFontWeight, wxBrushStyle,
             wxPenStyle, wxPenCap, wxPenJoin enum values instead!
*/

/* don't use any elements of this enum in the new code */
enum wxDeprecatedGUIConstants
{
    /*  Text font families */
    wxDEFAULT    = 70,
    wxDECORATIVE,
    wxROMAN,
    wxSCRIPT,
    wxSWISS,
    wxMODERN,
    wxTELETYPE,  /* @@@@ */

    /*  Proportional or Fixed width fonts (not yet used) */
    wxVARIABLE   = 80,
    wxFIXED,

    wxNORMAL     = 90,
    wxLIGHT,
    wxBOLD,
    /*  Also wxNORMAL for normal (non-italic text) */
    wxITALIC,
    wxSLANT,

    /*  Pen styles */
    wxSOLID      =   100,
    wxDOT,
    wxLONG_DASH,
    wxSHORT_DASH,
    wxDOT_DASH,
    wxUSER_DASH,

    wxTRANSPARENT,

    /*  Brush & Pen Stippling. Note that a stippled pen cannot be dashed!! */
    /*  Note also that stippling a Pen IS meaningful, because a Line is */
    wxSTIPPLE_MASK_OPAQUE, /* mask is used for blitting monochrome using text fore and back ground colors */
    wxSTIPPLE_MASK,        /* mask is used for masking areas in the stipple bitmap (TO DO) */
    /*  drawn with a Pen, and without any Brush -- and it can be stippled. */
    wxSTIPPLE =          110,

    wxBDIAGONAL_HATCH = wxHATCHSTYLE_BDIAGONAL,
    wxCROSSDIAG_HATCH = wxHATCHSTYLE_CROSSDIAG,
    wxFDIAGONAL_HATCH = wxHATCHSTYLE_FDIAGONAL,
    wxCROSS_HATCH = wxHATCHSTYLE_CROSS,
    wxHORIZONTAL_HATCH = wxHATCHSTYLE_HORIZONTAL,
    wxVERTICAL_HATCH = wxHATCHSTYLE_VERTICAL,
    wxFIRST_HATCH = wxHATCHSTYLE_FIRST,
    wxLAST_HATCH = wxHATCHSTYLE_LAST
};

/*  ToolPanel in wxFrame (VZ: unused?) */
enum
{
    wxTOOL_TOP = 1,
    wxTOOL_BOTTOM,
    wxTOOL_LEFT,
    wxTOOL_RIGHT
};

/*  the values of the format constants should be the same as corresponding */
/*  CF_XXX constants in Windows API */
enum wxDataFormatId
{
    wxDF_INVALID =          0,
    wxDF_TEXT =             1,  /* CF_TEXT */
    wxDF_BITMAP =           2,  /* CF_BITMAP */
    wxDF_METAFILE =         3,  /* CF_METAFILEPICT */
    wxDF_SYLK =             4,
    wxDF_DIF =              5,
    wxDF_TIFF =             6,
    wxDF_OEMTEXT =          7,  /* CF_OEMTEXT */
    wxDF_DIB =              8,  /* CF_DIB */
    wxDF_PALETTE =          9,
    wxDF_PENDATA =          10,
    wxDF_RIFF =             11,
    wxDF_WAVE =             12,
    wxDF_UNICODETEXT =      13,
    wxDF_ENHMETAFILE =      14,
    wxDF_FILENAME =         15, /* CF_HDROP */
    wxDF_LOCALE =           16,
    wxDF_PRIVATE =          20,
    wxDF_HTML =             30, /* Note: does not correspond to CF_ constant */
    wxDF_PNG =              31, /* Note: does not correspond to CF_ constant */
    wxDF_MAX
};

/* Key codes */
enum wxKeyCode
{
    WXK_NONE    =    0,

    WXK_CONTROL_A = 1,
    WXK_CONTROL_B,
    WXK_CONTROL_C,
    WXK_CONTROL_D,
    WXK_CONTROL_E,
    WXK_CONTROL_F,
    WXK_CONTROL_G,
    WXK_CONTROL_H,
    WXK_CONTROL_I,
    WXK_CONTROL_J,
    WXK_CONTROL_K,
    WXK_CONTROL_L,
    WXK_CONTROL_M,
    WXK_CONTROL_N,
    WXK_CONTROL_O,
    WXK_CONTROL_P,
    WXK_CONTROL_Q,
    WXK_CONTROL_R,
    WXK_CONTROL_S,
    WXK_CONTROL_T,
    WXK_CONTROL_U,
    WXK_CONTROL_V,
    WXK_CONTROL_W,
    WXK_CONTROL_X,
    WXK_CONTROL_Y,
    WXK_CONTROL_Z,

    WXK_BACK    =    8, /* backspace */
    WXK_TAB     =    9,
    WXK_RETURN  =    13,
    WXK_ESCAPE  =    27,

    /* values from 33 to 126 are reserved for the standard ASCII characters */

    WXK_SPACE   =    32,
    WXK_DELETE  =    127,

    /* values from 128 to 255 are reserved for ASCII extended characters
       (note that there isn't a single fixed standard for the meaning
       of these values; avoid them in portable apps!) */

    /* These are not compatible with unicode characters.
       If you want to get a unicode character from a key event, use
       wxKeyEvent::GetUnicodeKey                                    */
    WXK_START   = 300,
    WXK_LBUTTON,
    WXK_RBUTTON,
    WXK_CANCEL,
    WXK_MBUTTON,
    WXK_CLEAR,
    WXK_SHIFT,
    WXK_ALT,
    WXK_CONTROL,
    WXK_MENU,
    WXK_PAUSE,
    WXK_CAPITAL,
    WXK_END,
    WXK_HOME,
    WXK_LEFT,
    WXK_UP,
    WXK_RIGHT,
    WXK_DOWN,
    WXK_SELECT,
    WXK_PRINT,
    WXK_EXECUTE,
    WXK_SNAPSHOT,
    WXK_INSERT,
    WXK_HELP,
    WXK_NUMPAD0,
    WXK_NUMPAD1,
    WXK_NUMPAD2,
    WXK_NUMPAD3,
    WXK_NUMPAD4,
    WXK_NUMPAD5,
    WXK_NUMPAD6,
    WXK_NUMPAD7,
    WXK_NUMPAD8,
    WXK_NUMPAD9,
    WXK_MULTIPLY,
    WXK_ADD,
    WXK_SEPARATOR,
    WXK_SUBTRACT,
    WXK_DECIMAL,
    WXK_DIVIDE,
    WXK_F1,
    WXK_F2,
    WXK_F3,
    WXK_F4,
    WXK_F5,
    WXK_F6,
    WXK_F7,
    WXK_F8,
    WXK_F9,
    WXK_F10,
    WXK_F11,
    WXK_F12,
    WXK_F13,
    WXK_F14,
    WXK_F15,
    WXK_F16,
    WXK_F17,
    WXK_F18,
    WXK_F19,
    WXK_F20,
    WXK_F21,
    WXK_F22,
    WXK_F23,
    WXK_F24,
    WXK_NUMLOCK,
    WXK_SCROLL,
    WXK_PAGEUP,
    WXK_PAGEDOWN,
    WXK_NUMPAD_SPACE,
    WXK_NUMPAD_TAB,
    WXK_NUMPAD_ENTER,
    WXK_NUMPAD_F1,
    WXK_NUMPAD_F2,
    WXK_NUMPAD_F3,
    WXK_NUMPAD_F4,
    WXK_NUMPAD_HOME,
    WXK_NUMPAD_LEFT,
    WXK_NUMPAD_UP,
    WXK_NUMPAD_RIGHT,
    WXK_NUMPAD_DOWN,
    WXK_NUMPAD_PAGEUP,
    WXK_NUMPAD_PAGEDOWN,
    WXK_NUMPAD_END,
    WXK_NUMPAD_BEGIN,
    WXK_NUMPAD_INSERT,
    WXK_NUMPAD_DELETE,
    WXK_NUMPAD_EQUAL,
    WXK_NUMPAD_MULTIPLY,
    WXK_NUMPAD_ADD,
    WXK_NUMPAD_SEPARATOR,
    WXK_NUMPAD_SUBTRACT,
    WXK_NUMPAD_DECIMAL,
    WXK_NUMPAD_DIVIDE,

    WXK_WINDOWS_LEFT,
    WXK_WINDOWS_RIGHT,
    WXK_WINDOWS_MENU ,
#ifdef __WXOSX__
    WXK_RAW_CONTROL,
#else
    WXK_RAW_CONTROL = WXK_CONTROL,
#endif
    WXK_COMMAND = WXK_CONTROL,

    /* Hardware-specific buttons */
    WXK_SPECIAL1 = WXK_WINDOWS_MENU + 2, /* Skip WXK_RAW_CONTROL if necessary */
    WXK_SPECIAL2,
    WXK_SPECIAL3,
    WXK_SPECIAL4,
    WXK_SPECIAL5,
    WXK_SPECIAL6,
    WXK_SPECIAL7,
    WXK_SPECIAL8,
    WXK_SPECIAL9,
    WXK_SPECIAL10,
    WXK_SPECIAL11,
    WXK_SPECIAL12,
    WXK_SPECIAL13,
    WXK_SPECIAL14,
    WXK_SPECIAL15,
    WXK_SPECIAL16,
    WXK_SPECIAL17,
    WXK_SPECIAL18,
    WXK_SPECIAL19,
    WXK_SPECIAL20,

    WXK_BROWSER_BACK,
    WXK_BROWSER_FORWARD,
    WXK_BROWSER_REFRESH,
    WXK_BROWSER_STOP,
    WXK_BROWSER_SEARCH,
    WXK_BROWSER_FAVORITES,
    WXK_BROWSER_HOME,
    WXK_VOLUME_MUTE,
    WXK_VOLUME_DOWN,
    WXK_VOLUME_UP,
    WXK_MEDIA_NEXT_TRACK,
    WXK_MEDIA_PREV_TRACK,
    WXK_MEDIA_STOP,
    WXK_MEDIA_PLAY_PAUSE,
    WXK_LAUNCH_MAIL,
    WXK_LAUNCH_APP1,
    WXK_LAUNCH_APP2
};

/* This enum contains bit mask constants used in wxKeyEvent */
enum wxKeyModifier
{
    wxMOD_NONE      = 0x0000,
    wxMOD_ALT       = 0x0001,
    wxMOD_CONTROL   = 0x0002,
    wxMOD_ALTGR     = wxMOD_ALT | wxMOD_CONTROL,
    wxMOD_SHIFT     = 0x0004,
    wxMOD_META      = 0x0008,
    wxMOD_WIN       = wxMOD_META,
#if defined(__WXMAC__)
    wxMOD_RAW_CONTROL = 0x0010,
#else
    wxMOD_RAW_CONTROL = wxMOD_CONTROL,
#endif
    wxMOD_CMD       = wxMOD_CONTROL,
    wxMOD_ALL       = 0xffff
};

/* Shortcut for easier dialog-unit-to-pixel conversion */
#define wxDLG_UNIT(parent, pt) parent->ConvertDialogToPixels(pt)

/* Paper types */
enum wxPaperSize
{
    wxPAPER_NONE,               /*  Use specific dimensions */
    wxPAPER_LETTER,             /*  Letter, 8 1/2 by 11 inches */
    wxPAPER_LEGAL,              /*  Legal, 8 1/2 by 14 inches */
    wxPAPER_A4,                 /*  A4 Sheet, 210 by 297 millimeters */
    wxPAPER_CSHEET,             /*  C Sheet, 17 by 22 inches */
    wxPAPER_DSHEET,             /*  D Sheet, 22 by 34 inches */
    wxPAPER_ESHEET,             /*  E Sheet, 34 by 44 inches */
    wxPAPER_LETTERSMALL,        /*  Letter Small, 8 1/2 by 11 inches */
    wxPAPER_TABLOID,            /*  Tabloid, 11 by 17 inches */
    wxPAPER_LEDGER,             /*  Ledger, 17 by 11 inches */
    wxPAPER_STATEMENT,          /*  Statement, 5 1/2 by 8 1/2 inches */
    wxPAPER_EXECUTIVE,          /*  Executive, 7 1/4 by 10 1/2 inches */
    wxPAPER_A3,                 /*  A3 sheet, 297 by 420 millimeters */
    wxPAPER_A4SMALL,            /*  A4 small sheet, 210 by 297 millimeters */
    wxPAPER_A5,                 /*  A5 sheet, 148 by 210 millimeters */
    wxPAPER_B4,                 /*  B4 sheet, 250 by 354 millimeters */
    wxPAPER_B5,                 /*  B5 sheet, 182-by-257-millimeter paper */
    wxPAPER_FOLIO,              /*  Folio, 8-1/2-by-13-inch paper */
    wxPAPER_QUARTO,             /*  Quarto, 215-by-275-millimeter paper */
    wxPAPER_10X14,              /*  10-by-14-inch sheet */
    wxPAPER_11X17,              /*  11-by-17-inch sheet */
    wxPAPER_NOTE,               /*  Note, 8 1/2 by 11 inches */
    wxPAPER_ENV_9,              /*  #9 Envelope, 3 7/8 by 8 7/8 inches */
    wxPAPER_ENV_10,             /*  #10 Envelope, 4 1/8 by 9 1/2 inches */
    wxPAPER_ENV_11,             /*  #11 Envelope, 4 1/2 by 10 3/8 inches */
    wxPAPER_ENV_12,             /*  #12 Envelope, 4 3/4 by 11 inches */
    wxPAPER_ENV_14,             /*  #14 Envelope, 5 by 11 1/2 inches */
    wxPAPER_ENV_DL,             /*  DL Envelope, 110 by 220 millimeters */
    wxPAPER_ENV_C5,             /*  C5 Envelope, 162 by 229 millimeters */
    wxPAPER_ENV_C3,             /*  C3 Envelope, 324 by 458 millimeters */
    wxPAPER_ENV_C4,             /*  C4 Envelope, 229 by 324 millimeters */
    wxPAPER_ENV_C6,             /*  C6 Envelope, 114 by 162 millimeters */
    wxPAPER_ENV_C65,            /*  C65 Envelope, 114 by 229 millimeters */
    wxPAPER_ENV_B4,             /*  B4 Envelope, 250 by 353 millimeters */
    wxPAPER_ENV_B5,             /*  B5 Envelope, 176 by 250 millimeters */
    wxPAPER_ENV_B6,             /*  B6 Envelope, 176 by 125 millimeters */
    wxPAPER_ENV_ITALY,          /*  Italy Envelope, 110 by 230 millimeters */
    wxPAPER_ENV_MONARCH,        /*  Monarch Envelope, 3 7/8 by 7 1/2 inches */
    wxPAPER_ENV_PERSONAL,       /*  6 3/4 Envelope, 3 5/8 by 6 1/2 inches */
    wxPAPER_FANFOLD_US,         /*  US Std Fanfold, 14 7/8 by 11 inches */
    wxPAPER_FANFOLD_STD_GERMAN, /*  German Std Fanfold, 8 1/2 by 12 inches */
    wxPAPER_FANFOLD_LGL_GERMAN, /*  German Legal Fanfold, 8 1/2 by 13 inches */

    wxPAPER_ISO_B4,             /*  B4 (ISO) 250 x 353 mm */
    wxPAPER_JAPANESE_POSTCARD,  /*  Japanese Postcard 100 x 148 mm */
    wxPAPER_9X11,               /*  9 x 11 in */
    wxPAPER_10X11,              /*  10 x 11 in */
    wxPAPER_15X11,              /*  15 x 11 in */
    wxPAPER_ENV_INVITE,         /*  Envelope Invite 220 x 220 mm */
    wxPAPER_LETTER_EXTRA,       /*  Letter Extra 9 \275 x 12 in */
    wxPAPER_LEGAL_EXTRA,        /*  Legal Extra 9 \275 x 15 in */
    wxPAPER_TABLOID_EXTRA,      /*  Tabloid Extra 11.69 x 18 in */
    wxPAPER_A4_EXTRA,           /*  A4 Extra 9.27 x 12.69 in */
    wxPAPER_LETTER_TRANSVERSE,  /*  Letter Transverse 8 \275 x 11 in */
    wxPAPER_A4_TRANSVERSE,      /*  A4 Transverse 210 x 297 mm */
    wxPAPER_LETTER_EXTRA_TRANSVERSE, /*  Letter Extra Transverse 9\275 x 12 in */
    wxPAPER_A_PLUS,             /*  SuperA/SuperA/A4 227 x 356 mm */
    wxPAPER_B_PLUS,             /*  SuperB/SuperB/A3 305 x 487 mm */
    wxPAPER_LETTER_PLUS,        /*  Letter Plus 8.5 x 12.69 in */
    wxPAPER_A4_PLUS,            /*  A4 Plus 210 x 330 mm */
    wxPAPER_A5_TRANSVERSE,      /*  A5 Transverse 148 x 210 mm */
    wxPAPER_B5_TRANSVERSE,      /*  B5 (JIS) Transverse 182 x 257 mm */
    wxPAPER_A3_EXTRA,           /*  A3 Extra 322 x 445 mm */
    wxPAPER_A5_EXTRA,           /*  A5 Extra 174 x 235 mm */
    wxPAPER_B5_EXTRA,           /*  B5 (ISO) Extra 201 x 276 mm */
    wxPAPER_A2,                 /*  A2 420 x 594 mm */
    wxPAPER_A3_TRANSVERSE,      /*  A3 Transverse 297 x 420 mm */
    wxPAPER_A3_EXTRA_TRANSVERSE, /*  A3 Extra Transverse 322 x 445 mm */

    wxPAPER_DBL_JAPANESE_POSTCARD,/* Japanese Double Postcard 200 x 148 mm */
    wxPAPER_A6,                 /* A6 105 x 148 mm */
    wxPAPER_JENV_KAKU2,         /* Japanese Envelope Kaku #2 */
    wxPAPER_JENV_KAKU3,         /* Japanese Envelope Kaku #3 */
    wxPAPER_JENV_CHOU3,         /* Japanese Envelope Chou #3 */
    wxPAPER_JENV_CHOU4,         /* Japanese Envelope Chou #4 */
    wxPAPER_LETTER_ROTATED,     /* Letter Rotated 11 x 8 1/2 in */
    wxPAPER_A3_ROTATED,         /* A3 Rotated 420 x 297 mm */
    wxPAPER_A4_ROTATED,         /* A4 Rotated 297 x 210 mm */
    wxPAPER_A5_ROTATED,         /* A5 Rotated 210 x 148 mm */
    wxPAPER_B4_JIS_ROTATED,     /* B4 (JIS) Rotated 364 x 257 mm */
    wxPAPER_B5_JIS_ROTATED,     /* B5 (JIS) Rotated 257 x 182 mm */
    wxPAPER_JAPANESE_POSTCARD_ROTATED,/* Japanese Postcard Rotated 148 x 100 mm */
    wxPAPER_DBL_JAPANESE_POSTCARD_ROTATED,/* Double Japanese Postcard Rotated 148 x 200 mm */
    wxPAPER_A6_ROTATED,         /* A6 Rotated 148 x 105 mm */
    wxPAPER_JENV_KAKU2_ROTATED, /* Japanese Envelope Kaku #2 Rotated */
    wxPAPER_JENV_KAKU3_ROTATED, /* Japanese Envelope Kaku #3 Rotated */
    wxPAPER_JENV_CHOU3_ROTATED, /* Japanese Envelope Chou #3 Rotated */
    wxPAPER_JENV_CHOU4_ROTATED, /* Japanese Envelope Chou #4 Rotated */
    wxPAPER_B6_JIS,             /* B6 (JIS) 128 x 182 mm */
    wxPAPER_B6_JIS_ROTATED,     /* B6 (JIS) Rotated 182 x 128 mm */
    wxPAPER_12X11,              /* 12 x 11 in */
    wxPAPER_JENV_YOU4,          /* Japanese Envelope You #4 */
    wxPAPER_JENV_YOU4_ROTATED,  /* Japanese Envelope You #4 Rotated */
    wxPAPER_P16K,               /* PRC 16K 146 x 215 mm */
    wxPAPER_P32K,               /* PRC 32K 97 x 151 mm */
    wxPAPER_P32KBIG,            /* PRC 32K(Big) 97 x 151 mm */
    wxPAPER_PENV_1,             /* PRC Envelope #1 102 x 165 mm */
    wxPAPER_PENV_2,             /* PRC Envelope #2 102 x 176 mm */
    wxPAPER_PENV_3,             /* PRC Envelope #3 125 x 176 mm */
    wxPAPER_PENV_4,             /* PRC Envelope #4 110 x 208 mm */
    wxPAPER_PENV_5,             /* PRC Envelope #5 110 x 220 mm */
    wxPAPER_PENV_6,             /* PRC Envelope #6 120 x 230 mm */
    wxPAPER_PENV_7,             /* PRC Envelope #7 160 x 230 mm */
    wxPAPER_PENV_8,             /* PRC Envelope #8 120 x 309 mm */
    wxPAPER_PENV_9,             /* PRC Envelope #9 229 x 324 mm */
    wxPAPER_PENV_10,            /* PRC Envelope #10 324 x 458 mm */
    wxPAPER_P16K_ROTATED,       /* PRC 16K Rotated */
    wxPAPER_P32K_ROTATED,       /* PRC 32K Rotated */
    wxPAPER_P32KBIG_ROTATED,    /* PRC 32K(Big) Rotated */
    wxPAPER_PENV_1_ROTATED,     /* PRC Envelope #1 Rotated 165 x 102 mm */
    wxPAPER_PENV_2_ROTATED,     /* PRC Envelope #2 Rotated 176 x 102 mm */
    wxPAPER_PENV_3_ROTATED,     /* PRC Envelope #3 Rotated 176 x 125 mm */
    wxPAPER_PENV_4_ROTATED,     /* PRC Envelope #4 Rotated 208 x 110 mm */
    wxPAPER_PENV_5_ROTATED,     /* PRC Envelope #5 Rotated 220 x 110 mm */
    wxPAPER_PENV_6_ROTATED,     /* PRC Envelope #6 Rotated 230 x 120 mm */
    wxPAPER_PENV_7_ROTATED,     /* PRC Envelope #7 Rotated 230 x 160 mm */
    wxPAPER_PENV_8_ROTATED,     /* PRC Envelope #8 Rotated 309 x 120 mm */
    wxPAPER_PENV_9_ROTATED,     /* PRC Envelope #9 Rotated 324 x 229 mm */
    wxPAPER_PENV_10_ROTATED,    /* PRC Envelope #10 Rotated 458 x 324 m */
    wxPAPER_A0,                 /* A0 Sheet 841 x 1189 mm */
    wxPAPER_A1                  /* A1 Sheet 594 x 841 mm */
};

/* Printing orientation */
enum wxPrintOrientation
{
   wxPORTRAIT = 1,
   wxLANDSCAPE
};

/* Duplex printing modes
 */

enum wxDuplexMode
{
    wxDUPLEX_SIMPLEX, /*  Non-duplex */
    wxDUPLEX_HORIZONTAL,
    wxDUPLEX_VERTICAL
};

/* Print quality.
 */

#define wxPRINT_QUALITY_HIGH    -1
#define wxPRINT_QUALITY_MEDIUM  -2
#define wxPRINT_QUALITY_LOW     -3
#define wxPRINT_QUALITY_DRAFT   -4

typedef int wxPrintQuality;

/* Print mode (currently PostScript only)
 */

enum wxPrintMode
{
    wxPRINT_MODE_NONE =    0,
    wxPRINT_MODE_PREVIEW = 1,   /*  Preview in external application */
    wxPRINT_MODE_FILE =    2,   /*  Print to file */
    wxPRINT_MODE_PRINTER = 3,   /*  Send to printer */
    wxPRINT_MODE_STREAM =  4    /*  Send postscript data into a stream */
};

/*  ---------------------------------------------------------------------------- */
/*  UpdateWindowUI flags */
/*  ---------------------------------------------------------------------------- */

enum wxUpdateUI
{
    wxUPDATE_UI_NONE          = 0x0000,
    wxUPDATE_UI_RECURSE       = 0x0001,
    wxUPDATE_UI_FROMIDLE      = 0x0002 /*  Invoked from On(Internal)Idle */
};


/* ---------------------------------------------------------------------------- */
/* wxList types */
/* ---------------------------------------------------------------------------- */

/* type of compare function for list sort operation (as in 'qsort'): it should
   return a negative value, 0 or positive value if the first element is less
   than, equal or greater than the second */

typedef int (* LINKAGEMODE wxSortCompareFunction)(const void *elem1, const void *elem2);

/* wxList iterator function */
typedef int (* LINKAGEMODE wxListIterateFunction)(void *current);


/*  ---------------------------------------------------------------------------- */
/*  miscellaneous */
/*  ---------------------------------------------------------------------------- */

/*  define this macro if font handling is done using the X font names */
#if (defined(__WXGTK__) && !defined(__WXGTK20__)) || defined(__X__)
    #define _WX_X_FONTLIKE
#endif

/*  macro to specify "All Files" on different platforms */
#if defined(__WXMSW__)
#   define wxALL_FILES_PATTERN   wxT("*.*")
#   define wxALL_FILES           gettext_noop("All files (*.*)|*.*")
#else
#   define wxALL_FILES_PATTERN   wxT("*")
#   define wxALL_FILES           gettext_noop("All files (*)|*")
#endif

#if defined(__CYGWIN__) && defined(__WXMSW__)
#   if wxUSE_STD_CONTAINERS || defined(wxUSE_STD_STRING)
         /*
            NASTY HACK because the gethostname in sys/unistd.h which the gnu
            stl includes and wx builds with by default clash with each other
            (windows version 2nd param is int, sys/unistd.h version is unsigned
            int).
          */
#        define gethostname gethostnameHACK
#        include <unistd.h>
#        undef gethostname
#   endif
#endif

/*  --------------------------------------------------------------------------- */
/*  macros that enable wxWidgets apps to be compiled in absence of the */
/*  system headers, although some platform specific types are used in the */
/*  platform specific (implementation) parts of the headers */
/*  --------------------------------------------------------------------------- */

#ifdef __DARWIN__
#define DECLARE_WXOSX_OPAQUE_CFREF( name ) typedef struct __##name* name##Ref;
#define DECLARE_WXOSX_OPAQUE_CONST_CFREF( name ) typedef const struct __##name* name##Ref;
#endif

#ifdef __WXMAC__

#define WX_OPAQUE_TYPE( name ) struct wxOpaque##name

typedef void*       WXHCURSOR;
typedef void*       WXRECTPTR;
typedef void*       WXPOINTPTR;
typedef void*       WXHWND;
typedef void*       WXEVENTREF;
typedef void*       WXEVENTHANDLERREF;
typedef void*       WXEVENTHANDLERCALLREF;
typedef void*       WXAPPLEEVENTREF;

typedef unsigned int    WXUINT;
typedef unsigned long   WXDWORD;
typedef unsigned short  WXWORD;

typedef WX_OPAQUE_TYPE(PicHandle ) * WXHMETAFILE ;

typedef void*       WXDisplay;

/*
 * core frameworks
 */

typedef const void * CFTypeRef;

DECLARE_WXOSX_OPAQUE_CONST_CFREF( CFData )
DECLARE_WXOSX_OPAQUE_CONST_CFREF( CFString )
typedef struct __CFString * CFMutableStringRef;
DECLARE_WXOSX_OPAQUE_CONST_CFREF( CFDictionary )

DECLARE_WXOSX_OPAQUE_CONST_CFREF( CFArray )
typedef struct __CFArray * CFMutableArrayRef;

DECLARE_WXOSX_OPAQUE_CFREF( CFRunLoopSource )
DECLARE_WXOSX_OPAQUE_CONST_CFREF( CTFont )
DECLARE_WXOSX_OPAQUE_CONST_CFREF( CTFontDescriptor )

#define DECLARE_WXOSX_OPAQUE_CGREF( name ) typedef struct name* name##Ref;

DECLARE_WXOSX_OPAQUE_CGREF( CGColor )
DECLARE_WXOSX_OPAQUE_CGREF( CGImage )
DECLARE_WXOSX_OPAQUE_CGREF( CGContext )
DECLARE_WXOSX_OPAQUE_CGREF( CGFont )

typedef CGColorRef    WXCOLORREF;
typedef CGImageRef    WXCGIMAGEREF;
typedef CGContextRef  WXHDC;
typedef CGContextRef  WXHBITMAP;

/*
 * carbon
 */

typedef const struct __HIShape * HIShapeRef;
typedef struct __HIShape * HIMutableShapeRef;

#define DECLARE_WXMAC_OPAQUE_REF( name ) typedef struct Opaque##name* name;

DECLARE_WXMAC_OPAQUE_REF( PasteboardRef )
DECLARE_WXMAC_OPAQUE_REF( IconRef )
DECLARE_WXMAC_OPAQUE_REF( MenuRef )

typedef IconRef WXHICON ;
typedef HIShapeRef WXHRGN;

#endif // __WXMAC__

#if defined(__WXMAC__)

/* Definitions of 32-bit/64-bit types
 * These are typedef'd exactly the same way in newer OS X headers so
 * redefinition when real headers are included should not be a problem.  If
 * it is, the types are being defined wrongly here.
 * The purpose of these types is so they can be used from public wx headers.
 * and also because the older (pre-Leopard) headers don't define them.
 */

/* NOTE: We don't pollute namespace with CGFLOAT_MIN/MAX/IS_DOUBLE macros
 * since they are unlikely to be needed in a public header.
 */
#if defined(__LP64__) && __LP64__
    typedef double CGFloat;
#else
    typedef float CGFloat;
#endif

#if (defined(__LP64__) && __LP64__) || (defined(NS_BUILD_32_LIKE_64) && NS_BUILD_32_LIKE_64)
typedef long NSInteger;
typedef unsigned long NSUInteger;
#else
typedef int NSInteger;
typedef unsigned int NSUInteger;
#endif

/* Objective-C type declarations.
 * These are to be used in public headers in lieu of NSSomething* because
 * Objective-C class names are not available in C/C++ code.
 */

/*  NOTE: This ought to work with other compilers too, but I'm being cautious */
#if (defined(__GNUC__) && defined(__APPLE__))
/* It's desirable to have type safety for Objective-C(++) code as it does
at least catch typos of method names among other things.  However, it
is not possible to declare an Objective-C class from plain old C or C++
code.  Furthermore, because of C++ name mangling, the type name must
be the same for both C++ and Objective-C++ code.  Therefore, we define
what should be a pointer to an Objective-C class as a pointer to a plain
old C struct with the same name.  Unfortunately, because the compiler
does not see a struct as an Objective-C class we cannot declare it
as a struct in Objective-C(++) mode.
*/
#if defined(__OBJC__)
#define DECLARE_WXCOCOA_OBJC_CLASS(klass) \
@class klass; \
typedef klass *WX_##klass
#else /*  not defined(__OBJC__) */
#define DECLARE_WXCOCOA_OBJC_CLASS(klass) \
typedef struct klass *WX_##klass
#endif /*  defined(__OBJC__) */

#else /*  not Apple's gcc */
#warning "Objective-C types will not be checked by the compiler."
/*  NOTE: typedef struct objc_object *id; */
/*  IOW, we're declaring these using the id type without using that name, */
/*  since "id" is used extensively not only within wxWidgets itself, but */
/*  also in wxWidgets application code.  The following works fine when */
/*  compiling C(++) code, and works without typesafety for Obj-C(++) code */
#define DECLARE_WXCOCOA_OBJC_CLASS(klass) \
typedef struct objc_object *WX_##klass

#endif /*  (defined(__GNUC__) && defined(__APPLE__)) */

DECLARE_WXCOCOA_OBJC_CLASS(NSArray);
DECLARE_WXCOCOA_OBJC_CLASS(NSData);
DECLARE_WXCOCOA_OBJC_CLASS(NSMutableArray);
DECLARE_WXCOCOA_OBJC_CLASS(NSString);
DECLARE_WXCOCOA_OBJC_CLASS(NSObject);

#if wxOSX_USE_COCOA

DECLARE_WXCOCOA_OBJC_CLASS(NSApplication);
DECLARE_WXCOCOA_OBJC_CLASS(NSBitmapImageRep);
DECLARE_WXCOCOA_OBJC_CLASS(NSBox);
DECLARE_WXCOCOA_OBJC_CLASS(NSButton);
DECLARE_WXCOCOA_OBJC_CLASS(NSColor);
DECLARE_WXCOCOA_OBJC_CLASS(NSColorPanel);
DECLARE_WXCOCOA_OBJC_CLASS(NSControl);
DECLARE_WXCOCOA_OBJC_CLASS(NSCursor);
DECLARE_WXCOCOA_OBJC_CLASS(NSEvent);
DECLARE_WXCOCOA_OBJC_CLASS(NSFont);
DECLARE_WXCOCOA_OBJC_CLASS(NSFontDescriptor);
DECLARE_WXCOCOA_OBJC_CLASS(NSFontPanel);
DECLARE_WXCOCOA_OBJC_CLASS(NSImage);
DECLARE_WXCOCOA_OBJC_CLASS(NSLayoutManager);
DECLARE_WXCOCOA_OBJC_CLASS(NSMenu);
DECLARE_WXCOCOA_OBJC_CLASS(NSMenuExtra);
DECLARE_WXCOCOA_OBJC_CLASS(NSMenuItem);
DECLARE_WXCOCOA_OBJC_CLASS(NSNotification);
DECLARE_WXCOCOA_OBJC_CLASS(NSPanel);
DECLARE_WXCOCOA_OBJC_CLASS(NSResponder);
DECLARE_WXCOCOA_OBJC_CLASS(NSScrollView);
DECLARE_WXCOCOA_OBJC_CLASS(NSSound);
DECLARE_WXCOCOA_OBJC_CLASS(NSStatusItem);
DECLARE_WXCOCOA_OBJC_CLASS(NSTableColumn);
DECLARE_WXCOCOA_OBJC_CLASS(NSTableView);
DECLARE_WXCOCOA_OBJC_CLASS(NSTextContainer);
DECLARE_WXCOCOA_OBJC_CLASS(NSTextField);
DECLARE_WXCOCOA_OBJC_CLASS(NSTextStorage);
DECLARE_WXCOCOA_OBJC_CLASS(NSThread);
DECLARE_WXCOCOA_OBJC_CLASS(NSWindow);
DECLARE_WXCOCOA_OBJC_CLASS(NSView);
DECLARE_WXCOCOA_OBJC_CLASS(NSOpenGLContext);
DECLARE_WXCOCOA_OBJC_CLASS(NSOpenGLPixelFormat);
DECLARE_WXCOCOA_OBJC_CLASS(NSPrintInfo);
DECLARE_WXCOCOA_OBJC_CLASS(NSGestureRecognizer);
DECLARE_WXCOCOA_OBJC_CLASS(NSPanGestureRecognizer);
DECLARE_WXCOCOA_OBJC_CLASS(NSMagnificationGestureRecognizer);
DECLARE_WXCOCOA_OBJC_CLASS(NSRotationGestureRecognizer);
DECLARE_WXCOCOA_OBJC_CLASS(NSPressGestureRecognizer);
DECLARE_WXCOCOA_OBJC_CLASS(NSTouch);
DECLARE_WXCOCOA_OBJC_CLASS(NSPasteboard);
DECLARE_WXCOCOA_OBJC_CLASS(WKWebView);

typedef WX_NSWindow WXWindow;
typedef WX_NSView WXWidget;
typedef WX_NSImage WXImage;
typedef WX_NSMenu WXHMENU;
typedef WX_NSOpenGLPixelFormat WXGLPixelFormat;
typedef WX_NSOpenGLContext WXGLContext;
typedef WX_NSPasteboard OSXPasteboard;
typedef WX_WKWebView OSXWebViewPtr;

#elif wxOSX_USE_IPHONE

DECLARE_WXCOCOA_OBJC_CLASS(UIMenu);
DECLARE_WXCOCOA_OBJC_CLASS(UIMenuItem);
DECLARE_WXCOCOA_OBJC_CLASS(UIWindow);
DECLARE_WXCOCOA_OBJC_CLASS(UImage);
DECLARE_WXCOCOA_OBJC_CLASS(UIView);
DECLARE_WXCOCOA_OBJC_CLASS(UIFont);
DECLARE_WXCOCOA_OBJC_CLASS(UIImage);
DECLARE_WXCOCOA_OBJC_CLASS(UIEvent);
DECLARE_WXCOCOA_OBJC_CLASS(NSSet);
DECLARE_WXCOCOA_OBJC_CLASS(EAGLContext);
DECLARE_WXCOCOA_OBJC_CLASS(UIPasteboard);

typedef WX_UIWindow WXWindow;
typedef WX_UIView WXWidget;
typedef WX_UIImage WXImage;
typedef WX_UIMenu WXHMENU;
typedef WX_EAGLContext WXGLContext;
typedef WX_NSString WXGLPixelFormat;
typedef WX_UIPasteboard WXOSXPasteboard;

#endif



#endif /* __WXMAC__ */

/* ABX: check __WIN32__ instead of __WXMSW__ for the same MSWBase in any Win32 port */
#if defined(__WIN32__)

/*  Stand-ins for Windows types to avoid #including all of windows.h */

#ifndef NO_STRICT
    #define WX_MSW_DECLARE_HANDLE(type) typedef struct type##__ * WX##type
#else
    #define WX_MSW_DECLARE_HANDLE(type) typedef void * WX##type
#endif

typedef void* WXHANDLE;
WX_MSW_DECLARE_HANDLE(HWND);
WX_MSW_DECLARE_HANDLE(HICON);
WX_MSW_DECLARE_HANDLE(HFONT);
WX_MSW_DECLARE_HANDLE(HMENU);
WX_MSW_DECLARE_HANDLE(HPEN);
WX_MSW_DECLARE_HANDLE(HBRUSH);
WX_MSW_DECLARE_HANDLE(HPALETTE);
WX_MSW_DECLARE_HANDLE(HCURSOR);
WX_MSW_DECLARE_HANDLE(HRGN);
WX_MSW_DECLARE_HANDLE(RECTPTR);
WX_MSW_DECLARE_HANDLE(HACCEL);
WX_MSW_DECLARE_HANDLE(HINSTANCE);
WX_MSW_DECLARE_HANDLE(HBITMAP);
WX_MSW_DECLARE_HANDLE(HIMAGELIST);
WX_MSW_DECLARE_HANDLE(HGLOBAL);
WX_MSW_DECLARE_HANDLE(HDC);
WX_MSW_DECLARE_HANDLE(DPI_AWARENESS_CONTEXT);
typedef WXHINSTANCE WXHMODULE;

#undef WX_MSW_DECLARE_HANDLE

typedef unsigned int    WXUINT;
typedef unsigned long   WXDWORD;
typedef unsigned short  WXWORD;

typedef unsigned long   WXCOLORREF;
typedef void *          WXRGNDATA;
typedef struct tagMSG   WXMSG;
typedef void *          WXHCONV;
typedef void *          WXHKEY;
typedef void *          WXHTREEITEM;

typedef void *          WXDRAWITEMSTRUCT;
typedef void *          WXMEASUREITEMSTRUCT;
typedef void *          WXLPCREATESTRUCT;

#ifdef __WXMSW__
typedef WXHWND          WXWidget;
#endif

#ifdef __WIN64__
typedef wxUint64           WXWPARAM;
typedef wxInt64            WXLPARAM;
typedef wxInt64            WXLRESULT;
#else
typedef wxW64 unsigned int WXWPARAM;
typedef wxW64 long         WXLPARAM;
typedef wxW64 long         WXLRESULT;
#endif

/*
   This is defined for compatibility only, it's not really the same thing as
   FARPROC.
 */
#if defined(__GNUWIN32__)
typedef int             (*WXFARPROC)();
#else
typedef int             (__stdcall *WXFARPROC)();
#endif

typedef WXLRESULT (wxSTDCALL *WXWNDPROC)(WXHWND, WXUINT, WXWPARAM, WXLPARAM);

#endif /*  __WIN32__ */


#if defined(__WXMOTIF__) || defined(__WXX11__)
/* Stand-ins for X/Xt/Motif types */
typedef void*           WXWindow;
typedef void*           WXWidget;
typedef void*           WXAppContext;
typedef void*           WXColormap;
typedef void*           WXColor;
typedef void            WXDisplay;
typedef void            WXEvent;
typedef void*           WXCursor;
typedef void*           WXPixmap;
typedef void*           WXFontStructPtr;
typedef void*           WXGC;
typedef void*           WXRegion;
typedef void*           WXFont;
typedef void*           WXImage;
typedef void*           WXFontList;
typedef void*           WXFontSet;
typedef void*           WXRendition;
typedef void*           WXRenderTable;
typedef void*           WXFontType; /* either a XmFontList or XmRenderTable */
typedef void*           WXString;

typedef unsigned long   Atom;  /* this might fail on a few architectures */
typedef long            WXPixel; /* safety catch in src/motif/colour.cpp */

#endif /*  Motif */

#ifdef __WXGTK__

/* Stand-ins for GLIB types */
typedef struct _GSList GSList;

/* Stand-ins for GDK types */
typedef struct _GdkColor        GdkColor;
typedef struct _GdkCursor       GdkCursor;
typedef struct _GdkDragContext  GdkDragContext;

#if defined(__WXGTK20__)
    typedef struct _GdkAtom* GdkAtom;
#else
    typedef unsigned long GdkAtom;
#endif

#if !defined(__WXGTK3__)
    typedef struct _GdkColormap GdkColormap;
    typedef struct _GdkFont GdkFont;
    typedef struct _GdkGC GdkGC;
    typedef struct _GdkRegion GdkRegion;
#endif

#if defined(__WXGTK3__)
    typedef struct _GdkWindow GdkWindow;
    typedef struct _GdkEventSequence GdkEventSequence;
#elif defined(__WXGTK20__)
    typedef struct _GdkDrawable GdkWindow;
    typedef struct _GdkDrawable GdkPixmap;
#else
    typedef struct _GdkWindow GdkWindow;
    typedef struct _GdkWindow GdkBitmap;
    typedef struct _GdkWindow GdkPixmap;
#endif

/* Stand-ins for GTK types */
typedef struct _GtkWidget         GtkWidget;
typedef struct _GtkRcStyle        GtkRcStyle;
typedef struct _GtkAdjustment     GtkAdjustment;
typedef struct _GtkToolbar        GtkToolbar;
typedef struct _GtkNotebook       GtkNotebook;
typedef struct _GtkNotebookPage   GtkNotebookPage;
typedef struct _GtkAccelGroup     GtkAccelGroup;
typedef struct _GtkSelectionData  GtkSelectionData;
typedef struct _GtkTextBuffer     GtkTextBuffer;
typedef struct _GtkRange          GtkRange;
typedef struct _GtkCellRenderer   GtkCellRenderer;

typedef GtkWidget *WXWidget;

#ifndef __WXGTK20__
#define GTK_OBJECT_GET_CLASS(object) (GTK_OBJECT(object)->klass)
#define GTK_CLASS_TYPE(klass) ((klass)->type)
#endif

#endif /*  __WXGTK__ */

#if defined(__WXGTK20__) || (defined(__WXX11__) && wxUSE_UNICODE)
#define wxUSE_PANGO 1
#else
#define wxUSE_PANGO 0
#endif

#if wxUSE_PANGO
/* Stand-ins for Pango types */
typedef struct _PangoContext         PangoContext;
typedef struct _PangoLayout          PangoLayout;
typedef struct _PangoFontDescription PangoFontDescription;
#endif

#ifdef __WXDFB__
/* DirectFB doesn't have the concept of non-TLW window, so use
   something arbitrary */
typedef const void* WXWidget;
#endif /*  DFB */

#ifdef __WXQT__
#include "wx/qt/defs.h"
#endif

/*  include the feature test macros */
#include "wx/features.h"

/*  --------------------------------------------------------------------------- */
/*  macros to define a class without copy ctor nor assignment operator */
/*  --------------------------------------------------------------------------- */

#if defined(__cplusplus) && (__cplusplus >= 201103L || wxCHECK_VISUALC_VERSION(14))
    #define wxMEMBER_DELETE = delete
#else
    #define wxMEMBER_DELETE
#endif

#define wxDECLARE_NO_COPY_CLASS(classname)      \
    private:                                    \
        classname(const classname&) wxMEMBER_DELETE; \
        classname& operator=(const classname&) wxMEMBER_DELETE

#define wxDECLARE_NO_COPY_TEMPLATE_CLASS(classname, arg)  \
    private:                                              \
        classname(const classname<arg>&) wxMEMBER_DELETE; \
        classname& operator=(const classname<arg>&) wxMEMBER_DELETE

#define wxDECLARE_NO_COPY_TEMPLATE_CLASS_2(classname, arg1, arg2) \
    private:                                                      \
        classname(const classname<arg1, arg2>&) wxMEMBER_DELETE; \
        classname& operator=(const classname<arg1, arg2>&) wxMEMBER_DELETE

#define wxDECLARE_NO_ASSIGN_CLASS(classname)    \
    private:                                    \
        classname& operator=(const classname&) wxMEMBER_DELETE

/* deprecated variants _not_ requiring a semicolon after them */
#define DECLARE_NO_COPY_CLASS(classname) \
    wxDECLARE_NO_COPY_CLASS(classname);
#define DECLARE_NO_COPY_TEMPLATE_CLASS(classname, arg) \
    wxDECLARE_NO_COPY_TEMPLATE_CLASS(classname, arg);
#define DECLARE_NO_ASSIGN_CLASS(classname) \
    wxDECLARE_NO_ASSIGN_CLASS(classname);

/*  --------------------------------------------------------------------------- */
/*  If a manifest is being automatically generated, add common controls 6 to it */
/*  --------------------------------------------------------------------------- */

#if wxUSE_GUI && \
    (!defined wxUSE_NO_MANIFEST || wxUSE_NO_MANIFEST == 0 ) && \
    ( defined _MSC_FULL_VER && _MSC_FULL_VER >= 140040130 )

#define WX_CC_MANIFEST(cpu)                     \
    "/manifestdependency:\"type='win32'         \
     name='Microsoft.Windows.Common-Controls'   \
     version='6.0.0.0'                          \
     processorArchitecture='" cpu "'            \
     publicKeyToken='6595b64144ccf1df'          \
     language='*'\""

#if defined _M_IX86
    #pragma comment(linker, WX_CC_MANIFEST("x86"))
#elif defined _M_X64
    #pragma comment(linker, WX_CC_MANIFEST("amd64"))
#elif defined _M_ARM64
    #pragma comment(linker, WX_CC_MANIFEST("arm64"))
#elif defined _M_IA64
    #pragma comment(linker, WX_CC_MANIFEST("ia64"))
#else
    #pragma comment(linker, WX_CC_MANIFEST("*"))
#endif

#endif /* !wxUSE_NO_MANIFEST && _MSC_FULL_VER >= 140040130 */

/* wxThread and wxProcess priorities */
enum
{
    wxPRIORITY_MIN     = 0u,   /* lowest possible priority */
    wxPRIORITY_DEFAULT = 50u,  /* normal priority */
    wxPRIORITY_MAX     = 100u  /* highest possible priority */
};

#endif
    /*  _WX_DEFS_H_ */
