/**
* Name:        wx/math.h
* Purpose:     Declarations/definitions of common math functions
* Author:      John Labenski and others
* Modified by:
* Created:     02/02/03
* Copyright:   (c) John Labenski
* Licence:     wxWindows licence
*/

/* THIS IS A C FILE, DON'T USE C++ FEATURES (IN PARTICULAR COMMENTS) IN IT */

#ifndef _WX_MATH_H_
#define _WX_MATH_H_

#include "wx/defs.h"

#include <math.h>

#ifndef M_PI
    #define M_PI 3.1415926535897932384626433832795
#endif

/* Scaling factors for various unit conversions: 1 inch = 2.54 cm */
#ifndef METRIC_CONVERSION_CONSTANT
    #define METRIC_CONVERSION_CONSTANT (1/25.4)
#endif

#ifndef mm2inches
    #define mm2inches (METRIC_CONVERSION_CONSTANT)
#endif

#ifndef inches2mm
    #define inches2mm (1/(mm2inches))
#endif

#ifndef mm2twips
    #define mm2twips (METRIC_CONVERSION_CONSTANT*1440)
#endif

#ifndef twips2mm
    #define twips2mm (1/(mm2twips))
#endif

#ifndef mm2pt
    #define mm2pt (METRIC_CONVERSION_CONSTANT*72)
#endif

#ifndef pt2mm
    #define pt2mm (1/(mm2pt))
#endif


#ifdef __cplusplus

#include <cmath>

/*
    Things are simple with C++11: we have everything we need in std.
    Eventually we will only have this section and not the legacy stuff below.
 */
#if __cplusplus >= 201103
    #define wxFinite(x) std::isfinite(x)
    #define wxIsNaN(x) std::isnan(x)
#else /* C++98 */

#if defined(__VISUALC__)
    #include <float.h>
    #define wxFinite(x) _finite(x)
#elif defined(__MINGW64_TOOLCHAIN__) || defined(__clang__)
    /*
        add more compilers with C99 support here: using C99 isfinite() is
        preferable to using BSD-ish finite()
     */
    #if defined(_GLIBCXX_CMATH) || defined(_LIBCPP_CMATH)
        // these <cmath> headers #undef isfinite
        #define wxFinite(x) std::isfinite(x)
    #else
        #define wxFinite(x) isfinite(x)
    #endif
#elif defined(wxNEEDS_STRICT_ANSI_WORKAROUNDS)
    wxDECL_FOR_STRICT_MINGW32(int, _finite, (double))

    #define wxFinite(x) _finite(x)
#elif ( defined(__GNUG__)||defined(__GNUWIN32__)|| \
      defined(__SGI_CC__)||defined(__SUNCC__)||defined(__XLC__)|| \
      defined(__HPUX__) ) && ( !defined(wxOSX_USE_IPHONE) || wxOSX_USE_IPHONE == 0 )
#ifdef __SOLARIS__
#include <ieeefp.h>
#endif
    #define wxFinite(x) finite(x)
#else
    #define wxFinite(x) ((x) == (x))
#endif


#if defined(__VISUALC__)
    #define wxIsNaN(x) _isnan(x)
#elif defined(__GNUG__)||defined(__GNUWIN32__)|| \
      defined(__SGI_CC__)||defined(__SUNCC__)||defined(__XLC__)|| \
      defined(__HPUX__)
    #define wxIsNaN(x) isnan(x)
#else
    #define wxIsNaN(x) ((x) != (x))
#endif

#endif /* C++11/C++98 */

#ifdef __INTELC__

    inline bool wxIsSameDouble(double x, double y)
    {
        // VZ: this warning, given for operators==() and !=() is not wrong, as ==
        //     shouldn't be used with doubles, but we get too many of them and
        //     removing these operators is probably not a good idea
        //
        //     Maybe we should always compare doubles up to some "epsilon" precision
        #pragma warning(push)

        // floating-point equality and inequality comparisons are unreliable
        #pragma warning(disable: 1572)

        return x == y;

        #pragma warning(pop)
    }

#else /* !__INTELC__ */
    wxGCC_WARNING_SUPPRESS(float-equal)
    inline bool wxIsSameDouble(double x, double y) { return x == y; }
    wxGCC_WARNING_RESTORE(float-equal)

#endif /* __INTELC__/!__INTELC__ */

inline bool wxIsNullDouble(double x) { return wxIsSameDouble(x, 0.); }

inline int wxRound(double x)
{
    wxASSERT_MSG(x > double(INT_MIN) - 0.5 && x < double(INT_MAX) + 0.5,
        "argument out of supported range");

    #if __cplusplus >= 201103
        return int(std::lround(x));
    #elif defined(HAVE_ROUND) || wxCHECK_VISUALC_VERSION(12)
        return int(lround(x));
    #else
        return int(x < 0 ? x - 0.5 : x + 0.5);
    #endif
}

inline int wxRound(float x)
{
    wxASSERT_MSG(x > float(INT_MIN) && x < float(INT_MAX),
        "argument out of supported range");

    #if __cplusplus >= 201103
        return int(std::lround(x));
    #elif defined(HAVE_ROUND) || wxCHECK_VISUALC_VERSION(12)
        return int(lroundf(x));
    #else
        return int(x < 0.0f ? x - 0.5f : x + 0.5f);
    #endif
}

inline int wxRound(long double x) { return wxRound(double(x)); }

// For compatibility purposes, make wxRound() work with integer types too, as
// this used to compile with wx 3.0.
#if WXWIN_COMPATIBILITY_3_0

template <typename T>
wxDEPRECATED_MSG("rounding an integer is useless")
inline int wxRound(T x)
{
    // We have to disable this warning for the unsigned types. We do handle
    // them correctly in this comparison due to "x > 0" below (removing it
    // would make this fail for them!).
    wxGCC_WARNING_SUPPRESS(sign-compare)

    wxASSERT_MSG((x > 0 || x > INT_MIN) && x < INT_MAX,
        "argument out of supported range");

    wxGCC_WARNING_RESTORE(sign-compare)

    return int(x);
}

#endif // WXWIN_COMPATIBILITY_3_0

// Convert between degrees and radians.
inline double wxDegToRad(double deg) { return (deg * M_PI) / 180.0; }
inline double wxRadToDeg(double rad) { return (rad * 180.0) / M_PI; }

// Count trailing zeros.
WXDLLIMPEXP_BASE unsigned int wxCTZ(wxUint32 x);

#endif /* __cplusplus */


#if defined(__WINDOWS__)
    #define wxMulDivInt32( a , b , c ) ::MulDiv( a , b , c )
#else
    #define wxMulDivInt32( a , b , c ) (wxRound((a)*(((wxDouble)b)/((wxDouble)c))))
#endif

#if wxUSE_APPLE_IEEE
#ifdef __cplusplus
    extern "C" {
#endif
    /* functions from common/extended.c */
    WXDLLIMPEXP_BASE wxFloat64 wxConvertFromIeeeExtended(const wxInt8 *bytes);
    WXDLLIMPEXP_BASE void wxConvertToIeeeExtended(wxFloat64 num, wxInt8 *bytes);

    /* use wxConvertFromIeeeExtended() and wxConvertToIeeeExtended() instead */
#if WXWIN_COMPATIBILITY_2_8
    wxDEPRECATED( WXDLLIMPEXP_BASE wxFloat64 ConvertFromIeeeExtended(const wxInt8 *bytes) );
    wxDEPRECATED( WXDLLIMPEXP_BASE void ConvertToIeeeExtended(wxFloat64 num, wxInt8 *bytes) );
#endif

#ifdef __cplusplus
    }
#endif
#endif /* wxUSE_APPLE_IEEE */

/* Compute the greatest common divisor of two positive integers */
WXDLLIMPEXP_BASE unsigned int wxGCD(unsigned int u, unsigned int v);

#endif /* _WX_MATH_H_ */
