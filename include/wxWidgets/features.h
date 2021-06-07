/**
*  Name:        wx/features.h
*  Purpose:     test macros for the features which might be available in some
*               wxWidgets ports but not others
*  Author:      Vadim Zeitlin
*  Modified by: Ryan Norton (Converted to C)
*  Created:     18.03.02
*  Copyright:   (c) 2002 Vadim Zeitlin <vadim@wxwidgets.org>
*  Licence:     wxWindows licence
*/

/* THIS IS A C FILE, DON'T USE C++ FEATURES (IN PARTICULAR COMMENTS) IN IT */

#ifndef _WX_FEATURES_H_
#define _WX_FEATURES_H_

/*  radio menu items are currently not implemented in wxMotif, use this
    symbol (kept for compatibility from the time when they were not implemented
    under other platforms as well) to test for this */
#if !defined(__WXMOTIF__)
    #define wxHAS_RADIO_MENU_ITEMS
#else
    #undef wxHAS_RADIO_MENU_ITEMS
#endif

/*  the raw keyboard codes are generated under wxGTK and wxMSW only */
#if defined(__WXGTK__) || defined(__WXMSW__) || defined(__WXMAC__) \
    || defined(__WXDFB__)
    #define wxHAS_RAW_KEY_CODES
#else
    #undef wxHAS_RAW_KEY_CODES
#endif

/*  taskbar is implemented in the major ports */
#if defined(__WXMSW__) \
    || defined(__WXGTK__) || defined(__WXMOTIF__) || defined(__WXX11__) \
    || defined(__WXOSX_MAC__) || defined(__WXQT__)
    #define wxHAS_TASK_BAR_ICON
#else
    #undef wxUSE_TASKBARICON
    #define wxUSE_TASKBARICON 0
    #undef wxHAS_TASK_BAR_ICON
#endif

/*  wxIconLocation appeared in the middle of 2.5.0 so it's handy to have a */
/*  separate define for it */
#define wxHAS_ICON_LOCATION

/*  same for wxCrashReport */
#ifdef __WXMSW__
    #define wxHAS_CRASH_REPORT
#else
    #undef wxHAS_CRASH_REPORT
#endif

/*  wxRE_ADVANCED is not always available, depending on regex library used
 *  (it's unavailable only if compiling via configure against system library) */
#ifndef WX_NO_REGEX_ADVANCED
    #define wxHAS_REGEX_ADVANCED
#else
    #undef wxHAS_REGEX_ADVANCED
#endif

/* Pango-based ports and wxDFB use UTF-8 for text and font encodings
 * internally and so their fonts can handle any encodings: */
#if wxUSE_PANGO || defined(__WXDFB__)
    #define wxHAS_UTF8_FONTS
#endif

/* This is defined when the underlying toolkit handles tab traversal natively.
   Otherwise we implement it ourselves in wxControlContainer. */
#if defined(__WXGTK20__) || defined(__WXQT__)
    #define wxHAS_NATIVE_TAB_TRAVERSAL
#endif

/* This is defined when the compiler provides some type of extended locale
   functions.  Otherwise, we implement them ourselves to only support the
   'C' locale */
#if defined(HAVE_LOCALE_T) || \
    (wxCHECK_VISUALC_VERSION(8))
    #define wxHAS_XLOCALE_SUPPORT
#else
    #undef wxHAS_XLOCALE_SUPPORT
#endif

/* Direct access to bitmap data is not implemented in all ports yet */
#if defined(__WXGTK20__) || defined(__WXMAC__) || defined(__WXDFB__) || \
        defined(__WXMSW__) || defined(__WXQT__)

    /*
       HP aCC for PA-RISC can't deal with templates in wx/rawbmp.h.
     */
    #if !(defined(__HP_aCC) && defined(__hppa))
        #define wxHAS_RAW_BITMAP
    #endif
#endif

/* also define deprecated synonym which exists for compatibility only */
#ifdef wxHAS_RAW_BITMAP
    #define wxHAVE_RAW_BITMAP
#endif


// Previously this symbol wasn't defined for all compilers as Bind() couldn't
// be implemented for some of them (notably MSVC 6), but this is not the case
// any more and Bind() is always implemented when using any currently supported
// compiler, so this symbol exists purely for compatibility.
#define wxHAS_EVENT_BIND

#endif /*  _WX_FEATURES_H_ */

