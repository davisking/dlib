/*
 * Name:        wx/dlimpexp.h
 * Purpose:     Macros for declaring DLL-imported/exported functions
 * Author:      Vadim Zeitlin
 * Modified by:
 * Created:     16.10.2003 (extracted from wx/defs.h)
 * Copyright:   (c) 2003 Vadim Zeitlin <vadim@wxwidgets.org>
 * Licence:     wxWindows licence
 */

/*
    This is a C file, not C++ one, do not use C++ comments here!
 */

#ifndef _WX_DLIMPEXP_H_
#define _WX_DLIMPEXP_H_

#if defined(HAVE_VISIBILITY)
#    define WXEXPORT __attribute__ ((visibility("default")))
#    define WXIMPORT __attribute__ ((visibility("default")))
#elif defined(__WINDOWS__)
    /*
       __declspec works in as VC++.
     */
#    if defined(__VISUALC__)
#        define WXEXPORT __declspec(dllexport)
#        define WXIMPORT __declspec(dllimport)
    /*
        While gcc also supports __declspec(dllexport), it created unusably huge
        DLL files in gcc 4.[56] (while taking horribly long amounts of time),
        see http://gcc.gnu.org/bugzilla/show_bug.cgi?id=43601. Because of this
        we rely on binutils auto export/import support which seems to work
        quite well for 4.5+. However the problem was fixed in 4.7 and later and
        not exporting everything creates smaller DLLs (~8% size difference), so
        do use the explicit attributes again for the newer versions.
     */
#    elif defined(__GNUC__) && \
        (!wxCHECK_GCC_VERSION(4, 5) || wxCHECK_GCC_VERSION(4, 7))
        /*
            __declspec could be used here too but let's use the native
            __attribute__ instead for clarity.
        */
#       define WXEXPORT __attribute__((dllexport))
#       define WXIMPORT __attribute__((dllimport))
#    endif
#elif defined(__CYGWIN__)
#    define WXEXPORT __declspec(dllexport)
#    define WXIMPORT __declspec(dllimport)
#endif

/* for other platforms/compilers we don't anything */
#ifndef WXEXPORT
#    define WXEXPORT
#    define WXIMPORT
#endif

/*
   We support building wxWidgets as a set of several libraries but we don't
   support arbitrary combinations of libs/DLLs: either we build all of them as
   DLLs (in which case WXMAKINGDLL is defined) or none (it isn't).

   However we have a problem because we need separate WXDLLIMPEXP versions for
   different libraries as, for example, wxString class should be dllexported
   when compiled in wxBase and dllimported otherwise, so we do define separate
   WXMAKING/USINGDLL_XYZ constants for each component XYZ.
 */
#ifdef WXMAKINGDLL
#    if wxUSE_BASE
#        define WXMAKINGDLL_BASE
#    endif

#    define WXMAKINGDLL_NET
#    define WXMAKINGDLL_CORE
#    define WXMAKINGDLL_ADV
#    define WXMAKINGDLL_QA
#    define WXMAKINGDLL_HTML
#    define WXMAKINGDLL_GL
#    define WXMAKINGDLL_XML
#    define WXMAKINGDLL_XRC
#    define WXMAKINGDLL_AUI
#    define WXMAKINGDLL_PROPGRID
#    define WXMAKINGDLL_RIBBON
#    define WXMAKINGDLL_RICHTEXT
#    define WXMAKINGDLL_MEDIA
#    define WXMAKINGDLL_STC
#    define WXMAKINGDLL_WEBVIEW
#endif /* WXMAKINGDLL */

/*
   WXDLLIMPEXP_CORE maps to export declaration when building the DLL, to import
   declaration if using it or to nothing at all if we don't use wxWin as DLL
 */
#ifdef WXMAKINGDLL_BASE
#    define WXDLLIMPEXP_BASE WXEXPORT
#    define WXDLLIMPEXP_DATA_BASE(type) WXEXPORT type
#    if defined(HAVE_VISIBILITY)
#        define WXDLLIMPEXP_INLINE_BASE WXEXPORT
#    else
#        define WXDLLIMPEXP_INLINE_BASE
#    endif
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_BASE WXIMPORT
#    define WXDLLIMPEXP_DATA_BASE(type) WXIMPORT type
#    if defined(HAVE_VISIBILITY)
#        define WXDLLIMPEXP_INLINE_BASE WXIMPORT
#    else
#        define WXDLLIMPEXP_INLINE_BASE
#    endif
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_BASE
#    define WXDLLIMPEXP_DATA_BASE(type) type
#    define WXDLLIMPEXP_INLINE_BASE
#endif

#ifdef WXMAKINGDLL_NET
#    define WXDLLIMPEXP_NET WXEXPORT
#    define WXDLLIMPEXP_DATA_NET(type) WXEXPORT type
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_NET WXIMPORT
#    define WXDLLIMPEXP_DATA_NET(type) WXIMPORT type
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_NET
#    define WXDLLIMPEXP_DATA_NET(type) type
#endif

#ifdef WXMAKINGDLL_CORE
#    define WXDLLIMPEXP_CORE WXEXPORT
#    define WXDLLIMPEXP_DATA_CORE(type) WXEXPORT type
#    if defined(HAVE_VISIBILITY)
#        define WXDLLIMPEXP_INLINE_CORE WXEXPORT
#    else
#        define WXDLLIMPEXP_INLINE_CORE
#    endif
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_CORE WXIMPORT
#    define WXDLLIMPEXP_DATA_CORE(type) WXIMPORT type
#    if defined(HAVE_VISIBILITY)
#        define WXDLLIMPEXP_INLINE_CORE WXIMPORT
#    else
#        define WXDLLIMPEXP_INLINE_CORE
#    endif
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_CORE
#    define WXDLLIMPEXP_DATA_CORE(type) type
#    define WXDLLIMPEXP_INLINE_CORE
#endif

/* Advanced library doesn't exist any longer, but its macros are preserved for
   compatibility. Do not use them in the new code. */
#define WXDLLIMPEXP_ADV WXDLLIMPEXP_CORE
#define WXDLLIMPEXP_DATA_ADV(type) WXDLLIMPEXP_DATA_CORE(type)

#ifdef WXMAKINGDLL_QA
#    define WXDLLIMPEXP_QA WXEXPORT
#    define WXDLLIMPEXP_DATA_QA(type) WXEXPORT type
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_QA WXIMPORT
#    define WXDLLIMPEXP_DATA_QA(type) WXIMPORT type
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_QA
#    define WXDLLIMPEXP_DATA_QA(type) type
#endif

#ifdef WXMAKINGDLL_HTML
#    define WXDLLIMPEXP_HTML WXEXPORT
#    define WXDLLIMPEXP_DATA_HTML(type) WXEXPORT type
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_HTML WXIMPORT
#    define WXDLLIMPEXP_DATA_HTML(type) WXIMPORT type
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_HTML
#    define WXDLLIMPEXP_DATA_HTML(type) type
#endif

#ifdef WXMAKINGDLL_GL
#    define WXDLLIMPEXP_GL WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_GL WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_GL
#endif

#ifdef WXMAKINGDLL_XML
#    define WXDLLIMPEXP_XML WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_XML WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_XML
#endif

#ifdef WXMAKINGDLL_XRC
#    define WXDLLIMPEXP_XRC WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_XRC WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_XRC
#endif

#ifdef WXMAKINGDLL_AUI
#    define WXDLLIMPEXP_AUI WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_AUI WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_AUI
#endif

#ifdef WXMAKINGDLL_PROPGRID
#    define WXDLLIMPEXP_PROPGRID WXEXPORT
#    define WXDLLIMPEXP_DATA_PROPGRID(type) WXEXPORT type
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_PROPGRID WXIMPORT
#    define WXDLLIMPEXP_DATA_PROPGRID(type) WXIMPORT type
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_PROPGRID
#    define WXDLLIMPEXP_DATA_PROPGRID(type) type
#endif

#ifdef WXMAKINGDLL_RIBBON
#    define WXDLLIMPEXP_RIBBON WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_RIBBON WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_RIBBON
#endif

#ifdef WXMAKINGDLL_RICHTEXT
#    define WXDLLIMPEXP_RICHTEXT WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_RICHTEXT WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_RICHTEXT
#endif

#ifdef WXMAKINGDLL_MEDIA
#    define WXDLLIMPEXP_MEDIA WXEXPORT
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_MEDIA WXIMPORT
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_MEDIA
#endif

#ifdef WXMAKINGDLL_STC
#    define WXDLLIMPEXP_STC WXEXPORT
#    define WXDLLIMPEXP_DATA_STC(type) WXEXPORT type
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_STC WXIMPORT
#    define WXDLLIMPEXP_DATA_STC(type) WXIMPORT type
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_STC
#    define WXDLLIMPEXP_DATA_STC(type) type
#endif

#ifdef WXMAKINGDLL_WEBVIEW
#    define WXDLLIMPEXP_WEBVIEW WXEXPORT
#    define WXDLLIMPEXP_DATA_WEBVIEW(type) WXEXPORT type
#elif defined(WXUSINGDLL)
#    define WXDLLIMPEXP_WEBVIEW WXIMPORT
#    define WXDLLIMPEXP_DATA_WEBVIEW(type) WXIMPORT type
#else /* not making nor using DLL */
#    define WXDLLIMPEXP_WEBVIEW
#    define WXDLLIMPEXP_DATA_WEBVIEW(type) type
#endif

/*
   GCC warns about using __attribute__ (and also __declspec in mingw32 case) on
   forward declarations while MSVC complains about forward declarations without
   __declspec for the classes later declared with it, so we need a separate set
   of macros for forward declarations to hide this difference:
 */
#if defined(HAVE_VISIBILITY) || (defined(__WINDOWS__) && defined(__GNUC__))
    #define WXDLLIMPEXP_FWD_BASE
    #define WXDLLIMPEXP_FWD_NET
    #define WXDLLIMPEXP_FWD_CORE
    #define WXDLLIMPEXP_FWD_QA
    #define WXDLLIMPEXP_FWD_HTML
    #define WXDLLIMPEXP_FWD_GL
    #define WXDLLIMPEXP_FWD_XML
    #define WXDLLIMPEXP_FWD_XRC
    #define WXDLLIMPEXP_FWD_AUI
    #define WXDLLIMPEXP_FWD_PROPGRID
    #define WXDLLIMPEXP_FWD_RIBBON
    #define WXDLLIMPEXP_FWD_RICHTEXT
    #define WXDLLIMPEXP_FWD_MEDIA
    #define WXDLLIMPEXP_FWD_STC
    #define WXDLLIMPEXP_FWD_WEBVIEW
#else
    #define WXDLLIMPEXP_FWD_BASE      WXDLLIMPEXP_BASE
    #define WXDLLIMPEXP_FWD_NET       WXDLLIMPEXP_NET
    #define WXDLLIMPEXP_FWD_CORE      WXDLLIMPEXP_CORE
    #define WXDLLIMPEXP_FWD_QA        WXDLLIMPEXP_QA
    #define WXDLLIMPEXP_FWD_HTML      WXDLLIMPEXP_HTML
    #define WXDLLIMPEXP_FWD_GL        WXDLLIMPEXP_GL
    #define WXDLLIMPEXP_FWD_XML       WXDLLIMPEXP_XML
    #define WXDLLIMPEXP_FWD_XRC       WXDLLIMPEXP_XRC
    #define WXDLLIMPEXP_FWD_AUI       WXDLLIMPEXP_AUI
    #define WXDLLIMPEXP_FWD_PROPGRID  WXDLLIMPEXP_PROPGRID
    #define WXDLLIMPEXP_FWD_RIBBON    WXDLLIMPEXP_RIBBON
    #define WXDLLIMPEXP_FWD_RICHTEXT  WXDLLIMPEXP_RICHTEXT
    #define WXDLLIMPEXP_FWD_MEDIA     WXDLLIMPEXP_MEDIA
    #define WXDLLIMPEXP_FWD_STC       WXDLLIMPEXP_STC
    #define WXDLLIMPEXP_FWD_WEBVIEW   WXDLLIMPEXP_WEBVIEW
#endif

/* This macro continues to exist for backwards compatibility only. */
#define WXDLLIMPEXP_FWD_ADV       WXDLLIMPEXP_FWD_CORE

/* for backwards compatibility, define suffix-less versions too */
#define WXDLLEXPORT WXDLLIMPEXP_CORE
#define WXDLLEXPORT_DATA WXDLLIMPEXP_DATA_CORE

#endif /* _WX_DLIMPEXP_H_ */
