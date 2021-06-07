/*
 * Name:        wx/wxcrtbase.h
 * Purpose:     Type-safe ANSI and Unicode builds compatible wrappers for
 *              CRT functions
 * Author:      Joel Farley, Ove Kaaven
 * Modified by: Vadim Zeitlin, Robert Roebling, Ron Lee
 * Created:     1998/06/12
 * Copyright:   (c) 1998-2006 wxWidgets dev team
 * Licence:     wxWindows licence
 */

/* THIS IS A C FILE, DON'T USE C++ FEATURES (IN PARTICULAR COMMENTS) IN IT */

#ifndef _WX_WXCRTBASE_H_
#define _WX_WXCRTBASE_H_

/* -------------------------------------------------------------------------
                        headers and missing declarations
   ------------------------------------------------------------------------- */

#include "wx/chartype.h"

/*
    Standard headers we need here.

    NB: don't include any wxWidgets headers here because almost all of them
        include this one!

    NB2: User code should include wx/crt.h instead of including this
         header directly.

 */

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <wctype.h>
#include <time.h>

#if defined(__WINDOWS__)
    #include <io.h>
#endif

#if defined(HAVE_STRTOK_R) && defined(__DARWIN__) && defined(_MSL_USING_MW_C_HEADERS) && _MSL_USING_MW_C_HEADERS
    char *strtok_r(char *, const char *, char **);
#endif

/*
   Traditional MinGW doesn't declare isascii() in strict ANSI mode and we can't
   declare it here ourselves as it's an inline function, so use our own
   replacement instead.
 */
#ifndef isascii
    #if defined(wxNEEDS_STRICT_ANSI_WORKAROUNDS)
        #define wxNEED_ISASCII
    #endif
#endif /* isascii */

#ifdef wxNEED_ISASCII
    inline int isascii(int c) { return (unsigned)c < 0x80; }

    // Avoid further (re)definitions of it.
    #define isascii isascii
#endif

/* string.h functions */

#ifdef wxNEED_STRDUP
    WXDLLIMPEXP_BASE char *strdup(const char* s);
#endif


/* -------------------------------------------------------------------------
                            UTF-8 locale handling
   ------------------------------------------------------------------------- */

#ifdef __cplusplus
        /* flag indicating whether the current locale uses UTF-8 or not; must be
           updated every time the locale is changed! */
        #if wxUSE_UTF8_LOCALE_ONLY
        #define wxLocaleIsUtf8 true
        #else
        extern WXDLLIMPEXP_BASE bool wxLocaleIsUtf8;
        #endif
        /* function used to update the flag: */
        extern WXDLLIMPEXP_BASE void wxUpdateLocaleIsUtf8();
#endif /* __cplusplus */


/* -------------------------------------------------------------------------
                                 string.h
   ------------------------------------------------------------------------- */

#define wxCRT_StrcatA    strcat
#define wxCRT_StrchrA    strchr
#define wxCRT_StrcmpA    strcmp
#define wxCRT_StrcpyA    strcpy
#define wxCRT_StrcspnA   strcspn
#define wxCRT_StrlenA    strlen
#define wxCRT_StrncatA   strncat
#define wxCRT_StrncmpA   strncmp
#define wxCRT_StrncpyA   strncpy
#define wxCRT_StrpbrkA   strpbrk
#define wxCRT_StrrchrA   strrchr
#define wxCRT_StrspnA    strspn
#define wxCRT_StrstrA    strstr

#define wxCRT_StrcatW    wcscat
#define wxCRT_StrchrW    wcschr
#define wxCRT_StrcmpW    wcscmp
#define wxCRT_StrcpyW    wcscpy
#define wxCRT_StrcspnW   wcscspn
#define wxCRT_StrncatW   wcsncat
#define wxCRT_StrncmpW   wcsncmp
#define wxCRT_StrncpyW   wcsncpy
#define wxCRT_StrpbrkW   wcspbrk
#define wxCRT_StrrchrW   wcsrchr
#define wxCRT_StrspnW    wcsspn
#define wxCRT_StrstrW    wcsstr

#define wxCRT_StrcollA   strcoll
#define wxCRT_StrxfrmA   strxfrm

#define wxCRT_StrcollW   wcscoll
#define wxCRT_StrxfrmW   wcsxfrm

/* Almost all compilers have strdup(), but VC++ and MinGW call it _strdup().
   And we need to declare it manually for MinGW in strict ANSI mode. */
#if (defined(__VISUALC__) && __VISUALC__ >= 1400)
    #define wxCRT_StrdupA _strdup
#elif defined(__MINGW32__)
    wxDECL_FOR_STRICT_MINGW32(char*, _strdup, (const char *))
    #define wxCRT_StrdupA _strdup
#else
    #define wxCRT_StrdupA strdup
#endif

/* Windows compilers provide _wcsdup() except for (old) Cygwin */
#if defined(__WINDOWS__) && !defined(__CYGWIN__)
    wxDECL_FOR_STRICT_MINGW32(wchar_t*, _wcsdup, (const wchar_t*))
    #define wxCRT_StrdupW _wcsdup
#elif defined(HAVE_WCSDUP)
    #define wxCRT_StrdupW wcsdup
#endif

#ifdef wxHAVE_TCHAR_SUPPORT
    /* we surely have wchar_t if we have TCHAR have wcslen() */
    #ifndef HAVE_WCSLEN
        #define HAVE_WCSLEN
    #endif
#endif /* wxHAVE_TCHAR_SUPPORT */

#ifdef HAVE_WCSLEN
    #define wxCRT_StrlenW wcslen
#endif

#define wxCRT_StrtodA    strtod
#define wxCRT_StrtolA    strtol
#define wxCRT_StrtoulA   strtoul

#ifdef __ANDROID__ // these functions are broken on android

extern double android_wcstod(const wchar_t *nptr, wchar_t **endptr);
extern long android_wcstol(const wchar_t *nptr, wchar_t **endptr, int base);
extern unsigned long android_wcstoul(const wchar_t *nptr, wchar_t **endptr, int base);

#define wxCRT_StrtodW    android_wcstod
#define wxCRT_StrtolW    android_wcstol
#define wxCRT_StrtoulW   android_wcstoul
#else
#define wxCRT_StrtodW    wcstod
#define wxCRT_StrtolW    wcstol
#define wxCRT_StrtoulW   wcstoul
#endif

#ifdef __VISUALC__
    #define wxCRT_StrtollA   _strtoi64
    #define wxCRT_StrtoullA  _strtoui64
    #define wxCRT_StrtollW   _wcstoi64
    #define wxCRT_StrtoullW  _wcstoui64
#else
    /* Both of these functions are implemented in C++11 compilers */
    #if defined(__cplusplus) && __cplusplus >= 201103L
        #ifndef HAVE_STRTOULL
            #define HAVE_STRTOULL
        #endif
        #ifndef HAVE_WCSTOULL
            #define HAVE_WCSTOULL
        #endif
    #endif

    #ifdef HAVE_STRTOULL
        wxDECL_FOR_STRICT_MINGW32(long long, strtoll, (const char*, char**, int))
        wxDECL_FOR_STRICT_MINGW32(unsigned long long, strtoull, (const char*, char**, int))

        #define wxCRT_StrtollA   strtoll
        #define wxCRT_StrtoullA  strtoull
    #endif /* HAVE_STRTOULL */
    #ifdef HAVE_WCSTOULL
        /* assume that we have wcstoull(), which is also C99, too */
        #define wxCRT_StrtollW   wcstoll
        #define wxCRT_StrtoullW  wcstoull
    #endif /* HAVE_WCSTOULL */
#endif

/*
    Only VC8 and later provide strnlen() and wcsnlen() functions under Windows.
 */
#if wxCHECK_VISUALC_VERSION(8)
    #ifndef HAVE_STRNLEN
        #define HAVE_STRNLEN
    #endif
    #ifndef HAVE_WCSNLEN
        #define HAVE_WCSNLEN
    #endif
#endif

#ifdef HAVE_STRNLEN
    #define wxCRT_StrnlenA  strnlen
#endif

#ifdef HAVE_WCSNLEN
    /*
        When using MinGW, wcsnlen() is not declared, but is still found by
        configure -- just declare it in this case as it seems better to use it
        if it's available (see https://sourceforge.net/p/mingw/bugs/2332/)
     */
    wxDECL_FOR_MINGW32_ALWAYS(size_t, wcsnlen, (const wchar_t*, size_t))

    #define wxCRT_StrnlenW  wcsnlen
#endif

/* define wxCRT_StricmpA/W and wxCRT_StrnicmpA/W for various compilers */
#if defined(__VISUALC__) || defined(__MINGW32__)
    /*
        Due to MinGW 5.3 bug (https://sourceforge.net/p/mingw/bugs/2322/),
        _stricmp() and _strnicmp() are not declared in its standard headers
        when compiling without optimizations. Work around this by always
        declaring them ourselves (notice that if/when this bug were fixed, we'd
        still need to use wxDECL_FOR_STRICT_MINGW32() for them here.
     */
    wxDECL_FOR_MINGW32_ALWAYS(int, _stricmp, (const char*, const char*))
    wxDECL_FOR_MINGW32_ALWAYS(int, _strnicmp, (const char*, const char*, size_t))

    #define wxCRT_StricmpA _stricmp
    #define wxCRT_StrnicmpA _strnicmp
#elif defined(__UNIX__)
    #define wxCRT_StricmpA strcasecmp
    #define wxCRT_StrnicmpA strncasecmp
/* #else -- use wxWidgets implementation */
#endif

#ifdef __VISUALC__
    #define wxCRT_StricmpW _wcsicmp
    #define wxCRT_StrnicmpW _wcsnicmp
#elif defined(__UNIX__)
    #ifdef HAVE_WCSCASECMP
        #define wxCRT_StricmpW wcscasecmp
    #endif
    #ifdef HAVE_WCSNCASECMP
        #define wxCRT_StrnicmpW wcsncasecmp
    #endif
/* #else -- use wxWidgets implementation */
#endif

#ifdef HAVE_STRTOK_R
    #define  wxCRT_StrtokA(str, sep, last)    strtok_r(str, sep, last)
#endif
/* FIXME-UTF8: detect and use wcstok() if available for wxCRT_StrtokW */

/* these are extern "C" because they are used by regex lib: */
#ifdef __cplusplus
extern "C" {
#endif

#ifndef wxCRT_StrlenW
WXDLLIMPEXP_BASE size_t wxCRT_StrlenW(const wchar_t *s);
#endif

#ifndef wxCRT_StrncmpW
WXDLLIMPEXP_BASE int wxCRT_StrncmpW(const wchar_t *s1, const wchar_t *s2, size_t n);
#endif

#ifdef __cplusplus
}
#endif

/* FIXME-UTF8: remove this once we are Unicode only */
#if wxUSE_UNICODE
    #define wxCRT_StrlenNative  wxCRT_StrlenW
    #define wxCRT_StrncmpNative wxCRT_StrncmpW
    #define wxCRT_ToupperNative wxCRT_ToupperW
    #define wxCRT_TolowerNative wxCRT_TolowerW
#else
    #define wxCRT_StrlenNative  wxCRT_StrlenA
    #define wxCRT_StrncmpNative wxCRT_StrncmpA
    #define wxCRT_ToupperNative toupper
    #define wxCRT_TolowerNative tolower
#endif

#ifndef wxCRT_StrcatW
WXDLLIMPEXP_BASE wchar_t *wxCRT_StrcatW(wchar_t *dest, const wchar_t *src);
#endif

#ifndef wxCRT_StrchrW
WXDLLIMPEXP_BASE const wchar_t *wxCRT_StrchrW(const wchar_t *s, wchar_t c);
#endif

#ifndef wxCRT_StrcmpW
WXDLLIMPEXP_BASE int wxCRT_StrcmpW(const wchar_t *s1, const wchar_t *s2);
#endif

#ifndef wxCRT_StrcollW
WXDLLIMPEXP_BASE int wxCRT_StrcollW(const wchar_t *s1, const wchar_t *s2);
#endif

#ifndef wxCRT_StrcpyW
WXDLLIMPEXP_BASE wchar_t *wxCRT_StrcpyW(wchar_t *dest, const wchar_t *src);
#endif

#ifndef wxCRT_StrcspnW
WXDLLIMPEXP_BASE size_t wxCRT_StrcspnW(const wchar_t *s, const wchar_t *reject);
#endif

#ifndef wxCRT_StrncatW
WXDLLIMPEXP_BASE wchar_t *wxCRT_StrncatW(wchar_t *dest, const wchar_t *src, size_t n);
#endif

#ifndef wxCRT_StrncpyW
WXDLLIMPEXP_BASE wchar_t *wxCRT_StrncpyW(wchar_t *dest, const wchar_t *src, size_t n);
#endif

#ifndef wxCRT_StrpbrkW
WXDLLIMPEXP_BASE const wchar_t *wxCRT_StrpbrkW(const wchar_t *s, const wchar_t *accept);
#endif

#ifndef wxCRT_StrrchrW
WXDLLIMPEXP_BASE const wchar_t *wxCRT_StrrchrW(const wchar_t *s, wchar_t c);
#endif

#ifndef wxCRT_StrspnW
WXDLLIMPEXP_BASE size_t wxCRT_StrspnW(const wchar_t *s, const wchar_t *accept);
#endif

#ifndef wxCRT_StrstrW
WXDLLIMPEXP_BASE const wchar_t *wxCRT_StrstrW(const wchar_t *haystack, const wchar_t *needle);
#endif

#ifndef wxCRT_StrtodW
WXDLLIMPEXP_BASE double wxCRT_StrtodW(const wchar_t *nptr, wchar_t **endptr);
#endif

#ifndef wxCRT_StrtolW
WXDLLIMPEXP_BASE long int wxCRT_StrtolW(const wchar_t *nptr, wchar_t **endptr, int base);
#endif

#ifndef wxCRT_StrtoulW
WXDLLIMPEXP_BASE unsigned long int wxCRT_StrtoulW(const wchar_t *nptr, wchar_t **endptr, int base);
#endif

#ifndef wxCRT_StrxfrmW
WXDLLIMPEXP_BASE size_t wxCRT_StrxfrmW(wchar_t *dest, const wchar_t *src, size_t n);
#endif

#ifndef wxCRT_StrdupA
WXDLLIMPEXP_BASE char *wxCRT_StrdupA(const char *psz);
#endif

#ifndef wxCRT_StrdupW
WXDLLIMPEXP_BASE wchar_t *wxCRT_StrdupW(const wchar_t *pwz);
#endif

#ifndef wxCRT_StricmpA
WXDLLIMPEXP_BASE int wxCRT_StricmpA(const char *psz1, const char *psz2);
#endif

#ifndef wxCRT_StricmpW
WXDLLIMPEXP_BASE int wxCRT_StricmpW(const wchar_t *psz1, const wchar_t *psz2);
#endif

#ifndef wxCRT_StrnicmpA
WXDLLIMPEXP_BASE int wxCRT_StrnicmpA(const char *psz1, const char *psz2, size_t len);
#endif

#ifndef wxCRT_StrnicmpW
WXDLLIMPEXP_BASE int wxCRT_StrnicmpW(const wchar_t *psz1, const wchar_t *psz2, size_t len);
#endif

#ifndef wxCRT_StrtokA
WXDLLIMPEXP_BASE char *wxCRT_StrtokA(char *psz, const char *delim, char **save_ptr);
#endif

#ifndef wxCRT_StrtokW
WXDLLIMPEXP_BASE wchar_t *wxCRT_StrtokW(wchar_t *psz, const wchar_t *delim, wchar_t **save_ptr);
#endif

/* supply strtoll and strtoull, if needed */
#ifdef wxLongLong_t
    #ifndef wxCRT_StrtollA
        WXDLLIMPEXP_BASE wxLongLong_t wxCRT_StrtollA(const char* nptr,
                                                     char** endptr,
                                                     int base);
        WXDLLIMPEXP_BASE wxULongLong_t wxCRT_StrtoullA(const char* nptr,
                                                       char** endptr,
                                                       int base);
    #endif
    #ifndef wxCRT_StrtollW
        WXDLLIMPEXP_BASE wxLongLong_t wxCRT_StrtollW(const wchar_t* nptr,
                                                     wchar_t** endptr,
                                                     int base);
        WXDLLIMPEXP_BASE wxULongLong_t wxCRT_StrtoullW(const wchar_t* nptr,
                                                       wchar_t** endptr,
                                                       int base);
    #endif
#endif /* wxLongLong_t */


/* -------------------------------------------------------------------------
                                  stdio.h
   ------------------------------------------------------------------------- */

#if defined(__UNIX__) || defined(__WXMAC__)
    #define wxMBFILES 1
#else
    #define wxMBFILES 0
#endif


/* these functions are only needed in the form used for filenames (i.e. char*
   on Unix, wchar_t* on Windows), so we don't need to use A/W suffix: */
#if wxMBFILES || !wxUSE_UNICODE /* ANSI filenames */

    #define wxCRT_Fopen   fopen
    #define wxCRT_Freopen freopen
    #define wxCRT_Remove  remove
    #define wxCRT_Rename  rename

#else /* Unicode filenames */
    wxDECL_FOR_STRICT_MINGW32(FILE*, _wfopen, (const wchar_t*, const wchar_t*))
    wxDECL_FOR_STRICT_MINGW32(FILE*, _wfreopen, (const wchar_t*, const wchar_t*, FILE*))
    wxDECL_FOR_STRICT_MINGW32(int, _wrename, (const wchar_t*, const wchar_t*))
    wxDECL_FOR_STRICT_MINGW32(int, _wremove, (const wchar_t*))

    #define wxCRT_Rename   _wrename
    #define wxCRT_Remove _wremove
    #define wxCRT_Fopen    _wfopen
    #define wxCRT_Freopen  _wfreopen

#endif /* wxMBFILES/!wxMBFILES */

#define wxCRT_PutsA       puts
#define wxCRT_FputsA      fputs
#define wxCRT_FgetsA      fgets
#define wxCRT_FputcA      fputc
#define wxCRT_FgetcA      fgetc
#define wxCRT_UngetcA     ungetc

#ifdef wxHAVE_TCHAR_SUPPORT
    #define wxCRT_PutsW   _putws
    #define wxCRT_FputsW  fputws
    #define wxCRT_FputcW  fputwc
#endif
#ifdef HAVE_FPUTWS
    #define wxCRT_FputsW  fputws
#endif
#ifdef HAVE_PUTWS
    #define wxCRT_PutsW   putws
#endif
#ifdef HAVE_FPUTWC
    #define wxCRT_FputcW  fputwc
#endif
#define wxCRT_FgetsW  fgetws

#ifndef wxCRT_PutsW
WXDLLIMPEXP_BASE int wxCRT_PutsW(const wchar_t *ws);
#endif

#ifndef wxCRT_FputsW
WXDLLIMPEXP_BASE int wxCRT_FputsW(const wchar_t *ch, FILE *stream);
#endif

#ifndef wxCRT_FputcW
WXDLLIMPEXP_BASE int wxCRT_FputcW(wchar_t wc, FILE *stream);
#endif

/*
   NB: tmpnam() is unsafe and thus is not wrapped!
       Use other wxWidgets facilities instead:
        wxFileName::CreateTempFileName, wxTempFile, or wxTempFileOutputStream
*/
#define wxTmpnam(x)         wxTmpnam_is_insecure_use_wxTempFile_instead

#define wxCRT_PerrorA   perror
#ifdef wxHAVE_TCHAR_SUPPORT
    #define wxCRT_PerrorW _wperror
#endif

/* -------------------------------------------------------------------------
                                  stdlib.h
   ------------------------------------------------------------------------- */

#define wxCRT_GetenvA           getenv
#ifdef wxHAVE_TCHAR_SUPPORT
    #define wxCRT_GetenvW       _wgetenv
#endif

#ifndef wxCRT_GetenvW
WXDLLIMPEXP_BASE wchar_t * wxCRT_GetenvW(const wchar_t *name);
#endif


#define wxCRT_SystemA               system
#ifdef wxHAVE_TCHAR_SUPPORT
    #define  wxCRT_SystemW          _wsystem
#endif

#define wxCRT_AtofA                 atof
#define wxCRT_AtoiA                 atoi
#define wxCRT_AtolA                 atol

#if defined(wxHAVE_TCHAR_SUPPORT)
    wxDECL_FOR_STRICT_MINGW32(int, _wtoi, (const wchar_t*))
    wxDECL_FOR_STRICT_MINGW32(long, _wtol, (const wchar_t*))

    #define  wxCRT_AtoiW           _wtoi
    #define  wxCRT_AtolW           _wtol
    /* _wtof doesn't exist */
#else
#ifndef __VMS
    #define wxCRT_AtofW(s)         wcstod(s, NULL)
#endif
    #define wxCRT_AtolW(s)         wcstol(s, NULL, 10)
    /* wcstoi doesn't exist */
#endif

/* -------------------------------------------------------------------------
                                time.h
   ------------------------------------------------------------------------- */

#define wxCRT_StrftimeA  strftime
#ifdef __SGI__
    /*
        IRIX provides not one but two versions of wcsftime(): XPG4 one which
        uses "const char*" for the third parameter and so can't be used and the
        correct, XPG5, one. Unfortunately we can't just define _XOPEN_SOURCE
        high enough to get XPG5 version as this undefines other symbols which
        make other functions we use unavailable (see <standards.h> for gory
        details). So just declare the XPG5 version ourselves, we're extremely
        unlikely to ever be compiled on a system without it. But if we ever do,
        a configure test would need to be added for it (and _MIPS_SYMBOL_PRESENT
        should be used to check for its presence during run-time, i.e. it would
        probably be simpler to just always use our own wxCRT_StrftimeW() below
        if it does ever become a problem).
     */
#ifdef __cplusplus
    extern "C"
#endif
    size_t
    _xpg5_wcsftime(wchar_t *, size_t, const wchar_t *, const struct tm * );
    #define wxCRT_StrftimeW _xpg5_wcsftime
#else
    /*
        Assume it's always available under non-Unix systems as this does seem
        to be the case for now. And under Unix we trust configure to detect it
        (except for SGI special case above).
     */
    #if defined(HAVE_WCSFTIME) || !defined(__UNIX__)
        #define wxCRT_StrftimeW  wcsftime
    #endif
#endif

#ifndef wxCRT_StrftimeW
WXDLLIMPEXP_BASE size_t wxCRT_StrftimeW(wchar_t *s, size_t max,
                                        const wchar_t *fmt,
                                        const struct tm *tm);
#endif



/* -------------------------------------------------------------------------
                                ctype.h
   ------------------------------------------------------------------------- */

#define wxCRT_IsalnumW(c)   iswalnum(c)
#define wxCRT_IsalphaW(c)   iswalpha(c)
#define wxCRT_IscntrlW(c)   iswcntrl(c)
#define wxCRT_IsdigitW(c)   iswdigit(c)
#define wxCRT_IsgraphW(c)   iswgraph(c)
#define wxCRT_IslowerW(c)   iswlower(c)
#define wxCRT_IsprintW(c)   iswprint(c)
#define wxCRT_IspunctW(c)   iswpunct(c)
#define wxCRT_IsspaceW(c)   iswspace(c)
#define wxCRT_IsupperW(c)   iswupper(c)
#define wxCRT_IsxdigitW(c)  iswxdigit(c)

#ifdef __GLIBC__
    #if defined(__GLIBC__) && (__GLIBC__ == 2) && (__GLIBC_MINOR__ == 0)
        /* /usr/include/wctype.h incorrectly declares translations */
        /* tables which provokes tons of compile-time warnings -- try */
        /* to correct this */
        #define wxCRT_TolowerW(wc) towctrans((wc), (wctrans_t)__ctype_tolower)
        #define wxCRT_ToupperW(wc) towctrans((wc), (wctrans_t)__ctype_toupper)
    #else /* !glibc 2.0 */
        #define wxCRT_TolowerW   towlower
        #define wxCRT_ToupperW   towupper
    #endif
#else /* !__GLIBC__ */
    /* There is a bug in MSVC RTL: toxxx() functions don't do anything
       with signed chars < 0, so "fix" it here. */
    #define wxCRT_TolowerW(c)   towlower((wxUChar)(wxChar)(c))
    #define wxCRT_ToupperW(c)   towupper((wxUChar)(wxChar)(c))
#endif /* __GLIBC__/!__GLIBC__ */

/* The Android platform, as of 2014, only support most wide-char function with
   the exception of multi-byte encoding/decoding functions & wsprintf/wsscanf
   See android-ndk-r9d/docs/STANDALONE-TOOLCHAIN.html (section 7.2)
   In fact, mbstowcs/wcstombs are defined and compile, but don't work correctly
*/

#if defined(__WXQT__) && defined(__ANDROID__)
    #define wxNEED_WX_MBSTOWCS
    #undef HAVE_WCSRTOMBS
    // TODO: use Qt built-in required functionality
#endif

#if defined(wxNEED_WX_MBSTOWCS) && defined(__ANDROID__)
    #warning "Custom mb/wchar conv. only works for ASCII, see Android NDK notes"
    WXDLLIMPEXP_BASE size_t android_mbstowcs(wchar_t *, const char *, size_t);
    WXDLLIMPEXP_BASE size_t android_wcstombs(char *, const wchar_t *, size_t);
    #define wxMbstowcs android_mbstowcs
    #define wxWcstombs android_wcstombs
#else
    #define wxMbstowcs mbstowcs
    #define wxWcstombs wcstombs
#endif


/* -------------------------------------------------------------------------
       wx wrappers for CRT functions in both char* and wchar_t* versions
   ------------------------------------------------------------------------- */

#ifdef __cplusplus

/* NB: this belongs to wxcrt.h and not this header, but it makes life easier
 *     for buffer.h and stringimpl.h (both of which must be included before
 *     string.h, which is required by wxcrt.h) to have them here: */

/* safe version of strlen() (returns 0 if passed NULL pointer) */
inline size_t wxStrlen(const char *s) { return s ? wxCRT_StrlenA(s) : 0; }
inline size_t wxStrlen(const wchar_t *s) { return s ? wxCRT_StrlenW(s) : 0; }
#ifndef wxWCHAR_T_IS_WXCHAR16
       WXDLLIMPEXP_BASE size_t wxStrlen(const wxChar16 *s );
#endif
#ifndef wxWCHAR_T_IS_WXCHAR32
       WXDLLIMPEXP_BASE size_t wxStrlen(const wxChar32 *s );
#endif
#define wxWcslen wxCRT_StrlenW

#define wxStrdupA wxCRT_StrdupA
#define wxStrdupW wxCRT_StrdupW
inline char* wxStrdup(const char *s) { return wxCRT_StrdupA(s); }
inline wchar_t* wxStrdup(const wchar_t *s) { return wxCRT_StrdupW(s); }
#ifndef wxWCHAR_T_IS_WXCHAR16
       WXDLLIMPEXP_BASE wxChar16* wxStrdup(const wxChar16* s);
#endif
#ifndef wxWCHAR_T_IS_WXCHAR32
       WXDLLIMPEXP_BASE wxChar32* wxStrdup(const wxChar32* s);
#endif

#endif /* __cplusplus */

#endif /* _WX_WXCRTBASE_H_ */
