/*
 * Name:        wx/types.h
 * Purpose:     Declarations of common wx types and related constants.
 * Author:      Vadim Zeitlin (extracted from wx/defs.h)
 * Created:     2018-01-07
 * Copyright:   (c) 1997-2018 wxWidgets dev team
 * Licence:     wxWindows licence
 */

/* THIS IS A C FILE, DON'T USE C++ FEATURES (IN PARTICULAR COMMENTS) IN IT */

#ifndef _WX_TYPES_H_
#define _WX_TYPES_H_

/*
    Don't include wx/defs.h from here as we're included from it, but do include
    wx/platform.h which will take care of including wx/setup.h too.
 */
#include "wx/platform.h"

/*  ---------------------------------------------------------------------------- */
/*  standard wxWidgets types */
/*  ---------------------------------------------------------------------------- */

/*  the type for screen and DC coordinates */
typedef int wxCoord;

enum {  wxDefaultCoord = -1 };

/*  ---------------------------------------------------------------------------- */
/*  define fixed length types */
/*  ---------------------------------------------------------------------------- */

#if defined(__MINGW32__)
    #include <sys/types.h>
#endif

/*  chars are always one byte (by definition), shorts are always two (in */
/*  practice) */

/*  8bit */
typedef signed char wxInt8;
typedef unsigned char wxUint8;
typedef wxUint8 wxByte;


/*  16bit */
#ifdef SIZEOF_SHORT
    #if SIZEOF_SHORT != 2
        #error "wxWidgets assumes sizeof(short) == 2, please fix the code"
    #endif
#else
    #define SIZEOF_SHORT 2
#endif

typedef signed short wxInt16;
typedef unsigned short wxUint16;

typedef wxUint16 wxWord;

/*
  things are getting more interesting with ints, longs and pointers

  there are several different standard data models described by this table:

  +-----------+----------------------------+
  |type\model | LP64 ILP64 LLP64 ILP32 LP32|
  +-----------+----------------------------+
  |char       |  8     8     8     8     8 |
  |short      | 16    16    16    16    16 |
  |int        | 32    64    32    32    16 |
  |long       | 64    64    32    32    32 |
  |long long  | 64    64    64    --    -- |
  |void *     | 64    64    64    32    32 |
  +-----------+----------------------------+

  Win16 used LP32 (but we don't support it any longer), Win32 obviously used
  ILP32 and Win64 uses LLP64 (a.k.a. P64)

  Under Unix LP64 is the most widely used (the only I've ever seen, in fact)
 */

/*  32bit */
#if defined(__WINDOWS__)
    #if defined(__WIN32__)
        typedef int wxInt32;
        typedef unsigned int wxUint32;

        /*
            Win64 uses LLP64 model and so ints and longs have the same size as
            in Win32.
         */
        #ifndef SIZEOF_INT
            #define SIZEOF_INT 4
        #endif

        #ifndef SIZEOF_LONG
            #define SIZEOF_LONG 4
        #endif

        #ifndef SIZEOF_LONG_LONG
            #define SIZEOF_LONG_LONG 8
        #endif

        #ifndef SIZEOF_WCHAR_T
            /* Windows uses UTF-16 */
            #define SIZEOF_WCHAR_T 2
        #endif

        #ifndef SIZEOF_SIZE_T
            /*
               Under Win64 sizeof(size_t) == 8 and so it is neither unsigned
               int nor unsigned long!
             */
            #ifdef __WIN64__
                #define SIZEOF_SIZE_T 8

                #undef wxSIZE_T_IS_UINT
            #else /* Win32 */
                #define SIZEOF_SIZE_T 4

                #define wxSIZE_T_IS_UINT
            #endif
            #undef wxSIZE_T_IS_ULONG
        #endif

        #ifndef SIZEOF_VOID_P
            #ifdef __WIN64__
                #define SIZEOF_VOID_P 8
            #else /*  Win32 */
                #define SIZEOF_VOID_P 4
            #endif /*  Win64/32 */
        #endif
    #else
        #error "Unsupported Windows version"
    #endif
#else /*  !Windows */
    /*  SIZEOF_XXX are normally defined by configure */
    #ifdef SIZEOF_INT
        #if SIZEOF_INT == 8
            /*  must be ILP64 data model, there is normally a special 32 bit */
            /*  type in it but we don't know what it is... */
            #error "No 32bit int type on this platform"
        #elif SIZEOF_INT == 4
            typedef int wxInt32;
            typedef unsigned int wxUint32;
        #elif SIZEOF_INT == 2
            /*  must be LP32 */
            #if SIZEOF_LONG != 4
                #error "No 32bit int type on this platform"
            #endif

            typedef long wxInt32;
            typedef unsigned long wxUint32;
        #else
            /*  wxWidgets is not ready for 128bit systems yet... */
            #error "Unknown sizeof(int) value, what are you compiling for?"
        #endif
    #else /*  !defined(SIZEOF_INT) */
        /*  assume default 32bit machine -- what else can we do? */
        wxCOMPILE_TIME_ASSERT( sizeof(int) == 4, IntMustBeExactly4Bytes);
        wxCOMPILE_TIME_ASSERT( sizeof(size_t) == 4, SizeTMustBeExactly4Bytes);
        wxCOMPILE_TIME_ASSERT( sizeof(void *) == 4, PtrMustBeExactly4Bytes);

        #define SIZEOF_INT 4
        #define SIZEOF_SIZE_T 4
        #define SIZEOF_VOID_P 4

        typedef int wxInt32;
        typedef unsigned int wxUint32;

        #if defined(__MACH__) && !defined(SIZEOF_WCHAR_T)
            #define SIZEOF_WCHAR_T 4
        #endif
        #if !defined(SIZEOF_WCHAR_T)
            /*  also assume that sizeof(wchar_t) == 2 (under Unix the most */
            /*  common case is 4 but there configure would have defined */
            /*  SIZEOF_WCHAR_T for us) */
            /*  the most common case */
            wxCOMPILE_TIME_ASSERT( sizeof(wchar_t) == 2,
                                    Wchar_tMustBeExactly2Bytes);

            #define SIZEOF_WCHAR_T 2
        #endif /*  !defined(SIZEOF_WCHAR_T) */
    #endif
#endif /*  Win/!Win */

#ifndef SIZEOF_WCHAR_T
    #error "SIZEOF_WCHAR_T must be defined, but isn't"
#endif

/* also define C99-like sized MIN/MAX constants */
#define wxINT8_MIN CHAR_MIN
#define wxINT8_MAX CHAR_MAX
#define wxUINT8_MAX UCHAR_MAX

#define wxINT16_MIN SHRT_MIN
#define wxINT16_MAX SHRT_MAX
#define wxUINT16_MAX USHRT_MAX

#if SIZEOF_INT == 4
    #define wxINT32_MIN INT_MIN
    #define wxINT32_MAX INT_MAX
    #define wxUINT32_MAX UINT_MAX
#elif SIZEOF_LONG == 4
    #define wxINT32_MIN LONG_MIN
    #define wxINT32_MAX LONG_MAX
    #define wxUINT32_MAX ULONG_MAX
#else
    #error "Unknown 32 bit type"
#endif

typedef wxUint32 wxDword;

#ifdef LLONG_MAX
    #define wxINT64_MIN LLONG_MIN
    #define wxINT64_MAX LLONG_MAX
    #define wxUINT64_MAX ULLONG_MAX
#else
    #define wxINT64_MIN (wxLL(-9223372036854775807)-1)
    #define wxINT64_MAX wxLL(9223372036854775807)
    #define wxUINT64_MAX wxULL(0xFFFFFFFFFFFFFFFF)
#endif

/*  64 bit */

/*  NB: we #define and not typedef wxLongLong_t because we use "#ifdef */
/*      wxLongLong_t" in wx/longlong.h */

/*      wxULongLong_t is set later (usually to unsigned wxLongLong_t) */

/*  to avoid compilation problems on 64bit machines with ambiguous method calls */
/*  we will need to define this */
#undef wxLongLongIsLong

/*
   First check for specific compilers which have known 64 bit integer types,
   this avoids clashes with SIZEOF_LONG[_LONG] being defined incorrectly for
   e.g. MSVC builds (Python.h defines it as 8 even for MSVC).

   Also notice that we check for "long long" before checking for 64 bit long as
   we still want to use "long long" and not "long" for wxLongLong_t on 64 bit
   architectures to be able to pass wxLongLong_t to the standard functions
   prototyped as taking "long long" such as strtoll().
 */
#if (defined(__VISUALC__) || defined(__INTELC__)) && defined(__WIN32__)
    #define wxLongLong_t __int64
    #define wxLongLongSuffix i64
    #define wxLongLongFmtSpec "I64"
#elif defined(__MINGW32__) && \
    (defined(__USE_MINGW_ANSI_STDIO) && (__USE_MINGW_ANSI_STDIO != 1))
    #define wxLongLong_t long long
    #define wxLongLongSuffix ll
    #define wxLongLongFmtSpec "I64"
#elif (defined(SIZEOF_LONG_LONG) && SIZEOF_LONG_LONG >= 8)  || \
        defined(__GNUC__) || \
        defined(__CYGWIN__)
    #define wxLongLong_t long long
    #define wxLongLongSuffix ll
    #define wxLongLongFmtSpec "ll"
#elif defined(SIZEOF_LONG) && (SIZEOF_LONG == 8)
    #define wxLongLong_t long
    #define wxLongLongSuffix l
    #define wxLongLongFmtSpec "l"
    #define wxLongLongIsLong
#endif


#ifdef wxLongLong_t
    #define wxULongLong_t unsigned wxLongLong_t

    /*
        wxLL() and wxULL() macros allow to define 64 bit constants in a
        portable way.
     */
    #define wxLL(x) wxCONCAT(x, wxLongLongSuffix)
    #define wxULL(x) wxCONCAT(x, wxCONCAT(u, wxLongLongSuffix))

    typedef wxLongLong_t wxInt64;
    typedef wxULongLong_t wxUint64;

    #define wxHAS_INT64 1

    #ifndef wxLongLongIsLong
        #define wxHAS_LONG_LONG_T_DIFFERENT_FROM_LONG
    #endif
#elif wxUSE_LONGLONG
    /*  these macros allow to define 64 bit constants in a portable way */
    #define wxLL(x) wxLongLong(x)
    #define wxULL(x) wxULongLong(x)

    #define wxInt64 wxLongLong
    #define wxUint64 wxULongLong

    #define wxHAS_INT64 1

#else /* !wxUSE_LONGLONG */

    #define wxHAS_INT64 0

#endif

/*
    Helper macro for conditionally compiling some code only if wxLongLong_t is
    available and is a type different from the other integer types (i.e. not
    long).
 */
#ifdef wxHAS_LONG_LONG_T_DIFFERENT_FROM_LONG
    #define wxIF_LONG_LONG_TYPE(x) x
#else
    #define wxIF_LONG_LONG_TYPE(x)
#endif


/* Make sure ssize_t is defined (a signed type the same size as size_t). */
/* (HAVE_SSIZE_T is not already defined by configure) */
#ifndef HAVE_SSIZE_T
#ifdef __MINGW32__
    #if defined(_SSIZE_T_) || defined(_SSIZE_T_DEFINED)
        #define HAVE_SSIZE_T
    #endif
#endif
#endif /* !HAVE_SSIZE_T */

/* If we really don't have ssize_t, provide our own version. */
#ifdef HAVE_SSIZE_T
    #ifdef __UNIX__
        #include <sys/types.h>
    #endif
#else /* !HAVE_SSIZE_T */
    #if SIZEOF_SIZE_T == 4
        typedef wxInt32 ssize_t;
    #elif SIZEOF_SIZE_T == 8
        typedef wxInt64 ssize_t;
    #else
        #error "error defining ssize_t, size_t is not 4 or 8 bytes"
    #endif

    /* prevent ssize_t redefinitions in other libraries */
    #define HAVE_SSIZE_T
#endif

/*
    We can't rely on Windows _W64 being defined as windows.h may not be
    included so define our own equivalent: this should be used with types
    like WXLPARAM or WXWPARAM which are 64 bit under Win64 to avoid warnings
    each time we cast it to a pointer or a handle (which results in hundreds
    of warnings as Win32 API often passes pointers in them)
 */
#ifdef __VISUALC__
    #define wxW64 __w64
#else
    #define wxW64
#endif

/*
   Define signed and unsigned integral types big enough to contain all of long,
   size_t and void *.
 */
#if SIZEOF_LONG >= SIZEOF_VOID_P
    /*
       Normal case when long is the largest integral type.
     */
    typedef long wxIntPtr;
    typedef unsigned long wxUIntPtr;
#elif SIZEOF_SIZE_T >= SIZEOF_VOID_P
    /*
       Win64 case: size_t is the only integral type big enough for "void *".

       Notice that we must use __w64 to avoid warnings about casting pointers
       to wxIntPtr (which we do often as this is what it is defined for) in 32
       bit build with MSVC.
     */
    typedef wxW64 ssize_t wxIntPtr;
    typedef size_t wxUIntPtr;
#else
    /*
       This should never happen for the current architectures but if you're
       using one where it does, please contact wx-dev@googlegroups.com.
     */
    #error "Pointers can't be stored inside integer types."
#endif

#endif //  _WX_TYPES_H_
