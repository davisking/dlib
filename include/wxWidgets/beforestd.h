///////////////////////////////////////////////////////////////////////////////
// Name:        wx/beforestd.h
// Purpose:     #include before STL headers
// Author:      Vadim Zeitlin
// Modified by:
// Created:     07/07/03
// Copyright:   (c) 2003 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

/**
    Unfortunately, when compiling at maximum warning level, the standard
    headers themselves may generate warnings -- and really lots of them. So
    before including them, this header should be included to temporarily
    suppress the warnings and after this the header afterstd.h should be
    included to enable them back again.

    Note that there are intentionally no inclusion guards in this file, because
    it can be included several times.
 */

#if defined(__VISUALC__) && __VISUALC__ >= 1910
    #pragma warning(push, 1)

    // This warning, given when a malloc.h from 10.0.240.0 version of UCRT,
    // which is used when building projects targeting 8.1 SDK, compiled by MSVS
    // 2017 or later, is still given even at warning level 1, in spite of it
    // being level 4, so we have to explicitly disable it here (as we do it
    // after the warning push pragma, it will be restored after pop).
    //
    // expression before comma has no effect; expected expression with side-effect
    #pragma warning(disable:4548)
#endif // VC++ >= 14.1

/**
    GCC's visibility support is broken for libstdc++ in some older versions
    (namely Debian/Ubuntu's GCC 4.1, see
    https://bugs.launchpad.net/ubuntu/+source/gcc-4.1/+bug/109262). We fix it
    here by mimicking newer versions' behaviour of using default visibility
    for libstdc++ code.
 */
#if defined(HAVE_VISIBILITY) && defined(HAVE_BROKEN_LIBSTDCXX_VISIBILITY)
    #pragma GCC visibility push(default)
#endif
