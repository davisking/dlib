///////////////////////////////////////////////////////////////////////////////
// Name:        wx/ioswrap.h
// Purpose:     This file is obsolete, include <iostream> directly instead.
// Author:      Vadim Zeitlin
// Modified by:
// Created:     03.02.99
// Copyright:   (c) 1998 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#if wxUSE_STD_IOSTREAM

#include "wx/beforestd.h"

#include <iostream>

#include "wx/afterstd.h"

#ifdef __WINDOWS__
#   include "wx/msw/winundef.h"
#endif

#endif
  // wxUSE_STD_IOSTREAM

