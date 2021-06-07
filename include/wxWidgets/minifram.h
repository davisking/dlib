/////////////////////////////////////////////////////////////////////////////
// Name:        wx/minifram.h
// Purpose:     wxMiniFrame base header
// Author:      Julian Smart
// Modified by:
// Created:
// Copyright:   (c) Julian Smart
// Copyright:   (c) 2014 wxWidgets dev team
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_MINIFRAM_H_BASE_
#define _WX_MINIFRAM_H_BASE_

#include "wx/defs.h"

#if wxUSE_MINIFRAME

#if defined(__WXMSW__)
#include "wx/msw/minifram.h"
#elif defined(__WXMOTIF__)
#include "wx/motif/minifram.h"
#elif defined(__WXGTK20__)
#include "wx/gtk/minifram.h"
#elif defined(__WXGTK__)
#include "wx/gtk1/minifram.h"
#elif defined(__WXX11__)
#include "wx/x11/minifram.h"
#elif defined(__WXMAC__)
#include "wx/osx/minifram.h"
#elif defined(__WXQT__)
#include "wx/qt/minifram.h"
#else
// TODO: it seems that wxMiniFrame could be just defined here generically
//       instead of having all the above port-specific headers
#include "wx/frame.h"
typedef wxFrame wxMiniFrame;
#endif

#endif // wxUSE_MINIFRAME
#endif // _WX_MINIFRAM_H_BASE_
