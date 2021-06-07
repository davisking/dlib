/////////////////////////////////////////////////////////////////////////////
// Name:        wx/toolbar.h
// Purpose:     wxToolBar interface declaration
// Author:      Vadim Zeitlin
// Modified by:
// Created:     20.11.99
// Copyright:   (c) Vadim Zeitlin
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_TOOLBAR_H_BASE_
#define _WX_TOOLBAR_H_BASE_

#include "wx/defs.h"

// ----------------------------------------------------------------------------
// wxToolBar style flags
// ----------------------------------------------------------------------------

enum
{
    // lay out the toolbar horizontally
    wxTB_HORIZONTAL  = wxHORIZONTAL,    // == 0x0004
    wxTB_TOP         = wxTB_HORIZONTAL,

    // lay out the toolbar vertically
    wxTB_VERTICAL    = wxVERTICAL,      // == 0x0008
    wxTB_LEFT        = wxTB_VERTICAL,

    // "flat" buttons (Win32/GTK only)
    wxTB_FLAT        = 0x0020,

    // dockable toolbar (GTK only)
    wxTB_DOCKABLE    = 0x0040,

    // don't show the icons (they're shown by default)
    wxTB_NOICONS     = 0x0080,

    // show the text (not shown by default)
    wxTB_TEXT        = 0x0100,

    // don't show the divider between toolbar and the window (Win32 only)
    wxTB_NODIVIDER   = 0x0200,

    // no automatic alignment (Win32 only, useless)
    wxTB_NOALIGN     = 0x0400,

    // show the text and the icons alongside, not vertically stacked (Win32/GTK)
    wxTB_HORZ_LAYOUT = 0x0800,
    wxTB_HORZ_TEXT   = wxTB_HORZ_LAYOUT | wxTB_TEXT,

    // don't show the toolbar short help tooltips
    wxTB_NO_TOOLTIPS = 0x1000,

    // lay out toolbar at the bottom of the window
    wxTB_BOTTOM       = 0x2000,

    // lay out toolbar at the right edge of the window
    wxTB_RIGHT        = 0x4000,

    wxTB_DEFAULT_STYLE = wxTB_HORIZONTAL
};

#if wxUSE_TOOLBAR
    #include "wx/tbarbase.h"     // the base class for all toolbars

    #if defined(__WXUNIVERSAL__)
       #include "wx/univ/toolbar.h"
    #elif defined(__WXMSW__)
       #include "wx/msw/toolbar.h"
    #elif defined(__WXMOTIF__)
       #include "wx/motif/toolbar.h"
    #elif defined(__WXGTK20__)
        #include "wx/gtk/toolbar.h"
    #elif defined(__WXGTK__)
        #include "wx/gtk1/toolbar.h"
    #elif defined(__WXMAC__)
       #include "wx/osx/toolbar.h"
    #elif defined(__WXQT__)
        #include "wx/qt/toolbar.h"
    #endif
#endif // wxUSE_TOOLBAR

#endif
    // _WX_TOOLBAR_H_BASE_
