/////////////////////////////////////////////////////////////////////////////
// Name:        wx/statbmp.h
// Purpose:     wxStaticBitmap class interface
// Author:      Vadim Zeitlin
// Modified by:
// Created:     25.08.00
// Copyright:   (c) 2000 Vadim Zeitlin
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_STATBMP_H_BASE_
#define _WX_STATBMP_H_BASE_

#include "wx/defs.h"

#if wxUSE_STATBMP

#include "wx/control.h"
#include "wx/bitmap.h"
#include "wx/icon.h"

extern WXDLLIMPEXP_DATA_CORE(const char) wxStaticBitmapNameStr[];

// a control showing an icon or a bitmap
class WXDLLIMPEXP_CORE wxStaticBitmapBase : public wxControl
{
public:
    enum ScaleMode
    {
        Scale_None,
        Scale_Fill,
        Scale_AspectFit,
        Scale_AspectFill
    };

    wxStaticBitmapBase() { }
    virtual ~wxStaticBitmapBase();

    // our interface
    virtual void SetIcon(const wxIcon& icon) = 0;
    virtual void SetBitmap(const wxBitmap& bitmap) = 0;
    virtual wxBitmap GetBitmap() const = 0;
    virtual wxIcon GetIcon() const /* = 0 -- should be pure virtual */
    {
        // stub it out here for now as not all ports implement it (but they
        // should)
        return wxIcon();
    }
    virtual void SetScaleMode(ScaleMode WXUNUSED(scaleMode)) { }
    virtual ScaleMode GetScaleMode() const { return Scale_None; }

    // overridden base class virtuals
    virtual bool AcceptsFocus() const wxOVERRIDE { return false; }
    virtual bool HasTransparentBackground() wxOVERRIDE { return true; }

protected:
    // choose the default border for this window
    virtual wxBorder GetDefaultBorder() const wxOVERRIDE { return wxBORDER_NONE; }

    virtual wxSize DoGetBestSize() const wxOVERRIDE;

    wxDECLARE_NO_COPY_CLASS(wxStaticBitmapBase);
};

#if defined(__WXUNIVERSAL__)
    #include "wx/univ/statbmp.h"
#elif defined(__WXMSW__)
    #include "wx/msw/statbmp.h"
#elif defined(__WXMOTIF__)
    #include "wx/motif/statbmp.h"
#elif defined(__WXGTK20__)
    #include "wx/gtk/statbmp.h"
#elif defined(__WXGTK__)
    #include "wx/gtk1/statbmp.h"
#elif defined(__WXMAC__)
    #include "wx/osx/statbmp.h"
#elif defined(__WXQT__)
    #include "wx/qt/statbmp.h"
#endif

#endif // wxUSE_STATBMP

#endif
    // _WX_STATBMP_H_BASE_
