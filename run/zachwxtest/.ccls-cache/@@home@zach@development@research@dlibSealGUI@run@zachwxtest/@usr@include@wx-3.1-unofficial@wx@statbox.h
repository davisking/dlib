/////////////////////////////////////////////////////////////////////////////
// Name:        wx/statbox.h
// Purpose:     wxStaticBox base header
// Author:      Julian Smart
// Modified by:
// Created:
// Copyright:   (c) Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_STATBOX_H_BASE_
#define _WX_STATBOX_H_BASE_

#include "wx/defs.h"

#if wxUSE_STATBOX

#include "wx/control.h"
#include "wx/containr.h"

extern WXDLLIMPEXP_DATA_CORE(const char) wxStaticBoxNameStr[];

// ----------------------------------------------------------------------------
// wxStaticBox: a grouping box with a label
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxStaticBoxBase : public wxNavigationEnabled<wxControl>
{
public:
    wxStaticBoxBase();

    // overridden base class virtuals
    virtual bool HasTransparentBackground() wxOVERRIDE { return true; }
    virtual bool Enable(bool enable = true) wxOVERRIDE;

    // implementation only: this is used by wxStaticBoxSizer to account for the
    // need for extra space taken by the static box
    //
    // the top border is the margin at the top (where the title is),
    // borderOther is the margin on all other sides
    virtual void GetBordersForSizer(int *borderTop, int *borderOther) const;

    // This is an internal function currently used by wxStaticBoxSizer only.
    //
    // Reparent all children of the static box under its parent and destroy the
    // box itself.
    void WXDestroyWithoutChildren();

protected:
    // choose the default border for this window
    virtual wxBorder GetDefaultBorder() const wxOVERRIDE { return wxBORDER_NONE; }

    // If non-null, the window used as our label. This window is owned by the
    // static box and will be deleted when it is.
    wxWindow* m_labelWin;

    // For boxes with window label this member variable is used instead of
    // m_isEnabled to remember the last value passed to Enable(). It is
    // required because the box itself doesn't get disabled by Enable(false) in
    // this case (see comments in Enable() implementation), and m_isEnabled
    // must correspond to its real state.
    bool m_areChildrenEnabled;

    wxDECLARE_NO_COPY_CLASS(wxStaticBoxBase);
};

#if defined(__WXUNIVERSAL__)
    #include "wx/univ/statbox.h"
#elif defined(__WXMSW__)
    #include "wx/msw/statbox.h"
#elif defined(__WXMOTIF__)
    #include "wx/motif/statbox.h"
#elif defined(__WXGTK20__)
    #include "wx/gtk/statbox.h"
#elif defined(__WXGTK__)
    #include "wx/gtk1/statbox.h"
#elif defined(__WXMAC__)
    #include "wx/osx/statbox.h"
#elif defined(__WXQT__)
    #include "wx/qt/statbox.h"
#endif

#endif // wxUSE_STATBOX

#endif
    // _WX_STATBOX_H_BASE_
