///////////////////////////////////////////////////////////////////////////////
// Name:        wx/tipwin.h
// Purpose:     wxTipWindow is a window like the one typically used for
//              showing the tooltips
// Author:      Vadim Zeitlin
// Modified by:
// Created:     10.09.00
// Copyright:   (c) 2000 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_TIPWIN_H_
#define _WX_TIPWIN_H_

#if wxUSE_TIPWINDOW

#include "wx/popupwin.h"

class WXDLLIMPEXP_FWD_CORE wxTipWindowView;

// ----------------------------------------------------------------------------
// wxTipWindow
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxTipWindow : public wxPopupTransientWindow
{
public:
    // the mandatory ctor parameters are: the parent window and the text to
    // show
    //
    // optionally you may also specify the length at which the lines are going
    // to be broken in rows (100 pixels by default)
    //
    // windowPtr and rectBound are just passed to SetTipWindowPtr() and
    // SetBoundingRect() - see below
    wxTipWindow(wxWindow *parent,
                const wxString& text,
                wxCoord maxLength = 100,
                wxTipWindow** windowPtr = NULL,
                wxRect *rectBound = NULL);

    virtual ~wxTipWindow();

    // If windowPtr is not NULL the given address will be NULLed when the
    // window has closed
    void SetTipWindowPtr(wxTipWindow** windowPtr) { m_windowPtr = windowPtr; }

    // If rectBound is not NULL, the window will disappear automatically when
    // the mouse leave the specified rect: note that rectBound should be in the
    // screen coordinates!
    void SetBoundingRect(const wxRect& rectBound);

    // Hide and destroy the window
    void Close();

protected:
    // called by wxTipWindowView only
    bool CheckMouseInBounds(const wxPoint& pos);

    // event handlers
    void OnMouseClick(wxMouseEvent& event);

    virtual void OnDismiss() wxOVERRIDE;

private:
    wxTipWindowView *m_view;

    wxTipWindow** m_windowPtr;
    wxRect m_rectBound;

    wxDECLARE_EVENT_TABLE();

    friend class wxTipWindowView;

    wxDECLARE_NO_COPY_CLASS(wxTipWindow);
};

#endif // wxUSE_TIPWINDOW

#endif // _WX_TIPWIN_H_
