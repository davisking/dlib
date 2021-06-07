/////////////////////////////////////////////////////////////////////////////
// Name:        wx/colordlg.h
// Purpose:     wxColourDialog
// Author:      Vadim Zeitlin
// Modified by:
// Created:     01/02/97
// Copyright:   (c) wxWidgets team
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_COLORDLG_H_BASE_
#define _WX_COLORDLG_H_BASE_

#include "wx/defs.h"

#if wxUSE_COLOURDLG

#include "wx/colourdata.h"

#if defined(__WXMSW__) && !defined(__WXUNIVERSAL__)
    #include "wx/msw/colordlg.h"
#elif defined(__WXMAC__) && !defined(__WXUNIVERSAL__)
    #include "wx/osx/colordlg.h"
#elif defined(__WXGTK20__) && !defined(__WXUNIVERSAL__)
    #include "wx/gtk/colordlg.h"
#elif defined(__WXQT__)
    #include "wx/qt/colordlg.h"
#else
    #include "wx/generic/colrdlgg.h"

    #define wxColourDialog wxGenericColourDialog
#endif

// Under some platforms (currently only wxMSW) wxColourDialog can send events
// of this type while it is shown.
//
// Notice that this class is almost identical to wxColourPickerEvent but it
// doesn't really sense to reuse the same class for both controls.
class WXDLLIMPEXP_CORE wxColourDialogEvent : public wxCommandEvent
{
public:
    wxColourDialogEvent()
    {
    }

    wxColourDialogEvent(wxEventType evtType,
                        wxColourDialog* dialog,
                        const wxColour& colour)
        : wxCommandEvent(evtType, dialog->GetId()),
          m_colour(colour)
    {
        SetEventObject(dialog);
    }

    // default copy ctor and dtor are ok

    wxColour GetColour() const { return m_colour; }
    void SetColour(const wxColour& colour) { m_colour = colour; }

    virtual wxEvent *Clone() const wxOVERRIDE
    {
        return new wxColourDialogEvent(*this);
    }

private:
    wxColour m_colour;

    wxDECLARE_DYNAMIC_CLASS_NO_ASSIGN(wxColourDialogEvent);
};

wxDECLARE_EXPORTED_EVENT(WXDLLIMPEXP_CORE, wxEVT_COLOUR_CHANGED, wxColourDialogEvent);

#define wxColourDialogEventHandler(func) \
    wxEVENT_HANDLER_CAST(wxColourDialogEventFunction, func)

#define EVT_COLOUR_CHANGED(id, fn) \
    wx__DECLARE_EVT1(wxEVT_COLOUR_CHANGED, id, wxColourDialogEventHandler(fn))


// get the colour from user and return it
WXDLLIMPEXP_CORE wxColour wxGetColourFromUser(wxWindow *parent = NULL,
                                              const wxColour& colInit = wxNullColour,
                                              const wxString& caption = wxEmptyString,
                                              wxColourData *data = NULL);

#endif // wxUSE_COLOURDLG

#endif
    // _WX_COLORDLG_H_BASE_
