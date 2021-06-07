/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/mdi.h
// Purpose:     TDI-based MDI implementation for wxGTK
// Author:      Robert Roebling
// Modified by: 2008-10-31 Vadim Zeitlin: derive from the base classes
// Copyright:   (c) 1998 Robert Roebling
//              (c) 2008 Vadim Zeitlin
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTK_MDI_H_
#define _WX_GTK_MDI_H_

#include "wx/frame.h"

class WXDLLIMPEXP_FWD_CORE wxMDIChildFrame;
class WXDLLIMPEXP_FWD_CORE wxMDIClientWindow;

typedef struct _GtkNotebook GtkNotebook;

//-----------------------------------------------------------------------------
// wxMDIParentFrame
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxMDIParentFrame : public wxMDIParentFrameBase
{
public:
    wxMDIParentFrame() { Init(); }
    wxMDIParentFrame(wxWindow *parent,
                     wxWindowID id,
                     const wxString& title,
                     const wxPoint& pos = wxDefaultPosition,
                     const wxSize& size = wxDefaultSize,
                     long style = wxDEFAULT_FRAME_STYLE | wxVSCROLL | wxHSCROLL,
                     const wxString& name = wxASCII_STR(wxFrameNameStr))
    {
        Init();

        (void)Create(parent, id, title, pos, size, style, name);
    }

    bool Create(wxWindow *parent,
                wxWindowID id,
                const wxString& title,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize,
                long style = wxDEFAULT_FRAME_STYLE | wxVSCROLL | wxHSCROLL,
                const wxString& name = wxASCII_STR(wxFrameNameStr));

    // we don't store the active child in m_currentChild unlike the base class
    // version so override this method to find it dynamically
    virtual wxMDIChildFrame *GetActiveChild() const wxOVERRIDE;

    // implement base class pure virtuals
    // ----------------------------------

    virtual void ActivateNext() wxOVERRIDE;
    virtual void ActivatePrevious() wxOVERRIDE;

    static bool IsTDI() { return true; }

    // implementation

    bool                m_justInserted;

    virtual void OnInternalIdle() wxOVERRIDE;

protected:
    virtual void DoGetClientSize(int* width, int* height) const wxOVERRIDE;

private:
    friend class wxMDIChildFrame;
    void Init();

    wxDECLARE_DYNAMIC_CLASS(wxMDIParentFrame);
};

//-----------------------------------------------------------------------------
// wxMDIChildFrame
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxMDIChildFrame : public wxTDIChildFrame
{
public:
    wxMDIChildFrame() { Init(); }
    wxMDIChildFrame(wxMDIParentFrame *parent,
                    wxWindowID id,
                    const wxString& title,
                    const wxPoint& pos = wxDefaultPosition,
                    const wxSize& size = wxDefaultSize,
                    long style = wxDEFAULT_FRAME_STYLE,
                    const wxString& name = wxASCII_STR(wxFrameNameStr))
    {
        Init();

        Create(parent, id, title, pos, size, style, name);
    }

    bool Create(wxMDIParentFrame *parent,
                wxWindowID id,
                const wxString& title,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize,
                long style = wxDEFAULT_FRAME_STYLE,
                const wxString& name = wxASCII_STR(wxFrameNameStr));

    virtual ~wxMDIChildFrame();

    virtual void SetMenuBar( wxMenuBar *menu_bar ) wxOVERRIDE;
    virtual wxMenuBar *GetMenuBar() const wxOVERRIDE;

    virtual void Activate() wxOVERRIDE;

    virtual void SetTitle(const wxString& title) wxOVERRIDE;

    // implementation

    void OnActivate( wxActivateEvent& event );
    void OnMenuHighlight( wxMenuEvent& event );
    virtual void GTKHandleRealized() wxOVERRIDE;

    wxMenuBar         *m_menuBar;
    bool               m_justInserted;

protected:
    virtual void DoGetPosition(int *x, int *y) const wxOVERRIDE;

private:
    void Init();

    GtkNotebook *GTKGetNotebook() const;

    wxDECLARE_EVENT_TABLE();
    wxDECLARE_DYNAMIC_CLASS(wxMDIChildFrame);
};

//-----------------------------------------------------------------------------
// wxMDIClientWindow
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxMDIClientWindow : public wxMDIClientWindowBase
{
public:
    wxMDIClientWindow() { }
    ~wxMDIClientWindow();

    virtual bool CreateClient(wxMDIParentFrame *parent,
                              long style = wxVSCROLL | wxHSCROLL) wxOVERRIDE;

private:
    virtual void AddChildGTK(wxWindowGTK* child) wxOVERRIDE;

    wxDECLARE_DYNAMIC_CLASS(wxMDIClientWindow);
};

#endif // _WX_GTK_MDI_H_
