/////////////////////////////////////////////////////////////////////////////
// Name:        wx/dirdlg.h
// Purpose:     wxDirDialog base class
// Author:      Robert Roebling
// Modified by:
// Created:
// Copyright:   (c) Robert Roebling
// Licence:     wxWindows Licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_DIRDLG_H_BASE_
#define _WX_DIRDLG_H_BASE_

#include "wx/defs.h"

#if wxUSE_DIRDLG

#include "wx/dialog.h"

// ----------------------------------------------------------------------------
// constants
// ----------------------------------------------------------------------------

extern WXDLLIMPEXP_DATA_CORE(const char) wxDirDialogNameStr[];
extern WXDLLIMPEXP_DATA_CORE(const char) wxDirDialogDefaultFolderStr[];
extern WXDLLIMPEXP_DATA_CORE(const char) wxDirSelectorPromptStr[];


/*
    The flags below must coexist with the following flags in m_windowStyle
    #define wxCAPTION               0x20000000
    #define wxMAXIMIZE              0x00002000
    #define wxCLOSE_BOX             0x00001000
    #define wxSYSTEM_MENU           0x00000800
    wxBORDER_NONE   =               0x00200000
    #define wxRESIZE_BORDER         0x00000040
    #define wxDIALOG_NO_PARENT      0x00000020
*/

#define wxDD_CHANGE_DIR         0x0100
#define wxDD_DIR_MUST_EXIST     0x0200
#define wxDD_MULTIPLE           0x0400
#define wxDD_SHOW_HIDDEN        0x0001

// deprecated, on by default now, use wxDD_DIR_MUST_EXIST to disable it
#define wxDD_NEW_DIR_BUTTON     0

#define wxDD_DEFAULT_STYLE      (wxDEFAULT_DIALOG_STYLE|wxRESIZE_BORDER)

//-------------------------------------------------------------------------
// wxDirDialogBase
//-------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDirDialogBase : public wxDialog
{
public:
    wxDirDialogBase() {}
    wxDirDialogBase(wxWindow *parent,
                    const wxString& title = wxASCII_STR(wxDirSelectorPromptStr),
                    const wxString& defaultPath = wxEmptyString,
                    long style = wxDD_DEFAULT_STYLE,
                    const wxPoint& pos = wxDefaultPosition,
                    const wxSize& sz = wxDefaultSize,
                    const wxString& name = wxASCII_STR(wxDirDialogNameStr))
    {
        Create(parent, title, defaultPath, style, pos, sz, name);
    }

    virtual ~wxDirDialogBase() {}


    bool Create(wxWindow *parent,
                const wxString& title = wxASCII_STR(wxDirSelectorPromptStr),
                const wxString& defaultPath = wxEmptyString,
                long style = wxDD_DEFAULT_STYLE,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& sz = wxDefaultSize,
                const wxString& name = wxASCII_STR(wxDirDialogNameStr))
    {
        if (!wxDialog::Create(parent, wxID_ANY, title, pos, sz, style, name))
            return false;
        m_path = defaultPath;
        m_message = title;
        return true;
    }

    virtual void SetMessage(const wxString& message) { m_message = message; }
    virtual void SetPath(const wxString& path) { m_path = path; }

    virtual wxString GetMessage() const { return m_message; }
    virtual wxString GetPath() const
    {
        wxCHECK_MSG( !HasFlag(wxDD_MULTIPLE), wxString(),
                     "When using wxDD_MULTIPLE, must call GetPaths() instead" );
        return m_path;
    }

    virtual void GetPaths(wxArrayString& paths) const
    {
        paths = m_paths;
    }

protected:
    wxString m_message;
    wxString m_path;
    wxArrayString m_paths;
};


// Universal and non-port related switches with need for generic implementation
#if defined(__WXUNIVERSAL__)
    #include "wx/generic/dirdlgg.h"
    #define wxDirDialog wxGenericDirDialog
#elif defined(__WXMSW__) && !wxUSE_OLE
    #include "wx/generic/dirdlgg.h"
    #define wxDirDialog wxGenericDirDialog
#elif defined(__WXMSW__)
    #include "wx/msw/dirdlg.h"  // Native MSW
#elif defined(__WXGTK20__)
    #include "wx/gtk/dirdlg.h"  // Native GTK for gtk2.4
#elif defined(__WXGTK__)
    #include "wx/generic/dirdlgg.h"
    #define wxDirDialog wxGenericDirDialog
#elif defined(__WXMAC__)
    #include "wx/osx/dirdlg.h"      // Native Mac
#elif defined(__WXMOTIF__) || \
      defined(__WXX11__)
    #include "wx/generic/dirdlgg.h"     // Other ports use generic implementation
    #define wxDirDialog wxGenericDirDialog
#elif defined(__WXQT__)
    #include "wx/qt/dirdlg.h"
#endif

// ----------------------------------------------------------------------------
// common ::wxDirSelector() function
// ----------------------------------------------------------------------------

WXDLLIMPEXP_CORE wxString
wxDirSelector(const wxString& message = wxASCII_STR(wxDirSelectorPromptStr),
              const wxString& defaultPath = wxEmptyString,
              long style = wxDD_DEFAULT_STYLE,
              const wxPoint& pos = wxDefaultPosition,
              wxWindow *parent = NULL);

#endif // wxUSE_DIRDLG

#endif
    // _WX_DIRDLG_H_BASE_
