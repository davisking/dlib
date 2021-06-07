///////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/creddlgg.h
// Purpose:     wxGenericCredentialEntryDialog interface
// Author:      Tobias Taschner
// Created:     2018-10-23
// Copyright:   (c) 2018 wxWidgets development team
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_CREDDLGG_H_BASE
#define _WX_CREDDLGG_H_BASE

#include "wx/defs.h"

#if wxUSE_CREDENTIALDLG

#include "wx/dialog.h"
#include "wx/webrequest.h"

class WXDLLIMPEXP_CORE wxGenericCredentialEntryDialog : public wxDialog
{
public:
    wxGenericCredentialEntryDialog();

    wxGenericCredentialEntryDialog(wxWindow* parent, const wxString& message,
        const wxString& title,
        const wxWebCredentials& cred = wxWebCredentials());

    bool Create(wxWindow* parent, const wxString& message,
        const wxString& title,
        const wxWebCredentials& cred = wxWebCredentials());

    void SetUser(const wxString& user) { m_userTextCtrl->SetValue(user); }
    void SetPassword(const wxString& password)
        { m_passwordTextCtrl->SetValue(password); }

    wxWebCredentials GetCredentials() const;

private:
    wxTextCtrl* m_userTextCtrl;
    wxTextCtrl* m_passwordTextCtrl;

    void Init(const wxString& message, const wxWebCredentials& cred);

    wxDECLARE_NO_COPY_CLASS(wxGenericCredentialEntryDialog);
};

// Add this typedef as long as the generic version is the only one available
typedef wxGenericCredentialEntryDialog wxCredentialEntryDialog;

#endif // wxUSE_CREDENTIALDLG

#endif // _WX_CREDDLGG_H_BASE
