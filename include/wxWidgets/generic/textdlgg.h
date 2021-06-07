/////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/textdlgg.h
// Purpose:     wxTextEntryDialog class
// Author:      Julian Smart
// Modified by:
// Created:     01/02/97
// Copyright:   (c) Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_TEXTDLGG_H_
#define _WX_TEXTDLGG_H_

#include "wx/defs.h"

#if wxUSE_TEXTDLG

#include "wx/dialog.h"

#if wxUSE_VALIDATORS
#include "wx/valtext.h"
#include "wx/textctrl.h"
#endif

class WXDLLIMPEXP_FWD_CORE wxTextCtrl;

extern WXDLLIMPEXP_DATA_CORE(const char) wxGetTextFromUserPromptStr[];
extern WXDLLIMPEXP_DATA_CORE(const char) wxGetPasswordFromUserPromptStr[];

#define wxTextEntryDialogStyle (wxOK | wxCANCEL | wxCENTRE)

// ----------------------------------------------------------------------------
// wxTextEntryDialog: a dialog with text control, [ok] and [cancel] buttons
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxTextEntryDialog : public wxDialog
{
public:
    wxTextEntryDialog()
    {
        m_textctrl = NULL;
        m_dialogStyle = 0;
    }

    wxTextEntryDialog(wxWindow *parent,
                      const wxString& message,
                      const wxString& caption = wxASCII_STR(wxGetTextFromUserPromptStr),
                      const wxString& value = wxEmptyString,
                      long style = wxTextEntryDialogStyle,
                      const wxPoint& pos = wxDefaultPosition)
    {
        Create(parent, message, caption, value, style, pos);
    }

    bool Create(wxWindow *parent,
                const wxString& message,
                const wxString& caption = wxASCII_STR(wxGetTextFromUserPromptStr),
                const wxString& value = wxEmptyString,
                long style = wxTextEntryDialogStyle,
                const wxPoint& pos = wxDefaultPosition);

    void SetValue(const wxString& val);
    wxString GetValue() const { return m_value; }

    void SetMaxLength(unsigned long len);

    void ForceUpper();

#if wxUSE_VALIDATORS
    void SetTextValidator( const wxTextValidator& validator );
#if WXWIN_COMPATIBILITY_2_8
    wxDEPRECATED( void SetTextValidator( long style ) );
#endif
    void SetTextValidator( wxTextValidatorStyle style = wxFILTER_NONE );
    wxTextValidator* GetTextValidator() { return (wxTextValidator*)m_textctrl->GetValidator(); }
#endif // wxUSE_VALIDATORS

    virtual bool TransferDataToWindow() wxOVERRIDE;
    virtual bool TransferDataFromWindow() wxOVERRIDE;

    // implementation only
    void OnOK(wxCommandEvent& event);

protected:
    wxTextCtrl *m_textctrl;
    wxString    m_value;
    long        m_dialogStyle;

private:
    wxDECLARE_EVENT_TABLE();
    wxDECLARE_DYNAMIC_CLASS(wxTextEntryDialog);
    wxDECLARE_NO_COPY_CLASS(wxTextEntryDialog);
};

// ----------------------------------------------------------------------------
// wxPasswordEntryDialog: dialog with password control, [ok] and [cancel]
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxPasswordEntryDialog : public wxTextEntryDialog
{
public:
    wxPasswordEntryDialog() { }
    wxPasswordEntryDialog(wxWindow *parent,
                      const wxString& message,
                      const wxString& caption = wxASCII_STR(wxGetPasswordFromUserPromptStr),
                      const wxString& value = wxEmptyString,
                      long style = wxTextEntryDialogStyle,
                      const wxPoint& pos = wxDefaultPosition)
    {
        Create(parent, message, caption, value, style, pos);
    }

    bool Create(wxWindow *parent,
                const wxString& message,
                const wxString& caption = wxASCII_STR(wxGetPasswordFromUserPromptStr),
                const wxString& value = wxEmptyString,
                long style = wxTextEntryDialogStyle,
                const wxPoint& pos = wxDefaultPosition);


private:
    wxDECLARE_DYNAMIC_CLASS(wxPasswordEntryDialog);
    wxDECLARE_NO_COPY_CLASS(wxPasswordEntryDialog);
};

// ----------------------------------------------------------------------------
// function to get a string from user
// ----------------------------------------------------------------------------

WXDLLIMPEXP_CORE wxString
    wxGetTextFromUser(const wxString& message,
                    const wxString& caption = wxASCII_STR(wxGetTextFromUserPromptStr),
                    const wxString& default_value = wxEmptyString,
                    wxWindow *parent = NULL,
                    wxCoord x = wxDefaultCoord,
                    wxCoord y = wxDefaultCoord,
                    bool centre = true);

WXDLLIMPEXP_CORE wxString
    wxGetPasswordFromUser(const wxString& message,
                        const wxString& caption = wxASCII_STR(wxGetPasswordFromUserPromptStr),
                        const wxString& default_value = wxEmptyString,
                        wxWindow *parent = NULL,
                        wxCoord x = wxDefaultCoord,
                        wxCoord y = wxDefaultCoord,
                        bool centre = true);

#endif
    // wxUSE_TEXTDLG
#endif // _WX_TEXTDLGG_H_
