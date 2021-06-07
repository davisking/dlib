/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/srchctrl.h
// Purpose:     wxSearchCtrl class - native
// Author:      Kettab Ali
// Created:     2019-12-23
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTK_SEARCHCTRL_H_
#define _WX_GTK_SEARCHCTRL_H_

#if wxUSE_SEARCHCTRL

// ----------------------------------------------------------------------------
// wxSearchCtrl
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxSearchCtrl : public wxSearchCtrlBase
{
public:
    // creation
    // --------

    wxSearchCtrl() : wxSearchCtrlBase()
    {
        Init();
    }

    wxSearchCtrl(wxWindow *parent, wxWindowID id,
               const wxString& value = wxEmptyString,
               const wxPoint& pos = wxDefaultPosition,
               const wxSize& size = wxDefaultSize,
               long style = 0,
               const wxValidator& validator = wxDefaultValidator,
               const wxString& name = wxASCII_STR(wxSearchCtrlNameStr))
        : wxSearchCtrlBase()
    {
        Init();
        Create(parent, id, value, pos, size, style, validator, name);
    }

    virtual ~wxSearchCtrl();

    bool Create(wxWindow *parent, wxWindowID id,
                const wxString& value = wxEmptyString,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize,
                long style = 0,
                const wxValidator& validator = wxDefaultValidator,
                const wxString& name = wxASCII_STR(wxSearchCtrlNameStr));

#if wxUSE_MENUS
    // get/set search button menu
    // --------------------------
    virtual void SetMenu( wxMenu* menu ) wxOVERRIDE;
    virtual wxMenu* GetMenu() wxOVERRIDE;
#endif // wxUSE_MENUS

    // get/set search options
    // ----------------------
    virtual void ShowSearchButton( bool show ) wxOVERRIDE;
    virtual bool IsSearchButtonVisible() const wxOVERRIDE;

    virtual void ShowCancelButton( bool show ) wxOVERRIDE;
    virtual bool IsCancelButtonVisible() const wxOVERRIDE;

    virtual void SetDescriptiveText(const wxString& text) wxOVERRIDE;
    virtual wxString GetDescriptiveText() const wxOVERRIDE;

    virtual void Clear() wxOVERRIDE;

#if wxUSE_MENUS
    void PopupSearchMenu();
#endif // wxUSE_MENUS

private:
    // From wxTextEntry:
    virtual GtkEditable *GetEditable() const wxOVERRIDE;


    void Init();

    void GTKCreateSearchEntryWidget();

    // Event handlers
    void OnChar(wxKeyEvent& event);
    void OnText(wxCommandEvent& event);
    void OnTextEnter(wxCommandEvent& event);

    bool HasMenu() const
    {
#if wxUSE_MENUS
        return m_menu != NULL;
#else // !wxUSE_MENUS
        return false;
#endif // wxUSE_MENUS/!wxUSE_MENUS
    }

private:
    virtual GtkEntry *GetEntry() const wxOVERRIDE
        { return m_entry; }

    GtkEntry *m_entry;

#if wxUSE_MENUS
    wxMenu *m_menu;
#endif // wxUSE_MENUS

    bool m_cancelButtonVisible;

    wxDECLARE_DYNAMIC_CLASS(wxSearchCtrl);
    wxDECLARE_EVENT_TABLE();
};

#endif // wxUSE_SEARCHCTRL

#endif // _WX_GTK_SEARCHCTRL_H_
