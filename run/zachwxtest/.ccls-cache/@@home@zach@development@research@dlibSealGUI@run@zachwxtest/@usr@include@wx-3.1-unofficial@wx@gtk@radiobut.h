/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/radiobut.h
// Purpose:
// Author:      Robert Roebling
// Copyright:   (c) 1998 Robert Roebling
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTK_RADIOBUT_H_
#define _WX_GTK_RADIOBUT_H_

//-----------------------------------------------------------------------------
// wxRadioButton
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxRadioButton: public wxRadioButtonBase
{
public:
    wxRadioButton() { }
    wxRadioButton( wxWindow *parent,
                   wxWindowID id,
                   const wxString& label,
                   const wxPoint& pos = wxDefaultPosition,
                   const wxSize& size = wxDefaultSize,
                   long style = 0,
                   const wxValidator& validator = wxDefaultValidator,
                   const wxString& name = wxASCII_STR(wxRadioButtonNameStr) )
    {
        Create( parent, id, label, pos, size, style, validator, name );
    }

    bool Create( wxWindow *parent,
                 wxWindowID id,
                 const wxString& label,
                 const wxPoint& pos = wxDefaultPosition,
                 const wxSize& size = wxDefaultSize,
                 long style = 0,
                 const wxValidator& validator = wxDefaultValidator,
                 const wxString& name = wxASCII_STR(wxRadioButtonNameStr) );

    virtual void SetLabel(const wxString& label) wxOVERRIDE;
    virtual void SetValue(bool val) wxOVERRIDE;
    virtual bool GetValue() const wxOVERRIDE;

    static wxVisualAttributes
    GetClassDefaultAttributes(wxWindowVariant variant = wxWINDOW_VARIANT_NORMAL);

protected:
    virtual wxBorder GetDefaultBorder() const wxOVERRIDE { return wxBORDER_NONE; }

    virtual void DoApplyWidgetStyle(GtkRcStyle *style) wxOVERRIDE;
    virtual GdkWindow *GTKGetWindow(wxArrayGdkWindows& windows) const wxOVERRIDE;

    virtual void DoEnable(bool enable) wxOVERRIDE;

private:
    typedef wxControl base_type;

    wxDECLARE_DYNAMIC_CLASS(wxRadioButton);
};

#endif // _WX_GTK_RADIOBUT_H_
