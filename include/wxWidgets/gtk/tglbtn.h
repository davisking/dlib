/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/tglbtn.h
// Purpose:     Declaration of the wxToggleButton class, which implements a
//              toggle button under wxGTK.
// Author:      John Norris, minor changes by Axel Schlueter
// Modified by:
// Created:     08.02.01
// Copyright:   (c) 2000 Johnny C. Norris II
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTK_TOGGLEBUTTON_H_
#define _WX_GTK_TOGGLEBUTTON_H_

#include "wx/bitmap.h"

//-----------------------------------------------------------------------------
// wxToggleButton
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxToggleButton: public wxToggleButtonBase
{
public:
    // construction/destruction
    wxToggleButton() {}
    wxToggleButton(wxWindow *parent,
                   wxWindowID id,
                   const wxString& label,
                   const wxPoint& pos = wxDefaultPosition,
                   const wxSize& size = wxDefaultSize,
                   long style = 0,
                   const wxValidator& validator = wxDefaultValidator,
                   const wxString& name = wxASCII_STR(wxCheckBoxNameStr))
    {
        Create(parent, id, label, pos, size, style, validator, name);
    }

    // Create the control
    bool Create(wxWindow *parent,
                wxWindowID id,
                const wxString& label,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize, long style = 0,
                const wxValidator& validator = wxDefaultValidator,
                const wxString& name = wxASCII_STR(wxCheckBoxNameStr));

    // Get/set the value
    void SetValue(bool state) wxOVERRIDE;
    bool GetValue() const wxOVERRIDE;

    // Set the label
    void SetLabel(const wxString& label) wxOVERRIDE;

    static wxVisualAttributes
    GetClassDefaultAttributes(wxWindowVariant variant = wxWINDOW_VARIANT_NORMAL);

    void GTKDisableEvents();
    void GTKEnableEvents();

protected:
    virtual wxSize DoGetBestSize() const wxOVERRIDE;
    virtual void DoApplyWidgetStyle(GtkRcStyle *style) wxOVERRIDE;

#if wxUSE_MARKUP
    virtual bool DoSetLabelMarkup(const wxString& markup) wxOVERRIDE;
#endif // wxUSE_MARKUP

private:
    typedef wxToggleButtonBase base_type;

    // Return the GtkLabel used by this toggle button.
    GtkLabel *GTKGetLabel() const;

    wxDECLARE_DYNAMIC_CLASS(wxToggleButton);
};

//-----------------------------------------------------------------------------
// wxBitmapToggleButton
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxBitmapToggleButton: public wxToggleButton
{
public:
    // construction/destruction
    wxBitmapToggleButton() {}
    wxBitmapToggleButton(wxWindow *parent,
                   wxWindowID id,
                   const wxBitmap& label,
                   const wxPoint& pos = wxDefaultPosition,
                   const wxSize& size = wxDefaultSize,
                   long style = 0,
                   const wxValidator& validator = wxDefaultValidator,
                   const wxString& name = wxASCII_STR(wxCheckBoxNameStr))
    {
        Create(parent, id, label, pos, size, style, validator, name);
    }

    // Create the control
    bool Create(wxWindow *parent,
                wxWindowID id,
                const wxBitmap& label,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize, long style = 0,
                const wxValidator& validator = wxDefaultValidator,
                const wxString& name = wxASCII_STR(wxCheckBoxNameStr));

    // deprecated synonym for SetBitmapLabel()
    wxDEPRECATED_INLINE( void SetLabel(const wxBitmap& bitmap),
       SetBitmapLabel(bitmap); )
    // prevent virtual function hiding
    virtual void SetLabel(const wxString& label) wxOVERRIDE { wxToggleButton::SetLabel(label); }

private:
    typedef wxToggleButtonBase base_type;

    wxDECLARE_DYNAMIC_CLASS(wxBitmapToggleButton);
};

#endif // _WX_GTK_TOGGLEBUTTON_H_

