/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/statbox.h
// Purpose:
// Author:      Robert Roebling
// Copyright:   (c) 1998 Robert Roebling
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTKSTATICBOX_H_
#define _WX_GTKSTATICBOX_H_

//-----------------------------------------------------------------------------
// wxStaticBox
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxStaticBox : public wxStaticBoxBase
{
public:
    wxStaticBox()
    {
    }

    wxStaticBox( wxWindow *parent,
                 wxWindowID id,
                 const wxString &label,
                 const wxPoint &pos = wxDefaultPosition,
                 const wxSize &size = wxDefaultSize,
                 long style = 0,
                 const wxString &name = wxASCII_STR(wxStaticBoxNameStr) )
    {
        Create( parent, id, label, pos, size, style, name );
    }

    wxStaticBox( wxWindow *parent,
                 wxWindowID id,
                 wxWindow* label,
                 const wxPoint &pos = wxDefaultPosition,
                 const wxSize &size = wxDefaultSize,
                 long style = 0,
                 const wxString &name = wxASCII_STR(wxStaticBoxNameStr) )
    {
        Create( parent, id, label, pos, size, style, name );
    }

    bool Create( wxWindow *parent,
                 wxWindowID id,
                 const wxString &label,
                 const wxPoint &pos = wxDefaultPosition,
                 const wxSize &size = wxDefaultSize,
                 long style = 0,
                 const wxString &name = wxASCII_STR(wxStaticBoxNameStr) )
    {
        return DoCreate( parent, id, &label, NULL, pos, size, style, name );
    }

    bool Create( wxWindow *parent,
                 wxWindowID id,
                 wxWindow* label,
                 const wxPoint &pos = wxDefaultPosition,
                 const wxSize &size = wxDefaultSize,
                 long style = 0,
                 const wxString &name = wxASCII_STR(wxStaticBoxNameStr) )
    {
        return DoCreate( parent, id, NULL, label, pos, size, style, name );
    }

    virtual void SetLabel( const wxString &label ) wxOVERRIDE;

    static wxVisualAttributes
    GetClassDefaultAttributes(wxWindowVariant variant = wxWINDOW_VARIANT_NORMAL);

    // implementation

    virtual bool GTKIsTransparentForMouse() const wxOVERRIDE { return true; }

    virtual void GetBordersForSizer(int *borderTop, int *borderOther) const wxOVERRIDE;

    virtual void AddChild( wxWindowBase *child ) wxOVERRIDE;

protected:
    // Common implementation of both Create() overloads: exactly one of
    // labelStr and labelWin parameters must be non-null.
    bool DoCreate(wxWindow *parent,
                  wxWindowID id,
                  const wxString* labelStr,
                  wxWindow* labelWin,
                  const wxPoint& pos,
                  const wxSize& size,
                  long style,
                  const wxString& name);

    virtual bool GTKWidgetNeedsMnemonic() const wxOVERRIDE;
    virtual void GTKWidgetDoSetMnemonic(GtkWidget* w) wxOVERRIDE;

    void DoApplyWidgetStyle(GtkRcStyle *style) wxOVERRIDE;

    wxDECLARE_DYNAMIC_CLASS(wxStaticBox);
};

// Indicate that we have the ctor overload taking wxWindow as label.
#define wxHAS_WINDOW_LABEL_IN_STATIC_BOX

#endif // _WX_GTKSTATICBOX_H_
