/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/gauge.h
// Purpose:
// Author:      Robert Roebling
// Copyright:   (c) 1998 Robert Roebling
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTK_GAUGE_H_
#define _WX_GTK_GAUGE_H_

//-----------------------------------------------------------------------------
// wxGauge
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxGauge: public wxGaugeBase
{
public:
    wxGauge() { Init(); }

    wxGauge( wxWindow *parent,
             wxWindowID id,
             int range,
             const wxPoint& pos = wxDefaultPosition,
             const wxSize& size = wxDefaultSize,
             long style = wxGA_HORIZONTAL,
             const wxValidator& validator = wxDefaultValidator,
             const wxString& name = wxASCII_STR(wxGaugeNameStr) )
    {
        Init();

        Create(parent, id, range, pos, size, style, validator, name);
    }

    bool Create( wxWindow *parent,
                 wxWindowID id, int range,
                 const wxPoint& pos = wxDefaultPosition,
                 const wxSize& size = wxDefaultSize,
                 long style = wxGA_HORIZONTAL,
                 const wxValidator& validator = wxDefaultValidator,
                 const wxString& name = wxASCII_STR(wxGaugeNameStr) );

    // implement base class virtual methods
    void SetRange(int range) wxOVERRIDE;
    int GetRange() const wxOVERRIDE;

    void SetValue(int pos) wxOVERRIDE;
    int GetValue() const wxOVERRIDE;

    virtual void Pulse() wxOVERRIDE;

    static wxVisualAttributes
    GetClassDefaultAttributes(wxWindowVariant variant = wxWINDOW_VARIANT_NORMAL);

    virtual wxVisualAttributes GetDefaultAttributes() const wxOVERRIDE;

    // implementation
    // -------------

    // the max and current gauge values
    int m_rangeMax,
        m_gaugePos;

protected:
    // set the gauge value to the value of m_gaugePos
    void DoSetGauge();

private:
    void Init() { m_rangeMax = m_gaugePos = 0; }

    wxDECLARE_DYNAMIC_CLASS(wxGauge);
};

#endif
    // _WX_GTK_GAUGE_H_
