///////////////////////////////////////////////////////////////////////////////
// Name:        wx/itemattr.h
// Purpose:     wxItemAttr class declaration
// Author:      Vadim Zeitlin
// Created:     2016-04-16 (extracted from wx/listctrl.h)
// Copyright:   (c) 2016 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_ITEMATTR_H_
#define _WX_ITEMATTR_H_

// ----------------------------------------------------------------------------
// wxItemAttr: a structure containing the visual attributes of an item
// ----------------------------------------------------------------------------

class wxItemAttr
{
public:
    // ctors
    wxItemAttr() { }
    wxItemAttr(const wxColour& colText,
               const wxColour& colBack,
               const wxFont& font)
        : m_colText(colText), m_colBack(colBack), m_font(font)
    {
    }

    // default copy ctor, assignment operator and dtor are ok

    bool operator==(const wxItemAttr& other) const
    {
        return m_colText == other.m_colText &&
               m_colBack == other.m_colBack &&
               m_font == other.m_font;
    }

    bool operator!=(const wxItemAttr& other) const
    {
        return !(*this == other);
    }

    // setters
    void SetTextColour(const wxColour& colText) { m_colText = colText; }
    void SetBackgroundColour(const wxColour& colBack) { m_colBack = colBack; }
    void SetFont(const wxFont& font) { m_font = font; }

    // accessors
    bool HasTextColour() const { return m_colText.IsOk(); }
    bool HasBackgroundColour() const { return m_colBack.IsOk(); }
    bool HasFont() const { return m_font.IsOk(); }

    bool HasColours() const { return HasTextColour() || HasBackgroundColour(); }
    bool IsDefault() const { return !HasColours() && !HasFont(); }

    const wxColour& GetTextColour() const { return m_colText; }
    const wxColour& GetBackgroundColour() const { return m_colBack; }
    const wxFont& GetFont() const { return m_font; }


    // this is almost like assignment operator except it doesn't overwrite the
    // fields unset in the source attribute
    void AssignFrom(const wxItemAttr& source)
    {
        if ( source.HasTextColour() )
            SetTextColour(source.GetTextColour());
        if ( source.HasBackgroundColour() )
            SetBackgroundColour(source.GetBackgroundColour());
        if ( source.HasFont() )
            SetFont(source.GetFont());
    }

private:
    wxColour m_colText,
             m_colBack;
    wxFont   m_font;
};

#endif // _WX_ITEMATTR_H_
