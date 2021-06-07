///////////////////////////////////////////////////////////////////////////////
// Name:        wx/peninfobase.h
// Purpose:     Declaration of wxPenInfoBase class and related constants
// Author:      Adrien Tétar, Vadim Zeitlin
// Created:     2017-09-10
// Copyright:   (c) 2017 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PENINFOBASE_H_
#define _WX_PENINFOBASE_H_

#include "wx/bitmap.h"
#include "wx/colour.h"
#include "wx/gdicmn.h"  // for wxDash

enum wxPenStyle
{
    wxPENSTYLE_INVALID = -1,

    wxPENSTYLE_SOLID = wxSOLID,
    wxPENSTYLE_DOT = wxDOT,
    wxPENSTYLE_LONG_DASH = wxLONG_DASH,
    wxPENSTYLE_SHORT_DASH = wxSHORT_DASH,
    wxPENSTYLE_DOT_DASH = wxDOT_DASH,
    wxPENSTYLE_USER_DASH = wxUSER_DASH,

    wxPENSTYLE_TRANSPARENT = wxTRANSPARENT,

    wxPENSTYLE_STIPPLE_MASK_OPAQUE = wxSTIPPLE_MASK_OPAQUE,
    wxPENSTYLE_STIPPLE_MASK = wxSTIPPLE_MASK,
    wxPENSTYLE_STIPPLE = wxSTIPPLE,

    wxPENSTYLE_BDIAGONAL_HATCH = wxHATCHSTYLE_BDIAGONAL,
    wxPENSTYLE_CROSSDIAG_HATCH = wxHATCHSTYLE_CROSSDIAG,
    wxPENSTYLE_FDIAGONAL_HATCH = wxHATCHSTYLE_FDIAGONAL,
    wxPENSTYLE_CROSS_HATCH = wxHATCHSTYLE_CROSS,
    wxPENSTYLE_HORIZONTAL_HATCH = wxHATCHSTYLE_HORIZONTAL,
    wxPENSTYLE_VERTICAL_HATCH = wxHATCHSTYLE_VERTICAL,
    wxPENSTYLE_FIRST_HATCH = wxHATCHSTYLE_FIRST,
    wxPENSTYLE_LAST_HATCH = wxHATCHSTYLE_LAST
};

enum wxPenJoin
{
    wxJOIN_INVALID = -1,

    wxJOIN_BEVEL = 120,
    wxJOIN_MITER,
    wxJOIN_ROUND
};

enum wxPenCap
{
    wxCAP_INVALID = -1,

    wxCAP_ROUND = 130,
    wxCAP_PROJECTING,
    wxCAP_BUTT
};

// ----------------------------------------------------------------------------
// wxPenInfoBase is a common base for wxPenInfo and wxGraphicsPenInfo
// ----------------------------------------------------------------------------

// This class uses CRTP, the template parameter is the derived class itself.
template <class T>
class wxPenInfoBase
{
public:
    // Setters for the various attributes. All of them return the object itself
    // so that the calls to them could be chained.

    T& Colour(const wxColour& colour)
        { m_colour = colour; return This(); }

    T& Style(wxPenStyle style)
        { m_style = style; return This(); }
    T& Stipple(const wxBitmap& stipple)
        { m_stipple = stipple; m_style = wxPENSTYLE_STIPPLE; return This(); }
    T& Dashes(int nb_dashes, const wxDash *dash)
        { m_nb_dashes = nb_dashes; m_dash = const_cast<wxDash*>(dash); return This(); }
    T& Join(wxPenJoin join)
        { m_join = join; return This(); }
    T& Cap(wxPenCap cap)
        { m_cap = cap; return This(); }

    // Accessors are mostly meant to be used by wxWidgets itself.

    wxColour GetColour() const { return m_colour; }
    wxBitmap GetStipple() const { return m_stipple; }
    wxPenStyle GetStyle() const { return m_style; }
    wxPenJoin GetJoin() const { return m_join; }
    wxPenCap GetCap() const { return m_cap; }
    int GetDashes(wxDash **ptr) const { *ptr = m_dash; return m_nb_dashes; }

    int GetDashCount() const { return m_nb_dashes; }
    wxDash* GetDash() const { return m_dash; }

    // Convenience

    bool IsTransparent() const { return m_style == wxPENSTYLE_TRANSPARENT; }

protected:
    wxPenInfoBase(const wxColour& colour, wxPenStyle style)
        : m_colour(colour)
    {
        m_nb_dashes = 0;
        m_dash = NULL;
        m_join = wxJOIN_ROUND;
        m_cap = wxCAP_ROUND;
        m_style = style;
    }

private:
    // Helper to return this object itself cast to its real type T.
    T& This() { return static_cast<T&>(*this); }

    wxColour m_colour;
    wxBitmap m_stipple;
    wxPenStyle m_style;
    wxPenJoin m_join;
    wxPenCap m_cap;

    int m_nb_dashes;
    wxDash* m_dash;
};

#endif // _WX_PENINFOBASE_H_
