//////////////////////////////////////////////////////////////////////////////
// Name:        wx/aui/tabart.h
// Purpose:     wxaui: wx advanced user interface - notebook
// Author:      Benjamin I. Williams
// Modified by: Jens Lody (extracted from wx/aui/auibook.h)
// Created:     2012-03-21
// Copyright:   (C) Copyright 2006, Kirix Corporation, All Rights Reserved.
// Licence:     wxWindows Library Licence, Version 3.1
///////////////////////////////////////////////////////////////////////////////


#ifndef _WX_AUI_TABART_H_
#define _WX_AUI_TABART_H_

// ----------------------------------------------------------------------------
// headers
// ----------------------------------------------------------------------------

#include "wx/defs.h"

#if wxUSE_AUI

#include "wx/colour.h"
#include "wx/gdicmn.h"
#include "wx/font.h"
#include "wx/pen.h"
#include "wx/brush.h"
#include "wx/bitmap.h"


class wxAuiNotebookPage;
class wxAuiNotebookPageArray;
class wxWindow;
class wxDC;


// tab art class

class WXDLLIMPEXP_AUI wxAuiTabArt
{
public:

    wxAuiTabArt() { }
    virtual ~wxAuiTabArt() { }

    virtual wxAuiTabArt* Clone() = 0;
    virtual void SetFlags(unsigned int flags) = 0;

    virtual void SetSizingInfo(const wxSize& tabCtrlSize,
                               size_t tabCount) = 0;

    virtual void SetNormalFont(const wxFont& font) = 0;
    virtual void SetSelectedFont(const wxFont& font) = 0;
    virtual void SetMeasuringFont(const wxFont& font) = 0;
    virtual void SetColour(const wxColour& colour) = 0;
    virtual void SetActiveColour(const wxColour& colour) = 0;

    virtual void DrawBorder(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& rect) = 0;

    virtual void DrawBackground(
                         wxDC& dc,
                         wxWindow* wnd,
                         const wxRect& rect) = 0;

    virtual void DrawTab(wxDC& dc,
                         wxWindow* wnd,
                         const wxAuiNotebookPage& pane,
                         const wxRect& inRect,
                         int closeButtonState,
                         wxRect* outTabRect,
                         wxRect* outButtonRect,
                         int* xExtent) = 0;

    virtual void DrawButton(
                         wxDC& dc,
                         wxWindow* wnd,
                         const wxRect& inRect,
                         int bitmapId,
                         int buttonState,
                         int orientation,
                         wxRect* outRect) = 0;

    virtual wxSize GetTabSize(
                         wxDC& dc,
                         wxWindow* wnd,
                         const wxString& caption,
                         const wxBitmap& bitmap,
                         bool active,
                         int closeButtonState,
                         int* xExtent) = 0;

    virtual int ShowDropDown(
                         wxWindow* wnd,
                         const wxAuiNotebookPageArray& items,
                         int activeIdx) = 0;

    virtual int GetIndentSize() = 0;

    virtual int GetBorderWidth(
                         wxWindow* wnd) = 0;

    virtual int GetAdditionalBorderSpace(
                         wxWindow* wnd) = 0;

    virtual int GetBestTabCtrlSize(
                         wxWindow* wnd,
                         const wxAuiNotebookPageArray& pages,
                         const wxSize& requiredBmpSize) = 0;

    // Provide opportunity for subclasses to recalculate colours
    virtual void UpdateColoursFromSystem() {}
};


class WXDLLIMPEXP_AUI wxAuiGenericTabArt : public wxAuiTabArt
{

public:

    wxAuiGenericTabArt();
    virtual ~wxAuiGenericTabArt();

    wxAuiTabArt* Clone() wxOVERRIDE;
    void SetFlags(unsigned int flags) wxOVERRIDE;
    void SetSizingInfo(const wxSize& tabCtrlSize,
                       size_t tabCount) wxOVERRIDE;

    void SetNormalFont(const wxFont& font) wxOVERRIDE;
    void SetSelectedFont(const wxFont& font) wxOVERRIDE;
    void SetMeasuringFont(const wxFont& font) wxOVERRIDE;
    void SetColour(const wxColour& colour) wxOVERRIDE;
    void SetActiveColour(const wxColour& colour) wxOVERRIDE;

    void DrawBorder(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& rect) wxOVERRIDE;

    void DrawBackground(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& rect) wxOVERRIDE;

    void DrawTab(wxDC& dc,
                 wxWindow* wnd,
                 const wxAuiNotebookPage& pane,
                 const wxRect& inRect,
                 int closeButtonState,
                 wxRect* outTabRect,
                 wxRect* outButtonRect,
                 int* xExtent) wxOVERRIDE;

    void DrawButton(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& inRect,
                 int bitmapId,
                 int buttonState,
                 int orientation,
                 wxRect* outRect) wxOVERRIDE;

    int GetIndentSize() wxOVERRIDE;

    int GetBorderWidth(
                 wxWindow* wnd) wxOVERRIDE;

    int GetAdditionalBorderSpace(
                 wxWindow* wnd) wxOVERRIDE;

    wxSize GetTabSize(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxString& caption,
                 const wxBitmap& bitmap,
                 bool active,
                 int closeButtonState,
                 int* xExtent) wxOVERRIDE;

    int ShowDropDown(
                 wxWindow* wnd,
                 const wxAuiNotebookPageArray& items,
                 int activeIdx) wxOVERRIDE;

    int GetBestTabCtrlSize(wxWindow* wnd,
                 const wxAuiNotebookPageArray& pages,
                 const wxSize& requiredBmpSize) wxOVERRIDE;

    // Provide opportunity for subclasses to recalculate colours
    virtual void UpdateColoursFromSystem() wxOVERRIDE;

protected:

    wxFont m_normalFont;
    wxFont m_selectedFont;
    wxFont m_measuringFont;
    wxColour m_baseColour;
    wxPen m_baseColourPen;
    wxPen m_borderPen;
    wxBrush m_baseColourBrush;
    wxColour m_activeColour;
    wxBitmap m_activeCloseBmp;
    wxBitmap m_disabledCloseBmp;
    wxBitmap m_activeLeftBmp;
    wxBitmap m_disabledLeftBmp;
    wxBitmap m_activeRightBmp;
    wxBitmap m_disabledRightBmp;
    wxBitmap m_activeWindowListBmp;
    wxBitmap m_disabledWindowListBmp;

    int m_fixedTabWidth;
    int m_tabCtrlHeight;
    unsigned int m_flags;
};


class WXDLLIMPEXP_AUI wxAuiSimpleTabArt : public wxAuiTabArt
{

public:

    wxAuiSimpleTabArt();
    virtual ~wxAuiSimpleTabArt();

    wxAuiTabArt* Clone() wxOVERRIDE;
    void SetFlags(unsigned int flags) wxOVERRIDE;

    void SetSizingInfo(const wxSize& tabCtrlSize,
                       size_t tabCount) wxOVERRIDE;

    void SetNormalFont(const wxFont& font) wxOVERRIDE;
    void SetSelectedFont(const wxFont& font) wxOVERRIDE;
    void SetMeasuringFont(const wxFont& font) wxOVERRIDE;
    void SetColour(const wxColour& colour) wxOVERRIDE;
    void SetActiveColour(const wxColour& colour) wxOVERRIDE;

    void DrawBorder(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& rect) wxOVERRIDE;

    void DrawBackground(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& rect) wxOVERRIDE;

    void DrawTab(wxDC& dc,
                 wxWindow* wnd,
                 const wxAuiNotebookPage& pane,
                 const wxRect& inRect,
                 int closeButtonState,
                 wxRect* outTabRect,
                 wxRect* outButtonRect,
                 int* xExtent) wxOVERRIDE;

    void DrawButton(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxRect& inRect,
                 int bitmapId,
                 int buttonState,
                 int orientation,
                 wxRect* outRect) wxOVERRIDE;

    int GetIndentSize() wxOVERRIDE;

    int GetBorderWidth(
                 wxWindow* wnd) wxOVERRIDE;

    int GetAdditionalBorderSpace(
                 wxWindow* wnd) wxOVERRIDE;

    wxSize GetTabSize(
                 wxDC& dc,
                 wxWindow* wnd,
                 const wxString& caption,
                 const wxBitmap& bitmap,
                 bool active,
                 int closeButtonState,
                 int* xExtent) wxOVERRIDE;

    int ShowDropDown(
                 wxWindow* wnd,
                 const wxAuiNotebookPageArray& items,
                 int activeIdx) wxOVERRIDE;

    int GetBestTabCtrlSize(wxWindow* wnd,
                 const wxAuiNotebookPageArray& pages,
                 const wxSize& requiredBmpSize) wxOVERRIDE;

protected:

    wxFont m_normalFont;
    wxFont m_selectedFont;
    wxFont m_measuringFont;
    wxPen m_normalBkPen;
    wxPen m_selectedBkPen;
    wxBrush m_normalBkBrush;
    wxBrush m_selectedBkBrush;
    wxBrush m_bkBrush;
    wxBitmap m_activeCloseBmp;
    wxBitmap m_disabledCloseBmp;
    wxBitmap m_activeLeftBmp;
    wxBitmap m_disabledLeftBmp;
    wxBitmap m_activeRightBmp;
    wxBitmap m_disabledRightBmp;
    wxBitmap m_activeWindowListBmp;
    wxBitmap m_disabledWindowListBmp;

    int m_fixedTabWidth;
    unsigned int m_flags;
};

#ifndef __WXUNIVERSAL__
    #if defined(__WXGTK20__) && !defined(__WXGTK3__)
        #define wxHAS_NATIVE_TABART
        #include "wx/aui/tabartgtk.h"
        #define wxAuiDefaultTabArt wxAuiGtkTabArt
    #elif defined(__WXMSW__) && wxUSE_UXTHEME
        #define wxHAS_NATIVE_TABART
        #include "wx/aui/tabartmsw.h"
        #define wxAuiDefaultTabArt wxAuiMSWTabArt
    #endif
#endif // !__WXUNIVERSAL__

#ifndef wxHAS_NATIVE_TABART
    #define wxAuiDefaultTabArt wxAuiGenericTabArt
#endif

#endif  // wxUSE_AUI

#endif  // _WX_AUI_TABART_H_
