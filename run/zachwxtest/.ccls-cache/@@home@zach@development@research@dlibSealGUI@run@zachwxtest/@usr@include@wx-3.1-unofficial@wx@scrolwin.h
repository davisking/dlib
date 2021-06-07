/////////////////////////////////////////////////////////////////////////////
// Name:        wx/scrolwin.h
// Purpose:     wxScrolledWindow, wxScrolledControl and wxScrollHelper
// Author:      Vadim Zeitlin
// Modified by:
// Created:     30.08.00
// Copyright:   (c) 2000 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_SCROLWIN_H_BASE_
#define _WX_SCROLWIN_H_BASE_

#include "wx/control.h"
#include "wx/panel.h"

#ifdef  __WXOSX__
    #include "wx/scrolbar.h"
#endif

class WXDLLIMPEXP_FWD_CORE wxScrollHelperEvtHandler;
class WXDLLIMPEXP_FWD_BASE wxTimer;

// default scrolled window style: scroll in both directions
#define wxScrolledWindowStyle (wxHSCROLL | wxVSCROLL)

// values for the second argument of wxScrollHelper::ShowScrollbars()
enum wxScrollbarVisibility
{
    wxSHOW_SB_NEVER = -1,   // never show the scrollbar at all
    wxSHOW_SB_DEFAULT,      // show scrollbar only if it is needed
    wxSHOW_SB_ALWAYS        // always show scrollbar, even if not needed
};

// ----------------------------------------------------------------------------
// The hierarchy of scrolling classes is a bit complicated because we want to
// put as much functionality as possible in a mix-in class not deriving from
// wxWindow so that other classes could derive from the same base class on all
// platforms irrespectively of whether they are native controls (and hence
// don't use our scrolling) or not.
//
// So we have
//
//                          wxAnyScrollHelperBase
//                                   |
//                                   |
//                                  \|/
//                           wxScrollHelperBase
//                                   |
//                                   |
//                                  \|/
//      wxWindow               wxScrollHelper
//       |  \                   /        /
//       |   \                 /        /
//       |    _|             |_        /
//       |     wxScrolledWindow       /
//       |                           /
//      \|/                         /
//   wxControl                     /
//         \                      /
//          \                    /
//           _|                |_
//            wxScrolledControl
//
// ----------------------------------------------------------------------------

// This class allows reusing some of wxScrollHelperBase functionality in
// wxVarScrollHelperBase in wx/vscroll.h without duplicating its code.
class WXDLLIMPEXP_CORE wxAnyScrollHelperBase
{
public:
    explicit wxAnyScrollHelperBase(wxWindow* win);
    virtual ~wxAnyScrollHelperBase() {}

    // Disable use of keyboard keys for scrolling. By default cursor movement
    // keys (including Home, End, Page Up and Down) are used to scroll the
    // window appropriately. If the derived class uses these keys for something
    // else, e.g. changing the currently selected item, this function can be
    // used to disable this behaviour as it's not only not necessary then but
    // can actually be actively harmful if another object forwards a keyboard
    // event corresponding to one of the above keys to us using
    // ProcessWindowEvent() because the event will always be processed which
    // can be undesirable.
    void DisableKeyboardScrolling() { m_kbdScrollingEnabled = false; }

    // Override this function to draw the graphic (or just process EVT_PAINT)
    virtual void OnDraw(wxDC& WXUNUSED(dc)) { }

    // change the DC origin according to the scroll position.
    virtual void DoPrepareDC(wxDC& dc) = 0;

    // Simple accessor for the window that is really being scrolled.
    wxWindow *GetTargetWindow() const { return m_targetWindow; }


    // The methods called from the window event handlers.
    void HandleOnChar(wxKeyEvent& event);
    void HandleOnPaint(wxPaintEvent& event);

protected:
    // the window that receives the scroll events and the window to actually
    // scroll, respectively
    wxWindow *m_win,
             *m_targetWindow;

    // whether cursor keys should scroll the window
    bool m_kbdScrollingEnabled;
};

// This is the class containing the guts of (uniform) scrolling logic.
class WXDLLIMPEXP_CORE wxScrollHelperBase : public wxAnyScrollHelperBase
{
public:
    // ctor must be given the associated window
    wxScrollHelperBase(wxWindow *winToScroll);
    virtual ~wxScrollHelperBase();

    // configure the scrolling
    virtual void SetScrollbars(int pixelsPerUnitX, int pixelsPerUnitY,
                               int noUnitsX, int noUnitsY,
                               int xPos = 0, int yPos = 0,
                               bool noRefresh = false );

    // scroll to the given (in logical coords) position
    //
    // notice that for backwards compatibility reasons Scroll() is virtual as
    // the existing code could override it but new code should override
    // DoScroll() instead
    virtual void Scroll(int x, int y) { DoScroll(x, y); }
    virtual void Scroll(const wxPoint& pt) { DoScroll(pt.x, pt.y); }

    // get/set the page size for this orientation (wxVERTICAL/wxHORIZONTAL)
    int GetScrollPageSize(int orient) const;
    void SetScrollPageSize(int orient, int pageSize);

    // get the number of lines the window can scroll,
    // returns 0 if no scrollbars are there.
    int GetScrollLines( int orient ) const;

    // Set the x, y scrolling increments.
    void SetScrollRate( int xstep, int ystep );

    // get the size of one logical unit in physical ones
    void GetScrollPixelsPerUnit(int *pixelsPerUnitX, int *pixelsPerUnitY) const;

    // Set scrollbar visibility: it is possible to show scrollbar only if it is
    // needed (i.e. if our virtual size is greater than the current size of the
    // associated window), always (as wxALWAYS_SHOW_SB style does) or never (in
    // which case you should provide some other way to scroll the window as the
    // user wouldn't be able to do it at all)
    void ShowScrollbars(wxScrollbarVisibility horz, wxScrollbarVisibility vert)
    {
        DoShowScrollbars(horz, vert);
    }

    // Test whether the specified scrollbar is shown.
    virtual bool IsScrollbarShown(int orient) const = 0;

    // Enable/disable Windows scrolling in either direction. If true, wxWidgets
    // scrolls the canvas and only a bit of the canvas is invalidated; no
    // Clear() is necessary. If false, the whole canvas is invalidated and a
    // Clear() is necessary. Disable for when the scroll increment is used to
    // actually scroll a non-constant distance
    //
    // Notice that calling this method with a false argument doesn't disable
    // scrolling the window in this direction, it just changes the mechanism by
    // which it is implemented to not use wxWindow::ScrollWindow().
    virtual void EnableScrolling(bool x_scrolling, bool y_scrolling);

    // Get the view start
    void GetViewStart(int *x, int *y) const { DoGetViewStart(x, y); }

    wxPoint GetViewStart() const
    {
        wxPoint pt;
        DoGetViewStart(&pt.x, &pt.y);
        return pt;
    }

    // Set the scale factor, used in PrepareDC
    void SetScale(double xs, double ys) { m_scaleX = xs; m_scaleY = ys; }
    double GetScaleX() const { return m_scaleX; }
    double GetScaleY() const { return m_scaleY; }

    // translate between scrolled and unscrolled coordinates
    void CalcScrolledPosition(int x, int y, int *xx, int *yy) const
        {  DoCalcScrolledPosition(x, y, xx, yy); }
    wxPoint CalcScrolledPosition(const wxPoint& pt) const
    {
        wxPoint p2;
        DoCalcScrolledPosition(pt.x, pt.y, &p2.x, &p2.y);
        return p2;
    }

    void CalcUnscrolledPosition(int x, int y, int *xx, int *yy) const
        {  DoCalcUnscrolledPosition(x, y, xx, yy); }
    wxPoint CalcUnscrolledPosition(const wxPoint& pt) const
    {
        wxPoint p2;
        DoCalcUnscrolledPosition(pt.x, pt.y, &p2.x, &p2.y);
        return p2;
    }

    void DoCalcScrolledPosition(int x, int y, int *xx, int *yy) const;
    void DoCalcUnscrolledPosition(int x, int y, int *xx, int *yy) const;

    // Adjust the scrollbars
    virtual void AdjustScrollbars() = 0;

    // Calculate scroll increment
    int CalcScrollInc(wxScrollWinEvent& event);

    // Normally the wxScrolledWindow will scroll itself, but in some rare
    // occasions you might want it to scroll [part of] another window (e.g. a
    // child of it in order to scroll only a portion the area between the
    // scrollbars (spreadsheet: only cell area will move).
    void SetTargetWindow(wxWindow *target);

    void SetTargetRect(const wxRect& rect) { m_rectToScroll = rect; }
    wxRect GetTargetRect() const { return m_rectToScroll; }

    virtual void DoPrepareDC(wxDC& dc) wxOVERRIDE;

    // are we generating the autoscroll events?
    bool IsAutoScrolling() const { return m_timerAutoScroll != NULL; }

    // stop generating the scroll events when mouse is held outside the window
    void StopAutoScrolling();

    // this method can be overridden in a derived class to forbid sending the
    // auto scroll events - note that unlike StopAutoScrolling() it doesn't
    // stop the timer, so it will be called repeatedly and will typically
    // return different values depending on the current mouse position
    //
    // the base class version just returns true
    virtual bool SendAutoScrollEvents(wxScrollWinEvent& event) const;

    // the methods to be called from the window event handlers
    void HandleOnScroll(wxScrollWinEvent& event);
    void HandleOnSize(wxSizeEvent& event);
    void HandleOnMouseEnter(wxMouseEvent& event);
    void HandleOnMouseLeave(wxMouseEvent& event);
#if wxUSE_MOUSEWHEEL
    void HandleOnMouseWheel(wxMouseEvent& event);
#endif // wxUSE_MOUSEWHEEL
    void HandleOnChildFocus(wxChildFocusEvent& event);

#if WXWIN_COMPATIBILITY_2_8
    wxDEPRECATED(
        void OnScroll(wxScrollWinEvent& event) { HandleOnScroll(event); }
    )
#endif // WXWIN_COMPATIBILITY_2_8

protected:
    // get pointer to our scroll rect if we use it or NULL
    const wxRect *GetScrollRect() const
    {
        return m_rectToScroll.width != 0 ? &m_rectToScroll : NULL;
    }

    // get the size of the target window
    wxSize GetTargetSize() const
    {
        return m_rectToScroll.width != 0 ? m_rectToScroll.GetSize()
                                         : m_targetWindow->GetClientSize();
    }

    void GetTargetSize(int *w, int *h) const
    {
        wxSize size = GetTargetSize();
        if ( w )
            *w = size.x;
        if ( h )
            *h = size.y;
    }

    // implementation of public methods with the same name
    virtual void DoGetViewStart(int *x, int *y) const;
    virtual void DoScroll(int x, int y) = 0;
    virtual void DoShowScrollbars(wxScrollbarVisibility horz,
                                  wxScrollbarVisibility vert) = 0;

    // implementations of various wxWindow virtual methods which should be
    // forwarded to us (this can be done by WX_FORWARD_TO_SCROLL_HELPER())
    bool ScrollLayout();
    void ScrollDoSetVirtualSize(int x, int y);
    wxSize ScrollGetBestVirtualSize() const;

    // change just the target window (unlike SetWindow which changes m_win as
    // well)
    void DoSetTargetWindow(wxWindow *target);

    // delete the event handler we installed
    void DeleteEvtHandler();

    // this function should be overridden to return the size available for
    // m_targetWindow inside m_win of the given size
    //
    // the default implementation is only good for m_targetWindow == m_win
    // case, if we're scrolling a subwindow you must override this method
    virtual wxSize GetSizeAvailableForScrollTarget(const wxSize& size)
    {
        // returning just size from here is wrong but it was decided that it is
        // not wrong enough to break the existing code (which doesn't override
        // this recently added function at all) by adding this assert
        //
        // wxASSERT_MSG( m_targetWindow == m_win, "must be overridden" );

        return size;
    }

    // Can be overridden to return false if the child window shouldn't be
    // scrolled into view automatically when it gets focus, which is the
    // default behaviour.
    virtual bool ShouldScrollToChildOnFocus(wxWindow* child)
    {
#if defined(__WXOSX__) && wxUSE_SCROLLBAR
        if ( wxDynamicCast(child, wxScrollBar) )
            return false;
#else
        wxUnusedVar(child);
#endif

        return true;
    }


    double                m_scaleX;
    double                m_scaleY;

    wxRect                m_rectToScroll;

    wxTimer              *m_timerAutoScroll;

    // The number of pixels to scroll in horizontal and vertical directions
    // respectively.
    //
    // If 0, means that the scrolling in the given direction is disabled.
    int                   m_xScrollPixelsPerLine;
    int                   m_yScrollPixelsPerLine;
    int                   m_xScrollPosition;
    int                   m_yScrollPosition;
    int                   m_xScrollLines;
    int                   m_yScrollLines;
    int                   m_xScrollLinesPerPage;
    int                   m_yScrollLinesPerPage;

    bool                  m_xScrollingEnabled;
    bool                  m_yScrollingEnabled;

#if wxUSE_MOUSEWHEEL
    int m_wheelRotation;
#endif // wxUSE_MOUSEWHEEL

    wxScrollHelperEvtHandler *m_handler;

    wxDECLARE_NO_COPY_CLASS(wxScrollHelperBase);
};

// this macro can be used in a wxScrollHelper-derived class to forward wxWindow
// methods to corresponding wxScrollHelper methods
#define WX_FORWARD_TO_SCROLL_HELPER()                                         \
public:                                                                       \
    virtual void PrepareDC(wxDC& dc) wxOVERRIDE { DoPrepareDC(dc); }          \
    virtual bool Layout() wxOVERRIDE { return ScrollLayout(); }               \
    virtual bool CanScroll(int orient) const wxOVERRIDE                       \
        { return IsScrollbarShown(orient); }                                  \
    virtual void DoSetVirtualSize(int x, int y) wxOVERRIDE                    \
        { ScrollDoSetVirtualSize(x, y); }                                     \
    virtual wxSize GetBestVirtualSize() const wxOVERRIDE                      \
        { return ScrollGetBestVirtualSize(); }

// include the declaration of the real wxScrollHelper
#if defined(__WXGTK20__) && !defined(__WXUNIVERSAL__)
    #include "wx/gtk/scrolwin.h"
#elif defined(__WXGTK__) && !defined(__WXUNIVERSAL__)
    #include "wx/gtk1/scrolwin.h"
#else
    #define wxHAS_GENERIC_SCROLLWIN
    #include "wx/generic/scrolwin.h"
#endif

// ----------------------------------------------------------------------------
// wxScrolled<T>: a wxWindow which knows how to scroll
// ----------------------------------------------------------------------------

// helper class for wxScrolled<T> below
struct WXDLLIMPEXP_CORE wxScrolledT_Helper
{
    static wxSize FilterBestSize(const wxWindow *win,
                                 const wxScrollHelper *helper,
                                 const wxSize& origBest);
#ifdef __WXMSW__
    static WXLRESULT FilterMSWWindowProc(WXUINT nMsg, WXLRESULT origResult);
#endif
};

// Scrollable window base on window type T. This used to be wxScrolledWindow,
// but wxScrolledWindow includes wxControlContainer functionality and that's
// not always desirable.
template<class T>
bool wxCreateScrolled(T* self,
                      wxWindow *parent, wxWindowID winid,
                      const wxPoint& pos, const wxSize& size,
                      long style, const wxString& name)
{
    return self->Create(parent, winid, pos, size, style, name);
}

#if wxUSE_CONTROLS
// For wxControl we have to provide overloaded wxCreateScrolled()
// because wxControl::Create() has 7 parameters and therefore base
// template expecting 6-parameter T::Create() cannot be used.
inline bool wxCreateScrolled(wxControl* self,
                     wxWindow *parent, wxWindowID winid,
                     const wxPoint& pos, const wxSize& size,
                     long style, const wxString& name)
{
     return self->Create(parent, winid, pos, size, style, wxDefaultValidator, name);
}
#endif // wxUSE_CONTROLS

template<class T>
class wxScrolled : public T,
                   public wxScrollHelper,
                   private wxScrolledT_Helper
{
public:
    wxScrolled() : wxScrollHelper(this) { }
    wxScrolled(wxWindow *parent,
               wxWindowID winid = wxID_ANY,
               const wxPoint& pos = wxDefaultPosition,
               const wxSize& size = wxDefaultSize,
               long style = wxScrolledWindowStyle,
               const wxString& name = wxASCII_STR(wxPanelNameStr))
        : wxScrollHelper(this)
    {
        Create(parent, winid, pos, size, style, name);
    }

    bool Create(wxWindow *parent,
                wxWindowID winid,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize,
                long style = wxScrolledWindowStyle,
                const wxString& name = wxASCII_STR(wxPanelNameStr))
    {
        m_targetWindow = this;

#ifdef __WXMAC__
        this->MacSetClipChildren(true);
#endif

        // by default, we're scrollable in both directions (but if one of the
        // styles is specified explicitly, we shouldn't add the other one
        // automatically)
        if ( !(style & (wxHSCROLL | wxVSCROLL)) )
            style |= wxHSCROLL | wxVSCROLL;

        return wxCreateScrolled((T*)this, parent, winid, pos, size, style, name);
    }

#ifdef __WXMSW__
    // we need to return a special WM_GETDLGCODE value to process just the
    // arrows but let the other navigation characters through
    virtual WXLRESULT MSWWindowProc(WXUINT nMsg, WXWPARAM wParam, WXLPARAM lParam) wxOVERRIDE
    {
        return FilterMSWWindowProc(nMsg, T::MSWWindowProc(nMsg, wParam, lParam));
    }

    // Take into account the scroll origin.
    virtual void MSWAdjustBrushOrg(int* xOrg, int* yOrg) const wxOVERRIDE
    {
        CalcUnscrolledPosition(*xOrg, *yOrg, xOrg, yOrg);
    }
#endif // __WXMSW__

    WX_FORWARD_TO_SCROLL_HELPER()

protected:
    virtual wxSize DoGetBestSize() const wxOVERRIDE
    {
        return FilterBestSize(this, this, T::DoGetBestSize());
    }

private:
    wxDECLARE_NO_COPY_CLASS(wxScrolled);
};

// for compatibility with existing code, we provide wxScrolledWindow
// "typedef" for wxScrolled<wxPanel>. It's not a real typedef because we
// want wxScrolledWindow to show in wxRTTI information (the class is widely
// used and likelihood of its wxRTTI information being used too is high):
class WXDLLIMPEXP_CORE wxScrolledWindow : public wxScrolled<wxPanel>
{
public:
    wxScrolledWindow() : wxScrolled<wxPanel>() {}
    wxScrolledWindow(wxWindow *parent,
                     wxWindowID winid = wxID_ANY,
                     const wxPoint& pos = wxDefaultPosition,
                     const wxSize& size = wxDefaultSize,
                     long style = wxScrolledWindowStyle,
                     const wxString& name = wxASCII_STR(wxPanelNameStr))
        : wxScrolled<wxPanel>(parent, winid, pos, size, style, name) {}

    wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxScrolledWindow);
};

typedef wxScrolled<wxWindow> wxScrolledCanvas;

#endif // _WX_SCROLWIN_H_BASE_
