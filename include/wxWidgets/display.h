/////////////////////////////////////////////////////////////////////////////
// Name:        wx/display.h
// Purpose:     wxDisplay class
// Author:      Royce Mitchell III, Vadim Zeitlin
// Created:     06/21/02
// Copyright:   (c) 2002-2006 wxWidgets team
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_DISPLAY_H_BASE_
#define _WX_DISPLAY_H_BASE_

#include "wx/defs.h"
#include "wx/gdicmn.h"      // wxSize

// NB: no #if wxUSE_DISPLAY here, the display geometry part of this class (but
//     not the video mode stuff) is always available but if wxUSE_DISPLAY == 0
//     it becomes just a trivial wrapper around the old wxDisplayXXX() functions

#if wxUSE_DISPLAY
    #include "wx/dynarray.h"
    #include "wx/vidmode.h"

    WX_DECLARE_EXPORTED_OBJARRAY(wxVideoMode, wxArrayVideoModes);

    // default, uninitialized, video mode object
    extern WXDLLIMPEXP_DATA_CORE(const wxVideoMode) wxDefaultVideoMode;
#endif // wxUSE_DISPLAY

class WXDLLIMPEXP_FWD_CORE wxWindow;
class WXDLLIMPEXP_FWD_CORE wxPoint;
class WXDLLIMPEXP_FWD_CORE wxRect;
class WXDLLIMPEXP_FWD_BASE wxString;

class WXDLLIMPEXP_FWD_CORE wxDisplayFactory;
class WXDLLIMPEXP_FWD_CORE wxDisplayImpl;

// ----------------------------------------------------------------------------
// wxDisplay: represents a display/monitor attached to the system
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDisplay
{
public:
    // default ctor creates the object corresponding to the primary display
    wxDisplay();

    // initialize the object containing all information about the given
    // display
    //
    // the displays are numbered from 0 to GetCount() - 1
    explicit wxDisplay(unsigned n);

    // create display object corresponding to the display of the given window
    // or the default one if the window display couldn't be found
    explicit wxDisplay(const wxWindow* window);

    // dtor is not virtual as this is a concrete class not meant to be derived
    // from


    // return the number of available displays, valid parameters to
    // wxDisplay ctor are from 0 up to this number
    static unsigned GetCount();

    // find the display where the given point lies, return wxNOT_FOUND if
    // it doesn't belong to any display
    static int GetFromPoint(const wxPoint& pt);

    // find the display where the given window lies, return wxNOT_FOUND if it
    // is not shown at all
    static int GetFromWindow(const wxWindow *window);


    // return true if the object was initialized successfully
    bool IsOk() const { return m_impl != NULL; }

    // get the full display size
    wxRect GetGeometry() const;

    // get the client area of the display, i.e. without taskbars and such
    wxRect GetClientArea() const;

    // get the depth, i.e. number of bits per pixel (0 if unknown)
    int GetDepth() const;

    // get the resolution of this monitor in pixels per inch
    wxSize GetPPI() const;

    // get the default resolution for displays on this platform
    static int GetStdPPIValue()
    {
#ifdef __WXOSX__
        return 72;
#else
        return 96;
#endif
    }

    static wxSize GetStdPPI()
    {
        return wxSize(GetStdPPIValue(), GetStdPPIValue());
    }

    // get the scaling used by this display
    double GetScaleFactor() const;

    // name may be empty
    wxString GetName() const;

    // display 0 is usually the primary display
    bool IsPrimary() const;


#if wxUSE_DISPLAY
    // enumerate all video modes supported by this display matching the given
    // one (in the sense of wxVideoMode::Match())
    //
    // as any mode matches the default value of the argument and there is
    // always at least one video mode supported by display, the returned array
    // is only empty for the default value of the argument if this function is
    // not supported at all on this platform
    wxArrayVideoModes
        GetModes(const wxVideoMode& mode = wxDefaultVideoMode) const;

    // get current video mode
    wxVideoMode GetCurrentMode() const;

    // change current mode, return true if succeeded, false otherwise
    //
    // for the default value of the argument restores the video mode to default
    bool ChangeMode(const wxVideoMode& mode = wxDefaultVideoMode);

    // restore the default video mode (just a more readable synonym)
    void ResetMode() { (void)ChangeMode(); }
#endif // wxUSE_DISPLAY

    // If the implementation caches any information about the displays, calling
    // this function clears it -- this should be done e.g. after a display
    // [dis]connection.
    static void InvalidateCache();

private:
    // returns the factory used to implement our static methods and create new
    // displays
    static wxDisplayFactory& Factory();

    // creates the factory object, called by Factory() when it is called for
    // the first time and should return a pointer allocated with new (the
    // caller will delete it)
    //
    // this method must be implemented in platform-specific code if
    // wxUSE_DISPLAY == 1 (if it is 0 we provide the stub in common code)
    static wxDisplayFactory *CreateFactory();


    // the real implementation
    wxDisplayImpl *m_impl;


    wxDECLARE_NO_COPY_CLASS(wxDisplay);
};

#endif // _WX_DISPLAY_H_BASE_
