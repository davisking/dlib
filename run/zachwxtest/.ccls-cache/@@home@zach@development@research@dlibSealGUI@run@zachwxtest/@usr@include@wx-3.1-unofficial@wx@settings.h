/////////////////////////////////////////////////////////////////////////////
// Name:        wx/settings.h
// Purpose:     wxSystemSettings class
// Author:      Julian Smart
// Modified by:
// Created:     01/02/97
// Copyright:   (c) Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_SETTINGS_H_BASE_
#define _WX_SETTINGS_H_BASE_

#include "wx/colour.h"
#include "wx/font.h"

class WXDLLIMPEXP_FWD_CORE wxWindow;

// possible values for wxSystemSettings::GetFont() parameter
//
// NB: wxMSW assumes that they have the same values as the parameters of
//     Windows GetStockObject() API, don't change the values!
enum wxSystemFont
{
    wxSYS_OEM_FIXED_FONT = 10,
    wxSYS_ANSI_FIXED_FONT,
    wxSYS_ANSI_VAR_FONT,
    wxSYS_SYSTEM_FONT,
    wxSYS_DEVICE_DEFAULT_FONT,

    // don't use: this is here just to make the values of enum elements
    // coincide with the corresponding MSW constants
    wxSYS_DEFAULT_PALETTE,

    // don't use: MSDN says that this is a stock object provided only
    // for compatibility with 16-bit Windows versions earlier than 3.0!
    wxSYS_SYSTEM_FIXED_FONT,

    wxSYS_DEFAULT_GUI_FONT,

    // this was just a temporary aberration, do not use it any more
    wxSYS_ICONTITLE_FONT = wxSYS_DEFAULT_GUI_FONT
};

// possible values for wxSystemSettings::GetColour() parameter
//
// NB: wxMSW assumes that they have the same values as the parameters of
//     Windows GetSysColor() API, don't change the values!
enum wxSystemColour
{
    wxSYS_COLOUR_SCROLLBAR,
    wxSYS_COLOUR_DESKTOP,
    wxSYS_COLOUR_ACTIVECAPTION,
    wxSYS_COLOUR_INACTIVECAPTION,
    wxSYS_COLOUR_MENU,
    wxSYS_COLOUR_WINDOW,
    wxSYS_COLOUR_WINDOWFRAME,
    wxSYS_COLOUR_MENUTEXT,
    wxSYS_COLOUR_WINDOWTEXT,
    wxSYS_COLOUR_CAPTIONTEXT,
    wxSYS_COLOUR_ACTIVEBORDER,
    wxSYS_COLOUR_INACTIVEBORDER,
    wxSYS_COLOUR_APPWORKSPACE,
    wxSYS_COLOUR_HIGHLIGHT,
    wxSYS_COLOUR_HIGHLIGHTTEXT,
    wxSYS_COLOUR_BTNFACE,
    wxSYS_COLOUR_BTNSHADOW,
    wxSYS_COLOUR_GRAYTEXT,
    wxSYS_COLOUR_BTNTEXT,
    wxSYS_COLOUR_INACTIVECAPTIONTEXT,
    wxSYS_COLOUR_BTNHIGHLIGHT,
    wxSYS_COLOUR_3DDKSHADOW,
    wxSYS_COLOUR_3DLIGHT,
    wxSYS_COLOUR_INFOTEXT,
    wxSYS_COLOUR_INFOBK,
    wxSYS_COLOUR_LISTBOX,
    wxSYS_COLOUR_HOTLIGHT,
    wxSYS_COLOUR_GRADIENTACTIVECAPTION,
    wxSYS_COLOUR_GRADIENTINACTIVECAPTION,
    wxSYS_COLOUR_MENUHILIGHT,
    wxSYS_COLOUR_MENUBAR,
    wxSYS_COLOUR_LISTBOXTEXT,
    wxSYS_COLOUR_LISTBOXHIGHLIGHTTEXT,

    wxSYS_COLOUR_MAX,

    // synonyms
    wxSYS_COLOUR_BACKGROUND = wxSYS_COLOUR_DESKTOP,
    wxSYS_COLOUR_3DFACE = wxSYS_COLOUR_BTNFACE,
    wxSYS_COLOUR_3DSHADOW = wxSYS_COLOUR_BTNSHADOW,
    wxSYS_COLOUR_BTNHILIGHT = wxSYS_COLOUR_BTNHIGHLIGHT,
    wxSYS_COLOUR_3DHIGHLIGHT = wxSYS_COLOUR_BTNHIGHLIGHT,
    wxSYS_COLOUR_3DHILIGHT = wxSYS_COLOUR_BTNHIGHLIGHT,
    wxSYS_COLOUR_FRAMEBK = wxSYS_COLOUR_BTNFACE
};

// possible values for wxSystemSettings::GetMetric() index parameter
//
// NB: update the conversion table in msw/settings.cpp if you change the values
//     of the elements of this enum
enum wxSystemMetric
{
    wxSYS_MOUSE_BUTTONS = 1,
    wxSYS_BORDER_X,
    wxSYS_BORDER_Y,
    wxSYS_CURSOR_X,
    wxSYS_CURSOR_Y,
    wxSYS_DCLICK_X,
    wxSYS_DCLICK_Y,
    wxSYS_DRAG_X,
    wxSYS_DRAG_Y,
    wxSYS_EDGE_X,
    wxSYS_EDGE_Y,
    wxSYS_HSCROLL_ARROW_X,
    wxSYS_HSCROLL_ARROW_Y,
    wxSYS_HTHUMB_X,
    wxSYS_ICON_X,
    wxSYS_ICON_Y,
    wxSYS_ICONSPACING_X,
    wxSYS_ICONSPACING_Y,
    wxSYS_WINDOWMIN_X,
    wxSYS_WINDOWMIN_Y,
    wxSYS_SCREEN_X,
    wxSYS_SCREEN_Y,
    wxSYS_FRAMESIZE_X,
    wxSYS_FRAMESIZE_Y,
    wxSYS_SMALLICON_X,
    wxSYS_SMALLICON_Y,
    wxSYS_HSCROLL_Y,
    wxSYS_VSCROLL_X,
    wxSYS_VSCROLL_ARROW_X,
    wxSYS_VSCROLL_ARROW_Y,
    wxSYS_VTHUMB_Y,
    wxSYS_CAPTION_Y,
    wxSYS_MENU_Y,
    wxSYS_NETWORK_PRESENT,
    wxSYS_PENWINDOWS_PRESENT,
    wxSYS_SHOW_SOUNDS,
    wxSYS_SWAP_BUTTONS,
    wxSYS_DCLICK_MSEC,
    wxSYS_CARET_ON_MSEC,
    wxSYS_CARET_OFF_MSEC,
    wxSYS_CARET_TIMEOUT_MSEC
};

// possible values for wxSystemSettings::HasFeature() parameter
enum wxSystemFeature
{
    wxSYS_CAN_DRAW_FRAME_DECORATIONS = 1,
    wxSYS_CAN_ICONIZE_FRAME,
    wxSYS_TABLET_PRESENT
};

// values for different screen designs
enum wxSystemScreenType
{
    wxSYS_SCREEN_NONE = 0,  //   not yet defined

    wxSYS_SCREEN_TINY,      //   <
    wxSYS_SCREEN_PDA,       //   >= 320x240
    wxSYS_SCREEN_SMALL,     //   >= 640x480
    wxSYS_SCREEN_DESKTOP    //   >= 800x600
};

// ----------------------------------------------------------------------------
// wxSystemAppearance: describes the global appearance used for the UI
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxSystemAppearance
{
public:
    // Return the name if available or empty string otherwise.
    wxString GetName() const;

    // Return true if the current system there is explicitly recognized as
    // being a dark theme or if the default window background is dark.
    bool IsDark() const;

    // Return true if the background is darker than foreground. This is used by
    // IsDark() if there is no platform-specific way to determine whether a
    // dark mode is being used.
    bool IsUsingDarkBackground() const;

private:
    friend class wxSystemSettingsNative;

    // Ctor is private, even though it's trivial, because objects of this type
    // are only supposed to be created by wxSystemSettingsNative.
    wxSystemAppearance() { }

    // Currently this class doesn't have any internal state because the only
    // available implementation doesn't need it. If we do need it later, we
    // could add some "wxSystemAppearanceImpl* const m_impl" here, which we'd
    // forward our public functions to (we'd also need to add the copy ctor and
    // dtor to clone/free it).
};

// ----------------------------------------------------------------------------
// wxSystemSettingsNative: defines the API for wxSystemSettings class
// ----------------------------------------------------------------------------

// this is a namespace rather than a class: it has only non virtual static
// functions
//
// also note that the methods are implemented in the platform-specific source
// files (i.e. this is not a real base class as we can't override its virtual
// functions because it doesn't have any)

class WXDLLIMPEXP_CORE wxSystemSettingsNative
{
public:
    // get a standard system colour
    static wxColour GetColour(wxSystemColour index);

    // get a standard system font
    static wxFont GetFont(wxSystemFont index);

    // get a system-dependent metric
    static int GetMetric(wxSystemMetric index, const wxWindow* win = NULL);

    // get the object describing the current system appearance
    static wxSystemAppearance GetAppearance();

    // return true if the port has certain feature
    static bool HasFeature(wxSystemFeature index);
};

// ----------------------------------------------------------------------------
// include the declaration of the real platform-dependent class
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxSystemSettings : public wxSystemSettingsNative
{
public:
#ifdef __WXUNIVERSAL__
    // in wxUniversal we want to use the theme standard colours instead of the
    // system ones, otherwise wxSystemSettings is just the same as
    // wxSystemSettingsNative
    static wxColour GetColour(wxSystemColour index);

    // some metrics are toolkit-dependent and provided by wxUniv, some are
    // lowlevel
    static int GetMetric(wxSystemMetric index, const wxWindow* win = NULL);
#endif // __WXUNIVERSAL__

    // Get system screen design (desktop, pda, ..) used for
    // laying out various dialogs.
    static wxSystemScreenType GetScreenType();

    // Override default.
    static void SetScreenType( wxSystemScreenType screen );

    // Value
    static wxSystemScreenType ms_screen;

};

#endif
    // _WX_SETTINGS_H_BASE_

