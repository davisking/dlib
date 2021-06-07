///////////////////////////////////////////////////////////////////////////////
// Name:        wx/apptrait.h
// Purpose:     declaration of wxAppTraits and derived classes
// Author:      Vadim Zeitlin
// Modified by:
// Created:     19.06.2003
// Copyright:   (c) 2003 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_APPTRAIT_H_
#define _WX_APPTRAIT_H_

#include "wx/string.h"
#include "wx/platinfo.h"

class WXDLLIMPEXP_FWD_BASE wxArrayString;
class WXDLLIMPEXP_FWD_BASE wxConfigBase;
class WXDLLIMPEXP_FWD_BASE wxEventLoopBase;
#if wxUSE_FONTMAP
    class WXDLLIMPEXP_FWD_CORE wxFontMapper;
#endif // wxUSE_FONTMAP
class WXDLLIMPEXP_FWD_BASE wxLog;
class WXDLLIMPEXP_FWD_BASE wxMessageOutput;
class WXDLLIMPEXP_FWD_BASE wxObject;
class WXDLLIMPEXP_FWD_CORE wxRendererNative;
class WXDLLIMPEXP_FWD_BASE wxStandardPaths;
class WXDLLIMPEXP_FWD_BASE wxString;
class WXDLLIMPEXP_FWD_BASE wxTimer;
class WXDLLIMPEXP_FWD_BASE wxTimerImpl;

class wxSocketManager;


// ----------------------------------------------------------------------------
// wxAppTraits: this class defines various configurable aspects of wxApp
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxAppTraitsBase
{
public:
    // needed since this class declares virtual members
    virtual ~wxAppTraitsBase() { }

    // hooks for working with the global objects, may be overridden by the user
    // ------------------------------------------------------------------------

#if wxUSE_CONFIG
    // create the default configuration object (base class version is
    // implemented in config.cpp and creates wxRegConfig for wxMSW and
    // wxFileConfig for all the other platforms)
    virtual wxConfigBase *CreateConfig();
#endif // wxUSE_CONFIG

#if wxUSE_LOG
    // create the default log target
    virtual wxLog *CreateLogTarget() = 0;
#endif // wxUSE_LOG

    // create the global object used for printing out messages
    virtual wxMessageOutput *CreateMessageOutput() = 0;

#if wxUSE_FONTMAP
    // create the global font mapper object used for encodings/charset mapping
    virtual wxFontMapper *CreateFontMapper() = 0;
#endif // wxUSE_FONTMAP

    // get the renderer to use for drawing the generic controls (return value
    // may be NULL in which case the default renderer for the current platform
    // is used); this is used in GUI only and always returns NULL in console
    //
    // NB: returned pointer will be deleted by the caller
    virtual wxRendererNative *CreateRenderer() = 0;

    // wxStandardPaths object is normally the same for wxBase and wxGUI
    virtual wxStandardPaths& GetStandardPaths();


    // functions abstracting differences between GUI and console modes
    // ------------------------------------------------------------------------

    // show the assert dialog with the specified message in GUI or just print
    // the string to stderr in console mode
    //
    // base class version has an implementation (in spite of being pure
    // virtual) in base/appbase.cpp which can be called as last resort.
    //
    // return true to suppress subsequent asserts, false to continue as before
    virtual bool ShowAssertDialog(const wxString& msg) = 0;

    // show the message safely to the user, i.e. show it in a message box if
    // possible (even in a console application!) or return false if we can't do
    // it (e.g. GUI is not initialized at all)
    //
    // note that this function can be called even when wxApp doesn't exist, as
    // it's supposed to be always safe to call -- hence the name
    //
    // return true if the message box was shown, false if nothing was done
    virtual bool SafeMessageBox(const wxString& text, const wxString& title) = 0;

    // return true if fprintf(stderr) goes somewhere, false otherwise
    virtual bool HasStderr() = 0;

#if wxUSE_SOCKETS
    // this function is used by wxNet library to set the default socket manager
    // to use: doing it like this allows us to keep all socket-related code in
    // wxNet instead of having to pull it in wxBase itself as we'd have to do
    // if we really implemented wxSocketManager here
    //
    // we don't take ownership of this pointer, it should have a lifetime
    // greater than that of any socket (e.g. be a pointer to a static object)
    static void SetDefaultSocketManager(wxSocketManager *manager)
    {
        ms_manager = manager;
    }

    // return socket manager: this is usually different for console and GUI
    // applications (although some ports use the same implementation for both)
    virtual wxSocketManager *GetSocketManager() { return ms_manager; }
#endif

    // create a new, port specific, instance of the event loop used by wxApp
    virtual wxEventLoopBase *CreateEventLoop() = 0;

#if wxUSE_TIMER
    // return platform and toolkit dependent wxTimer implementation
    virtual wxTimerImpl *CreateTimerImpl(wxTimer *timer) = 0;
#endif

#if wxUSE_THREADS
    virtual void MutexGuiEnter();
    virtual void MutexGuiLeave();
#endif

    // functions returning port-specific information
    // ------------------------------------------------------------------------

    // return information about the (native) toolkit currently used and its
    // runtime (not compile-time) version.
    // returns wxPORT_BASE for console applications and one of the remaining
    // wxPORT_* values for GUI applications.
    virtual wxPortId GetToolkitVersion(int *majVer = NULL,
                                       int *minVer = NULL,
                                       int *microVer = NULL) const = 0;

    // return true if the port is using wxUniversal for the GUI, false if not
    virtual bool IsUsingUniversalWidgets() const = 0;

    // return the name of the Desktop Environment such as
    // "KDE" or "GNOME". May return an empty string.
    virtual wxString GetDesktopEnvironment() const = 0;

    // returns a short string to identify the block of the standard command
    // line options parsed automatically by current port: if this string is
    // empty, there are no such options, otherwise the function also fills
    // passed arrays with the names and the descriptions of those options.
    virtual wxString GetStandardCmdLineOptions(wxArrayString& names,
                                               wxArrayString& desc) const
    {
        wxUnusedVar(names);
        wxUnusedVar(desc);

        return wxEmptyString;
    }


#if wxUSE_STACKWALKER
    // Helper function mostly useful for derived classes ShowAssertDialog()
    // implementation.
    //
    // Returns the stack frame as a plain (and possibly empty) wxString.
    virtual wxString GetAssertStackTrace();
#endif // wxUSE_STACKWALKER

private:
    static wxSocketManager *ms_manager;
};

// ----------------------------------------------------------------------------
// include the platform-specific version of the class
// ----------------------------------------------------------------------------

// NB:  test for __UNIX__ before __WXMAC__ as under Darwin we want to use the
//      Unix code (and otherwise __UNIX__ wouldn't be defined)
// ABX: check __WIN32__ instead of __WXMSW__ for the same MSWBase in any Win32 port
#if defined(__WIN32__)
    #include "wx/msw/apptbase.h"
#elif defined(__UNIX__)
    #include "wx/unix/apptbase.h"
#else // no platform-specific methods to add to wxAppTraits
    // wxAppTraits must be a class because it was forward declared as class
    class WXDLLIMPEXP_BASE wxAppTraits : public wxAppTraitsBase
    {
    };
#endif // platform

// ============================================================================
// standard traits for console and GUI applications
// ============================================================================

// ----------------------------------------------------------------------------
// wxConsoleAppTraitsBase: wxAppTraits implementation for the console apps
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxConsoleAppTraitsBase : public wxAppTraits
{
public:
#if !wxUSE_CONSOLE_EVENTLOOP
    virtual wxEventLoopBase *CreateEventLoop() wxOVERRIDE { return NULL; }
#endif // !wxUSE_CONSOLE_EVENTLOOP

#if wxUSE_LOG
    virtual wxLog *CreateLogTarget() wxOVERRIDE;
#endif // wxUSE_LOG
    virtual wxMessageOutput *CreateMessageOutput() wxOVERRIDE;
#if wxUSE_FONTMAP
    virtual wxFontMapper *CreateFontMapper() wxOVERRIDE;
#endif // wxUSE_FONTMAP
    virtual wxRendererNative *CreateRenderer() wxOVERRIDE;

    virtual bool ShowAssertDialog(const wxString& msg) wxOVERRIDE;
    virtual bool HasStderr() wxOVERRIDE;
    virtual bool SafeMessageBox(const wxString& text,
                                const wxString& title) wxOVERRIDE;

    // the GetToolkitVersion for console application is always the same
    wxPortId GetToolkitVersion(int *verMaj = NULL,
                               int *verMin = NULL,
                               int *verMicro = NULL) const wxOVERRIDE
    {
        // no toolkits (wxBase is for console applications without GUI support)
        // NB: zero means "no toolkit", -1 means "not initialized yet"
        //     so we must use zero here!
        if (verMaj) *verMaj = 0;
        if (verMin) *verMin = 0;
        if (verMicro) *verMicro = 0;
        return wxPORT_BASE;
    }

    virtual bool IsUsingUniversalWidgets() const wxOVERRIDE { return false; }
    virtual wxString GetDesktopEnvironment() const wxOVERRIDE { return wxEmptyString; }
};

// ----------------------------------------------------------------------------
// wxGUIAppTraitsBase: wxAppTraits implementation for the GUI apps
// ----------------------------------------------------------------------------

#if wxUSE_GUI

class WXDLLIMPEXP_CORE wxGUIAppTraitsBase : public wxAppTraits
{
public:
#if wxUSE_LOG
    virtual wxLog *CreateLogTarget() wxOVERRIDE;
#endif // wxUSE_LOG
    virtual wxMessageOutput *CreateMessageOutput() wxOVERRIDE;
#if wxUSE_FONTMAP
    virtual wxFontMapper *CreateFontMapper() wxOVERRIDE;
#endif // wxUSE_FONTMAP
    virtual wxRendererNative *CreateRenderer() wxOVERRIDE;

    virtual bool ShowAssertDialog(const wxString& msg) wxOVERRIDE;
    virtual bool HasStderr() wxOVERRIDE;

    // Win32 has its own implementation using native message box directly in
    // the base class, don't override it.
#ifndef __WIN32__
    virtual bool SafeMessageBox(const wxString& text,
                                const wxString& title) wxOVERRIDE;
#endif // !__WIN32__

    virtual bool IsUsingUniversalWidgets() const wxOVERRIDE
    {
    #ifdef __WXUNIVERSAL__
        return true;
    #else
        return false;
    #endif
    }

    virtual wxString GetDesktopEnvironment() const wxOVERRIDE { return wxEmptyString; }
};

#endif // wxUSE_GUI

// ----------------------------------------------------------------------------
// include the platform-specific version of the classes above
// ----------------------------------------------------------------------------

// ABX: check __WIN32__ instead of __WXMSW__ for the same MSWBase in any Win32 port
#if defined(__WIN32__)
    #include "wx/msw/apptrait.h"
#elif defined(__UNIX__)
    #include "wx/unix/apptrait.h"
#else
    #if wxUSE_GUI
        class wxGUIAppTraits : public wxGUIAppTraitsBase
        {
        };
    #endif // wxUSE_GUI
    class wxConsoleAppTraits: public wxConsoleAppTraitsBase
    {
    };
#endif // platform

#endif // _WX_APPTRAIT_H_

