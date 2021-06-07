///////////////////////////////////////////////////////////////////////////////
// Name:        wx/unix/apptrait.h
// Purpose:     standard implementations of wxAppTraits for Unix
// Author:      Vadim Zeitlin
// Modified by:
// Created:     23.06.2003
// Copyright:   (c) 2003 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_UNIX_APPTRAIT_H_
#define _WX_UNIX_APPTRAIT_H_

// ----------------------------------------------------------------------------
// wxGUI/ConsoleAppTraits: must derive from wxAppTraits, not wxAppTraitsBase
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxConsoleAppTraits : public wxConsoleAppTraitsBase
{
public:
#if wxUSE_CONSOLE_EVENTLOOP
    virtual wxEventLoopBase *CreateEventLoop() wxOVERRIDE;
#endif // wxUSE_CONSOLE_EVENTLOOP
#if wxUSE_TIMER
    virtual wxTimerImpl *CreateTimerImpl(wxTimer *timer) wxOVERRIDE;
#endif
};

#if wxUSE_GUI

// GTK+ and Motif integrate sockets and child processes monitoring directly in
// their main loop, the other Unix ports do it at wxEventLoop level and so use
// the non-GUI traits and don't need anything here
//
// TODO: Should we use XtAddInput() for wxX11 too? Or, vice versa, if there is
//       no advantage in doing this compared to the generic way currently used
//       by wxX11, should we continue to use GTK/Motif-specific stuff?
#if defined(__WXGTK__) || defined(__WXMOTIF__) || defined(__WXQT__)
    #define wxHAS_GUI_FDIOMANAGER
    #define wxHAS_GUI_PROCESS_CALLBACKS
#endif // ports using wxFDIOManager

#if defined(__WXMAC__)
    #define wxHAS_GUI_PROCESS_CALLBACKS
    #define wxHAS_GUI_SOCKET_MANAGER
#endif

class WXDLLIMPEXP_CORE wxGUIAppTraits : public wxGUIAppTraitsBase
{
public:
    virtual wxEventLoopBase *CreateEventLoop() wxOVERRIDE;
    virtual int WaitForChild(wxExecuteData& execData) wxOVERRIDE;
#if wxUSE_TIMER
    virtual wxTimerImpl *CreateTimerImpl(wxTimer *timer) wxOVERRIDE;
#endif
#if wxUSE_THREADS && defined(__WXGTK20__)
    virtual void MutexGuiEnter() wxOVERRIDE;
    virtual void MutexGuiLeave() wxOVERRIDE;
#endif

    wxPortId GetToolkitVersion(int *majVer = NULL,
                               int *minVer = NULL,
                               int *microVer = NULL) const wxOVERRIDE;

#ifdef __WXGTK20__
    virtual wxString GetDesktopEnvironment() const wxOVERRIDE;
#endif // __WXGTK20____

#if defined(__WXGTK20__)
    virtual bool ShowAssertDialog(const wxString& msg) wxOVERRIDE;
#endif

#if wxUSE_SOCKETS

#ifdef wxHAS_GUI_SOCKET_MANAGER
    virtual wxSocketManager *GetSocketManager() wxOVERRIDE;
#endif

#ifdef wxHAS_GUI_FDIOMANAGER
    virtual wxFDIOManager *GetFDIOManager() wxOVERRIDE;
#endif

#endif // wxUSE_SOCKETS

#if wxUSE_EVENTLOOP_SOURCE
    virtual wxEventLoopSourcesManagerBase* GetEventLoopSourcesManager() wxOVERRIDE;
#endif
};

#endif // wxUSE_GUI

#endif // _WX_UNIX_APPTRAIT_H_

