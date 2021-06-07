/////////////////////////////////////////////////////////////////////////////
// Name:        webview.h
// Purpose:     Common interface and events for web view component
// Author:      Marianne Gagnon
// Copyright:   (c) 2010 Marianne Gagnon, 2011 Steven Lamerton
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_WEBVIEW_H_
#define _WX_WEBVIEW_H_

#include "wx/defs.h"

#if wxUSE_WEBVIEW

#include "wx/control.h"
#include "wx/event.h"
#include "wx/sstream.h"
#include "wx/sharedptr.h"
#include "wx/vector.h"
#include "wx/versioninfo.h"

#if defined(__WXOSX__)
    #include "wx/osx/webviewhistoryitem_webkit.h"
#elif defined(__WXGTK__)
    #include "wx/gtk/webviewhistoryitem_webkit.h"
#elif defined(__WXMSW__)
    #include "wx/msw/webviewhistoryitem_ie.h"
#else
    #error "wxWebView not implemented on this platform."
#endif

class wxFSFile;
class wxFileSystem;
class wxWebView;

enum wxWebViewZoom
{
    wxWEBVIEW_ZOOM_TINY,
    wxWEBVIEW_ZOOM_SMALL,
    wxWEBVIEW_ZOOM_MEDIUM,
    wxWEBVIEW_ZOOM_LARGE,
    wxWEBVIEW_ZOOM_LARGEST
};

enum wxWebViewZoomType
{
    //Scales entire page, including images
    wxWEBVIEW_ZOOM_TYPE_LAYOUT,
    wxWEBVIEW_ZOOM_TYPE_TEXT
};

enum wxWebViewNavigationError
{
    wxWEBVIEW_NAV_ERR_CONNECTION,
    wxWEBVIEW_NAV_ERR_CERTIFICATE,
    wxWEBVIEW_NAV_ERR_AUTH,
    wxWEBVIEW_NAV_ERR_SECURITY,
    wxWEBVIEW_NAV_ERR_NOT_FOUND,
    wxWEBVIEW_NAV_ERR_REQUEST,
    wxWEBVIEW_NAV_ERR_USER_CANCELLED,
    wxWEBVIEW_NAV_ERR_OTHER
};

enum wxWebViewReloadFlags
{
    //Default, may access cache
    wxWEBVIEW_RELOAD_DEFAULT,
    wxWEBVIEW_RELOAD_NO_CACHE
};

enum wxWebViewFindFlags
{
    wxWEBVIEW_FIND_WRAP =             0x0001,
    wxWEBVIEW_FIND_ENTIRE_WORD =      0x0002,
    wxWEBVIEW_FIND_MATCH_CASE =       0x0004,
    wxWEBVIEW_FIND_HIGHLIGHT_RESULT = 0x0008,
    wxWEBVIEW_FIND_BACKWARDS =        0x0010,
    wxWEBVIEW_FIND_DEFAULT =          0
};

enum wxWebViewNavigationActionFlags
{
    wxWEBVIEW_NAV_ACTION_NONE,
    wxWEBVIEW_NAV_ACTION_USER,
    wxWEBVIEW_NAV_ACTION_OTHER
};

enum wxWebViewUserScriptInjectionTime
{
    wxWEBVIEW_INJECT_AT_DOCUMENT_START,
    wxWEBVIEW_INJECT_AT_DOCUMENT_END
};

//Base class for custom scheme handlers
class WXDLLIMPEXP_WEBVIEW wxWebViewHandler
{
public:
    wxWebViewHandler(const wxString& scheme)
        : m_scheme(scheme), m_securityURL() {}
    virtual ~wxWebViewHandler() {}
    virtual wxString GetName() const { return m_scheme; }
    virtual wxFSFile* GetFile(const wxString &uri) = 0;
    virtual void SetSecurityURL(const wxString& url) { m_securityURL = url; }
    virtual wxString GetSecurityURL() const { return m_securityURL; }
private:
    wxString m_scheme;
    wxString m_securityURL;
};

extern WXDLLIMPEXP_DATA_WEBVIEW(const char) wxWebViewNameStr[];
extern WXDLLIMPEXP_DATA_WEBVIEW(const char) wxWebViewDefaultURLStr[];
extern WXDLLIMPEXP_DATA_WEBVIEW(const char) wxWebViewBackendDefault[];
extern WXDLLIMPEXP_DATA_WEBVIEW(const char) wxWebViewBackendIE[];
extern WXDLLIMPEXP_DATA_WEBVIEW(const char) wxWebViewBackendEdge[];
extern WXDLLIMPEXP_DATA_WEBVIEW(const char) wxWebViewBackendWebKit[];

class WXDLLIMPEXP_WEBVIEW wxWebViewFactory : public wxObject
{
public:
    virtual wxWebView* Create() = 0;
    virtual wxWebView* Create(wxWindow* parent,
                              wxWindowID id,
                              const wxString& url = wxASCII_STR(wxWebViewDefaultURLStr),
                              const wxPoint& pos = wxDefaultPosition,
                              const wxSize& size = wxDefaultSize,
                              long style = 0,
                              const wxString& name = wxASCII_STR(wxWebViewNameStr)) = 0;
    virtual bool IsAvailable() { return true; }
    virtual wxVersionInfo GetVersionInfo() { return wxVersionInfo(); }
};

WX_DECLARE_STRING_HASH_MAP(wxSharedPtr<wxWebViewFactory>, wxStringWebViewFactoryMap);

class WXDLLIMPEXP_WEBVIEW wxWebView : public wxControl
{
public:
    wxWebView()
    {
        m_showMenu = true;
        m_runScriptCount = 0;
    }

    virtual ~wxWebView() {}

    virtual bool Create(wxWindow* parent,
           wxWindowID id,
           const wxString& url = wxASCII_STR(wxWebViewDefaultURLStr),
           const wxPoint& pos = wxDefaultPosition,
           const wxSize& size = wxDefaultSize,
           long style = 0,
           const wxString& name = wxASCII_STR(wxWebViewNameStr)) = 0;

    // Factory methods allowing the use of custom factories registered with
    // RegisterFactory
    static wxWebView* New(const wxString& backend = wxASCII_STR(wxWebViewBackendDefault));
    static wxWebView* New(wxWindow* parent,
                          wxWindowID id,
                          const wxString& url = wxASCII_STR(wxWebViewDefaultURLStr),
                          const wxPoint& pos = wxDefaultPosition,
                          const wxSize& size = wxDefaultSize,
                          const wxString& backend = wxASCII_STR(wxWebViewBackendDefault),
                          long style = 0,
                          const wxString& name = wxASCII_STR(wxWebViewNameStr));

    static void RegisterFactory(const wxString& backend,
                                wxSharedPtr<wxWebViewFactory> factory);
    static bool IsBackendAvailable(const wxString& backend);
    static wxVersionInfo GetBackendVersionInfo(const wxString& backend = wxASCII_STR(wxWebViewBackendDefault));

    // General methods
    virtual void EnableContextMenu(bool enable = true)
    {
        m_showMenu = enable;
    }
    virtual void EnableAccessToDevTools(bool WXUNUSED(enable) = true) { }
    virtual wxString GetCurrentTitle() const = 0;
    virtual wxString GetCurrentURL() const = 0;
    // TODO: handle choosing a frame when calling GetPageSource()?
    virtual wxString GetPageSource() const;
    virtual wxString GetPageText() const;
    virtual bool IsBusy() const = 0;
    virtual bool IsContextMenuEnabled() const { return m_showMenu; }
    virtual bool IsAccessToDevToolsEnabled() const { return false; }
    virtual bool IsEditable() const = 0;
    virtual void LoadURL(const wxString& url) = 0;
    virtual void Print() = 0;
    virtual void RegisterHandler(wxSharedPtr<wxWebViewHandler> handler) = 0;
    virtual void Reload(wxWebViewReloadFlags flags = wxWEBVIEW_RELOAD_DEFAULT) = 0;
    virtual bool SetUserAgent(const wxString& userAgent) { wxUnusedVar(userAgent); return false; }
    virtual wxString GetUserAgent() const;

    // Script
    virtual bool RunScript(const wxString& javascript, wxString* output = NULL) const = 0;
    virtual bool AddScriptMessageHandler(const wxString& name)
    { wxUnusedVar(name); return false; }
    virtual bool RemoveScriptMessageHandler(const wxString& name)
    { wxUnusedVar(name); return false; }
    virtual bool AddUserScript(const wxString& javascript,
        wxWebViewUserScriptInjectionTime injectionTime = wxWEBVIEW_INJECT_AT_DOCUMENT_START)
    {  wxUnusedVar(javascript); wxUnusedVar(injectionTime); return false; }
    virtual void RemoveAllUserScripts() {}

    virtual void SetEditable(bool enable = true) = 0;
    void SetPage(const wxString& html, const wxString& baseUrl)
    {
        DoSetPage(html, baseUrl);
    }
    void SetPage(wxInputStream& html, wxString baseUrl)
    {
        wxStringOutputStream stream;
        stream.Write(html);
        DoSetPage(stream.GetString(), baseUrl);
    }
    virtual void Stop() = 0;

    //History
    virtual bool CanGoBack() const = 0;
    virtual bool CanGoForward() const = 0;
    virtual void GoBack() = 0;
    virtual void GoForward() = 0;
    virtual void ClearHistory() = 0;
    virtual void EnableHistory(bool enable = true) = 0;
    virtual wxVector<wxSharedPtr<wxWebViewHistoryItem> > GetBackwardHistory() = 0;
    virtual wxVector<wxSharedPtr<wxWebViewHistoryItem> > GetForwardHistory() = 0;
    virtual void LoadHistoryItem(wxSharedPtr<wxWebViewHistoryItem> item) = 0;

    //Zoom
    virtual bool CanSetZoomType(wxWebViewZoomType type) const = 0;
    virtual wxWebViewZoom GetZoom() const;
    virtual float GetZoomFactor() const = 0;
    virtual wxWebViewZoomType GetZoomType() const = 0;
    virtual void SetZoom(wxWebViewZoom zoom);
    virtual void SetZoomFactor(float zoom) = 0;
    virtual void SetZoomType(wxWebViewZoomType zoomType) = 0;

    //Selection
    virtual void SelectAll() ;
    virtual bool HasSelection() const;
    virtual void DeleteSelection();
    virtual wxString GetSelectedText() const;
    virtual wxString GetSelectedSource() const;
    virtual void ClearSelection();

    //Clipboard functions
    virtual bool CanCut() const;
    virtual bool CanCopy() const;
    virtual bool CanPaste() const;
    virtual void Cut();
    virtual void Copy();
    virtual void Paste();

    //Undo / redo functionality
    virtual bool CanUndo() const = 0;
    virtual bool CanRedo() const = 0;
    virtual void Undo() = 0;
    virtual void Redo() = 0;

    //Get the pointer to the underlying native engine.
    virtual void* GetNativeBackend() const = 0;
    //Find function
    virtual long Find(const wxString& text, int flags = wxWEBVIEW_FIND_DEFAULT);

protected:
    virtual void DoSetPage(const wxString& html, const wxString& baseUrl) = 0;

    bool QueryCommandEnabled(const wxString& command) const;
    void ExecCommand(const wxString& command);

    // Count the number of calls to RunScript() in order to prevent
    // the_same variable from being used twice in more than one call.
    mutable int m_runScriptCount;

private:
    static void InitFactoryMap();
    static wxStringWebViewFactoryMap::iterator FindFactory(const wxString &backend);

    bool m_showMenu;
    wxString m_findText;
    static wxStringWebViewFactoryMap m_factoryMap;

    wxDECLARE_ABSTRACT_CLASS(wxWebView);
};

class WXDLLIMPEXP_WEBVIEW wxWebViewEvent : public wxNotifyEvent
{
public:
    wxWebViewEvent() {}
    wxWebViewEvent(wxEventType type, int id, const wxString& url,
                   const wxString target,
                   wxWebViewNavigationActionFlags flags = wxWEBVIEW_NAV_ACTION_NONE,
                   const wxString& messageHandler = wxString())
        : wxNotifyEvent(type, id), m_url(url), m_target(target),
          m_actionFlags(flags), m_messageHandler(messageHandler)
    {}


    const wxString& GetURL() const { return m_url; }
    const wxString& GetTarget() const { return m_target; }

    wxWebViewNavigationActionFlags GetNavigationAction() const { return m_actionFlags; }
    const wxString& GetMessageHandler() const { return m_messageHandler; }

    virtual wxEvent* Clone() const wxOVERRIDE { return new wxWebViewEvent(*this); }
private:
    wxString m_url;
    wxString m_target;
    wxWebViewNavigationActionFlags m_actionFlags;
    wxString m_messageHandler;

    wxDECLARE_DYNAMIC_CLASS_NO_ASSIGN(wxWebViewEvent);
};

wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_NAVIGATING, wxWebViewEvent );
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_NAVIGATED, wxWebViewEvent );
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_LOADED, wxWebViewEvent );
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_ERROR, wxWebViewEvent );
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_NEWWINDOW, wxWebViewEvent );
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_TITLE_CHANGED, wxWebViewEvent );
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_FULLSCREEN_CHANGED, wxWebViewEvent);
wxDECLARE_EXPORTED_EVENT( WXDLLIMPEXP_WEBVIEW, wxEVT_WEBVIEW_SCRIPT_MESSAGE_RECEIVED, wxWebViewEvent);

typedef void (wxEvtHandler::*wxWebViewEventFunction)
             (wxWebViewEvent&);

#define wxWebViewEventHandler(func) \
    wxEVENT_HANDLER_CAST(wxWebViewEventFunction, func)

#define EVT_WEBVIEW_NAVIGATING(id, fn) \
    wx__DECLARE_EVT1(wxEVT_WEBVIEW_NAVIGATING, id, \
                     wxWebViewEventHandler(fn))

#define EVT_WEBVIEW_NAVIGATED(id, fn) \
    wx__DECLARE_EVT1(wxEVT_WEBVIEW_NAVIGATED, id, \
                     wxWebViewEventHandler(fn))

#define EVT_WEBVIEW_LOADED(id, fn) \
    wx__DECLARE_EVT1(wxEVT_WEBVIEW_LOADED, id, \
                     wxWebViewEventHandler(fn))

#define EVT_WEBVIEW_ERROR(id, fn) \
    wx__DECLARE_EVT1(wxEVT_WEBVIEW_ERROR, id, \
                     wxWebViewEventHandler(fn))

#define EVT_WEBVIEW_NEWWINDOW(id, fn) \
    wx__DECLARE_EVT1(wxEVT_WEBVIEW_NEWWINDOW, id, \
                     wxWebViewEventHandler(fn))

#define EVT_WEBVIEW_TITLE_CHANGED(id, fn) \
    wx__DECLARE_EVT1(wxEVT_WEBVIEW_TITLE_CHANGED, id, \
                     wxWebViewEventHandler(fn))

// old wxEVT_COMMAND_* constants
#define wxEVT_COMMAND_WEBVIEW_NAVIGATING      wxEVT_WEBVIEW_NAVIGATING
#define wxEVT_COMMAND_WEBVIEW_NAVIGATED       wxEVT_WEBVIEW_NAVIGATED
#define wxEVT_COMMAND_WEBVIEW_LOADED          wxEVT_WEBVIEW_LOADED
#define wxEVT_COMMAND_WEBVIEW_ERROR           wxEVT_WEBVIEW_ERROR
#define wxEVT_COMMAND_WEBVIEW_NEWWINDOW       wxEVT_WEBVIEW_NEWWINDOW
#define wxEVT_COMMAND_WEBVIEW_TITLE_CHANGED   wxEVT_WEBVIEW_TITLE_CHANGED

#endif // wxUSE_WEBVIEW

#endif // _WX_WEBVIEW_H_
