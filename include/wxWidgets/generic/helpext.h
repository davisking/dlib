/////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/helpext.h
// Purpose:     an external help controller for wxWidgets
// Author:      Karsten Ballueder (Ballueder@usa.net)
// Modified by:
// Copyright:   (c) Karsten Ballueder 1998
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef __WX_HELPEXT_H_
#define __WX_HELPEXT_H_

#if wxUSE_HELP


// ----------------------------------------------------------------------------
// headers
// ----------------------------------------------------------------------------

#include "wx/helpbase.h"


// ----------------------------------------------------------------------------
// wxExtHelpController
// ----------------------------------------------------------------------------

// This class implements help via an external browser.
class WXDLLIMPEXP_ADV wxExtHelpController : public wxHelpControllerBase
{
public:
    wxExtHelpController(wxWindow* parentWindow = NULL);
    virtual ~wxExtHelpController();

#if WXWIN_COMPATIBILITY_2_8
    wxDEPRECATED(void SetBrowser(const wxString& browsername = wxEmptyString, bool isNetscape = false) );
#endif

    // Set viewer: new name for SetBrowser
    virtual void SetViewer(const wxString& viewer = wxEmptyString,
                            long flags = wxHELP_NETSCAPE) wxOVERRIDE;

    virtual bool Initialize(const wxString& dir, int WXUNUSED(server)) wxOVERRIDE
        { return Initialize(dir); }

    virtual bool Initialize(const wxString& dir) wxOVERRIDE;
    virtual bool LoadFile(const wxString& file = wxEmptyString) wxOVERRIDE;
    virtual bool DisplayContents() wxOVERRIDE;
    virtual bool DisplaySection(int sectionNo) wxOVERRIDE;
    virtual bool DisplaySection(const wxString& section) wxOVERRIDE;
    virtual bool DisplayBlock(long blockNo) wxOVERRIDE;
    virtual bool KeywordSearch(const wxString& k,
                                wxHelpSearchMode mode = wxHELP_SEARCH_ALL) wxOVERRIDE;

    virtual bool Quit() wxOVERRIDE;
    virtual void OnQuit() wxOVERRIDE;

    virtual bool DisplayHelp(const wxString &) ;

    virtual void SetFrameParameters(const wxString& WXUNUSED(title),
                                    const wxSize& WXUNUSED(size),
                                    const wxPoint& WXUNUSED(pos) = wxDefaultPosition,
                                    bool WXUNUSED(newFrameEachTime) = false) wxOVERRIDE
        {
            // does nothing by default
        }

    virtual wxFrame *GetFrameParameters(wxSize *WXUNUSED(size) = NULL,
                                    wxPoint *WXUNUSED(pos) = NULL,
                                    bool *WXUNUSED(newFrameEachTime) = NULL) wxOVERRIDE
        {
            return NULL; // does nothing by default
        }

protected:
    // Filename of currently active map file.
    wxString         m_helpDir;

    // How many entries do we have in the map file?
    int              m_NumOfEntries;

    // A list containing all id,url,documentation triples.
    wxList          *m_MapList;

private:
    // parse a single line of the map file (called by LoadFile())
    //
    // return true if the line was valid or false otherwise
    bool ParseMapFileLine(const wxString& line);

    // Deletes the list and all objects.
    void DeleteList();


    // How to call the html viewer.
    wxString         m_BrowserName;

    // Is the viewer a variant of netscape?
    bool             m_BrowserIsNetscape;

    wxDECLARE_CLASS(wxExtHelpController);
};

#endif // wxUSE_HELP

#endif // __WX_HELPEXT_H_
