/////////////////////////////////////////////////////////////////////////////
// Name:        wx/html/htmlfilt.h
// Purpose:     filters
// Author:      Vaclav Slavik
// Copyright:   (c) 1999 Vaclav Slavik
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_HTMLFILT_H_
#define _WX_HTMLFILT_H_

#include "wx/defs.h"

#if wxUSE_HTML

#include "wx/filesys.h"


//--------------------------------------------------------------------------------
// wxHtmlFilter
//                  This class is input filter. It can "translate" files
//                  in non-HTML format to HTML format
//                  interface to access certain
//                  kinds of files (HTPP, FTP, local, tar.gz etc..)
//--------------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlFilter : public wxObject
{
    wxDECLARE_ABSTRACT_CLASS(wxHtmlFilter);

public:
    wxHtmlFilter() : wxObject() {}
    virtual ~wxHtmlFilter() {}

    // returns true if this filter is able to open&read given file
    virtual bool CanRead(const wxFSFile& file) const = 0;

    // Reads given file and returns HTML document.
    // Returns empty string if opening failed
    virtual wxString ReadFile(const wxFSFile& file) const = 0;
};



//--------------------------------------------------------------------------------
// wxHtmlFilterPlainText
//                  This filter is used as default filter if no other can
//                  be used (= unknown type of file). It is used by
//                  wxHtmlWindow itself.
//--------------------------------------------------------------------------------


class WXDLLIMPEXP_HTML wxHtmlFilterPlainText : public wxHtmlFilter
{
    wxDECLARE_DYNAMIC_CLASS(wxHtmlFilterPlainText);

public:
    virtual bool CanRead(const wxFSFile& file) const wxOVERRIDE;
    virtual wxString ReadFile(const wxFSFile& file) const wxOVERRIDE;
};

//--------------------------------------------------------------------------------
// wxHtmlFilterHTML
//          filter for text/html
//--------------------------------------------------------------------------------

class wxHtmlFilterHTML : public wxHtmlFilter
{
    wxDECLARE_DYNAMIC_CLASS(wxHtmlFilterHTML);

    public:
        virtual bool CanRead(const wxFSFile& file) const wxOVERRIDE;
        virtual wxString ReadFile(const wxFSFile& file) const wxOVERRIDE;
};



#endif // wxUSE_HTML

#endif // _WX_HTMLFILT_H_

