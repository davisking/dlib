/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_htmllbox.h
// Purpose:     XML resource handler for wxSimpleHtmlListBox
// Author:      Francesco Montorsi
// Created:     2006/10/21
// Copyright:   (c) 2006 Francesco Montorsi
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_SIMPLEHTMLLISTBOX_H_
#define _WX_XH_SIMPLEHTMLLISTBOX_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_HTML

class WXDLLIMPEXP_XRC wxSimpleHtmlListBoxXmlHandler : public wxXmlResourceHandler
{
public:
    wxSimpleHtmlListBoxXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;

private:
    bool m_insideBox;
    wxArrayString strList;

    wxDECLARE_DYNAMIC_CLASS(wxSimpleHtmlListBoxXmlHandler);
};

#endif // wxUSE_XRC && wxUSE_HTML

#endif // _WX_XH_SIMPLEHTMLLISTBOX_H_
