/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_stlin.h
// Purpose:     XML resource handler for wxStaticLine
// Author:      Vaclav Slavik
// Created:     2000/09/00
// Copyright:   (c) 2000 Vaclav Slavik
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_STLIN_H_
#define _WX_XH_STLIN_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_STATLINE

class WXDLLIMPEXP_XRC wxStaticLineXmlHandler : public wxXmlResourceHandler
{
    wxDECLARE_DYNAMIC_CLASS(wxStaticLineXmlHandler);

public:
    wxStaticLineXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;
};

#endif // wxUSE_XRC && wxUSE_STATLINE

#endif // _WX_XH_STLIN_H_
