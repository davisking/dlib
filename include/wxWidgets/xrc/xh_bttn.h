/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_bttn.h
// Purpose:     XML resource handler for buttons
// Author:      Vaclav Slavik
// Created:     2000/03/05
// Copyright:   (c) 2000 Vaclav Slavik
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_BTTN_H_
#define _WX_XH_BTTN_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_BUTTON

class WXDLLIMPEXP_XRC wxButtonXmlHandler : public wxXmlResourceHandler
{
    wxDECLARE_DYNAMIC_CLASS(wxButtonXmlHandler);

public:
    wxButtonXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;
};

#endif // wxUSE_XRC && wxUSE_BUTTON

#endif // _WX_XH_BTTN_H_
