/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_toolb.h
// Purpose:     XML resource handler for wxToolBar
// Author:      Vaclav Slavik
// Created:     2000/08/11
// Copyright:   (c) 2000 Vaclav Slavik
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_TOOLB_H_
#define _WX_XH_TOOLB_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_TOOLBAR

class WXDLLIMPEXP_FWD_CORE wxToolBar;

class WXDLLIMPEXP_XRC wxToolBarXmlHandler : public wxXmlResourceHandler
{
    wxDECLARE_DYNAMIC_CLASS(wxToolBarXmlHandler);

public:
    wxToolBarXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;

private:
    bool m_isInside;
    wxToolBar *m_toolbar;
    wxSize m_toolSize;
};

#endif // wxUSE_XRC && wxUSE_TOOLBAR

#endif // _WX_XH_TOOLB_H_
