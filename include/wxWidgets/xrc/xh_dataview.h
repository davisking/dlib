/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_dataview.h
// Purpose:     XML resource handler for wxDataViewCtrl
// Author:      Anton Triest
// Created:     2019/06/27
// Copyright:   (c) 2019 Anton Triest, Bricsys
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_DATAVIEW_H_
#define _WX_XH_DATAVIEW_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_DATAVIEWCTRL

class WXDLLIMPEXP_XRC wxDataViewXmlHandler : public wxXmlResourceHandler
{
    wxDECLARE_DYNAMIC_CLASS(wxDataViewXmlHandler);

public:
    wxDataViewXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;

private:
    wxObject* HandleCtrl();
    wxObject* HandleListCtrl();
    wxObject* HandleTreeCtrl();
};

#endif // wxUSE_XRC && wxUSE_DATAVIEWCTRL

#endif // _WX_XH_DATAVIEW_H_
