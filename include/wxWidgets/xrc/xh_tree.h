/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_tree.h
// Purpose:     XML resource handler for wxTreeCtrl
// Author:      Brian Gavin
// Created:     2000/09/09
// Copyright:   (c) 2000 Brian Gavin
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_TREE_H_
#define _WX_XH_TREE_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_TREECTRL

class WXDLLIMPEXP_XRC wxTreeCtrlXmlHandler : public wxXmlResourceHandler
{
    wxDECLARE_DYNAMIC_CLASS(wxTreeCtrlXmlHandler);

public:
    wxTreeCtrlXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;
};

#endif // wxUSE_XRC && wxUSE_TREECTRL

#endif // _WX_XH_TREE_H_
