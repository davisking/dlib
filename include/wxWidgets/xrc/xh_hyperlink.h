/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_hyperlink.h
// Purpose:     Hyperlink control (wxAdv)
// Author:      David Norris <danorris@gmail.com>
// Modified by: Ryan Norton, Francesco Montorsi
// Created:     04/02/2005
// Copyright:   (c) 2005 David Norris
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_HYPERLINKH__
#define _WX_XH_HYPERLINKH__

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_HYPERLINKCTRL

class WXDLLIMPEXP_XRC wxHyperlinkCtrlXmlHandler : public wxXmlResourceHandler
{
    // Register with wxWindows' dynamic class subsystem.
    wxDECLARE_DYNAMIC_CLASS(wxHyperlinkCtrlXmlHandler);

public:
    // Constructor.
    wxHyperlinkCtrlXmlHandler();

    // Creates the control and returns a pointer to it.
    virtual wxObject *DoCreateResource() wxOVERRIDE;

    // Returns true if we know how to create a control for the given node.
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;
};

#endif // wxUSE_XRC && wxUSE_HYPERLINKCTRL

#endif // _WX_XH_HYPERLINKH__
