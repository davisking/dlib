/////////////////////////////////////////////////////////////////////////////
// Name:        wx/xrc/xh_dirpicker.h
// Purpose:     XML resource handler for wxDirPickerCtrl
// Author:      Francesco Montorsi
// Created:     2006-04-17
// Copyright:   (c) 2006 Francesco Montorsi
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_XH_DIRPICKERCTRL_H_
#define _WX_XH_DIRPICKERCTRL_H_

#include "wx/xrc/xmlres.h"

#if wxUSE_XRC && wxUSE_DIRPICKERCTRL

class WXDLLIMPEXP_XRC wxDirPickerCtrlXmlHandler : public wxXmlResourceHandler
{
    wxDECLARE_DYNAMIC_CLASS(wxDirPickerCtrlXmlHandler);

public:
    wxDirPickerCtrlXmlHandler();
    virtual wxObject *DoCreateResource() wxOVERRIDE;
    virtual bool CanHandle(wxXmlNode *node) wxOVERRIDE;
};

#endif // wxUSE_XRC && wxUSE_DIRPICKERCTRL

#endif // _WX_XH_DIRPICKERCTRL_H_
