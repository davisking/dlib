///////////////////////////////////////////////////////////////////////////////
// Name:        wx/persist/bookctrl.h
// Purpose:     persistence support for wxBookCtrl
// Author:      Vadim Zeitlin
// Created:     2009-01-19
// Copyright:   (c) 2009 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PERSIST_BOOKCTRL_H_
#define _WX_PERSIST_BOOKCTRL_H_

#include "wx/persist/window.h"

#if wxUSE_BOOKCTRL

#include "wx/bookctrl.h"

// ----------------------------------------------------------------------------
// string constants used by wxPersistentBookCtrl
// ----------------------------------------------------------------------------

#define wxPERSIST_BOOK_KIND wxASCII_STR("Book")

#define wxPERSIST_BOOK_SELECTION wxASCII_STR("Selection")

// ----------------------------------------------------------------------------
// wxPersistentBookCtrl: supports saving/restoring book control selection
// ----------------------------------------------------------------------------

class wxPersistentBookCtrl : public wxPersistentWindow<wxBookCtrlBase>
{
public:
    wxPersistentBookCtrl(wxBookCtrlBase *book)
        : wxPersistentWindow<wxBookCtrlBase>(book)
    {
    }

    virtual void Save() const wxOVERRIDE
    {
        SaveValue(wxPERSIST_BOOK_SELECTION, Get()->GetSelection());
    }

    virtual bool Restore() wxOVERRIDE
    {
        long sel;
        if ( RestoreValue(wxPERSIST_BOOK_SELECTION, &sel) )
        {
            wxBookCtrlBase * const book = Get();
            if ( sel >= 0 && (unsigned)sel < book->GetPageCount() )
            {
                book->SetSelection(sel);
                return true;
            }
        }

        return false;
    }

    virtual wxString GetKind() const wxOVERRIDE { return wxPERSIST_BOOK_KIND; }
};

inline wxPersistentObject *wxCreatePersistentObject(wxBookCtrlBase *book)
{
    return new wxPersistentBookCtrl(book);
}

#endif // wxUSE_BOOKCTRL

#endif // _WX_PERSIST_BOOKCTRL_H_
