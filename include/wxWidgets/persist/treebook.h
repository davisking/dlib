///////////////////////////////////////////////////////////////////////////////
// Name:        wx/persist/treebook.h
// Purpose:     persistence support for wxBookCtrl
// Author:      Vadim Zeitlin
// Created:     2009-01-19
// Copyright:   (c) 2009 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PERSIST_TREEBOOK_H_
#define _WX_PERSIST_TREEBOOK_H_

#include "wx/persist/bookctrl.h"

#if wxUSE_TREEBOOK

#include "wx/arrstr.h"
#include "wx/treebook.h"

// ----------------------------------------------------------------------------
// string constants used by wxPersistentTreeBookCtrl
// ----------------------------------------------------------------------------

#define wxPERSIST_TREEBOOK_KIND wxASCII_STR("TreeBook")

// this key contains the indices of all expanded nodes in the tree book
// separated by wxPERSIST_TREEBOOK_EXPANDED_SEP
#define wxPERSIST_TREEBOOK_EXPANDED_BRANCHES wxASCII_STR("Expanded")
#define wxPERSIST_TREEBOOK_EXPANDED_SEP ','

// ----------------------------------------------------------------------------
// wxPersistentTreeBookCtrl: supports saving/restoring open tree branches
// ----------------------------------------------------------------------------

class wxPersistentTreeBookCtrl : public wxPersistentBookCtrl
{
public:
    wxPersistentTreeBookCtrl(wxTreebook *book)
        : wxPersistentBookCtrl(book)
    {
    }

    virtual void Save() const wxOVERRIDE
    {
        const wxTreebook * const book = GetTreeBook();

        wxString expanded;
        const size_t count = book->GetPageCount();
        for ( size_t n = 0; n < count; n++ )
        {
            if ( book->IsNodeExpanded(n) )
            {
                if ( !expanded.empty() )
                    expanded += wxPERSIST_TREEBOOK_EXPANDED_SEP;

                expanded += wxString::Format(wxASCII_STR("%u"), static_cast<unsigned>(n));
            }
        }

        SaveValue(wxPERSIST_TREEBOOK_EXPANDED_BRANCHES, expanded);

        wxPersistentBookCtrl::Save();
    }

    virtual bool Restore() wxOVERRIDE
    {
        wxTreebook * const book = GetTreeBook();

        wxString expanded;
        if ( RestoreValue(wxPERSIST_TREEBOOK_EXPANDED_BRANCHES, &expanded) )
        {
            const wxArrayString
                indices(wxSplit(expanded, wxPERSIST_TREEBOOK_EXPANDED_SEP));

            const size_t pageCount = book->GetPageCount();
            const size_t count = indices.size();
            for ( size_t n = 0; n < count; n++ )
            {
                unsigned long idx;
                if ( indices[n].ToULong(&idx) && idx < pageCount )
                    book->ExpandNode(idx);
            }
        }

        return wxPersistentBookCtrl::Restore();
    }

    virtual wxString GetKind() const wxOVERRIDE { return wxPERSIST_TREEBOOK_KIND; }

    wxTreebook *GetTreeBook() const { return static_cast<wxTreebook *>(Get()); }
};

inline wxPersistentObject *wxCreatePersistentObject(wxTreebook *book)
{
    return new wxPersistentTreeBookCtrl(book);
}

#endif // wxUSE_TREEBOOK

#endif // _WX_PERSIST_TREEBOOK_H_
