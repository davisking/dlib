/////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/imaglist.h
// Purpose:
// Author:      Robert Roebling
// Created:     01/02/97
// Copyright:   (c) 1998 Robert Roebling and Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_IMAGLISTG_H_
#define _WX_IMAGLISTG_H_

#include "wx/bitmap.h"
#include "wx/gdicmn.h"
#include "wx/vector.h"

class WXDLLIMPEXP_FWD_CORE wxDC;
class WXDLLIMPEXP_FWD_CORE wxIcon;
class WXDLLIMPEXP_FWD_CORE wxColour;


class WXDLLIMPEXP_CORE wxGenericImageList: public wxObject
{
public:
    wxGenericImageList() { }
    wxGenericImageList( int width, int height, bool mask = true, int initialCount = 1 );
    virtual ~wxGenericImageList();
    bool Create( int width, int height, bool mask = true, int initialCount = 1 );

    virtual int GetImageCount() const;
    virtual bool GetSize( int index, int &width, int &height ) const;
    virtual wxSize GetSize() const { return m_size; }

    int Add( const wxBitmap& bitmap );
    int Add( const wxBitmap& bitmap, const wxBitmap& mask );
    int Add( const wxBitmap& bitmap, const wxColour& maskColour );
    wxBitmap GetBitmap(int index) const;
    wxIcon GetIcon(int index) const;
    bool Replace( int index,
                  const wxBitmap& bitmap,
                  const wxBitmap& mask = wxNullBitmap );
    bool Remove( int index );
    bool RemoveAll();

    virtual bool Draw(int index, wxDC& dc, int x, int y,
              int flags = wxIMAGELIST_DRAW_NORMAL,
              bool solidBackground = false);

#if WXWIN_COMPATIBILITY_3_0
    wxDEPRECATED_MSG("Don't use this overload: it's not portable and does nothing")
    bool Create() { return true; }

    wxDEPRECATED_MSG("Use GetBitmap() instead")
    const wxBitmap *GetBitmapPtr(int index) const { return DoGetPtr(index); }
#endif // WXWIN_COMPATIBILITY_3_0

private:
    const wxBitmap *DoGetPtr(int index) const;

    wxVector<wxBitmap> m_images;
    bool m_useMask;

    // Size of a single bitmap in the list.
    wxSize m_size;
    // Images in the list should have the same scale factor.
    double m_scaleFactor;

    wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxGenericImageList);
};

#ifndef wxHAS_NATIVE_IMAGELIST

/*
 * wxImageList has to be a real class or we have problems with
 * the run-time information.
 */

class WXDLLIMPEXP_CORE wxImageList: public wxGenericImageList
{
    wxDECLARE_DYNAMIC_CLASS(wxImageList);

public:
    wxImageList() {}

    wxImageList( int width, int height, bool mask = true, int initialCount = 1 )
        : wxGenericImageList(width, height, mask, initialCount)
    {
    }
};
#endif // !wxHAS_NATIVE_IMAGELIST

#endif  // _WX_IMAGLISTG_H_
