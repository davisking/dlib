/////////////////////////////////////////////////////////////////////////////
// Name:        wx/progdlg.h
// Purpose:     Base header for wxProgressDialog
// Author:      Julian Smart
// Modified by:
// Created:
// Copyright:   (c) Julian Smart
// Licence:     wxWindows Licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PROGDLG_H_BASE_
#define _WX_PROGDLG_H_BASE_

#include "wx/defs.h"

#if wxUSE_PROGRESSDLG

/*
 * wxProgressDialog flags
 */
#define wxPD_CAN_ABORT          0x0001
#define wxPD_APP_MODAL          0x0002
#define wxPD_AUTO_HIDE          0x0004
#define wxPD_ELAPSED_TIME       0x0008
#define wxPD_ESTIMATED_TIME     0x0010
#define wxPD_SMOOTH             0x0020
#define wxPD_REMAINING_TIME     0x0040
#define wxPD_CAN_SKIP           0x0080


#include "wx/generic/progdlgg.h"

#if defined(__WXMSW__) && !defined(__WXUNIVERSAL__)
    // The native implementation requires the use of threads and still has some
    // problems, so it can be explicitly disabled.
    #if wxUSE_THREADS && wxUSE_NATIVE_PROGRESSDLG
        #define wxHAS_NATIVE_PROGRESSDIALOG
        #include "wx/msw/progdlg.h"
    #endif
#endif

// If there is no native one, just use the generic version.
#ifndef wxHAS_NATIVE_PROGRESSDIALOG
    class WXDLLIMPEXP_CORE wxProgressDialog
                           : public wxGenericProgressDialog
    {
    public:
        wxProgressDialog( const wxString& title, const wxString& message,
                          int maximum = 100,
                          wxWindow *parent = NULL,
                          int style = wxPD_APP_MODAL | wxPD_AUTO_HIDE )
            : wxGenericProgressDialog( title, message, maximum,
                                       parent, style )
            { }

    private:
        wxDECLARE_DYNAMIC_CLASS_NO_COPY( wxProgressDialog );
    };
#endif // !wxHAS_NATIVE_PROGRESSDIALOG

#endif // wxUSE_PROGRESSDLG

#endif // _WX_PROGDLG_H_BASE_
