///////////////////////////////////////////////////////////////////////////////
// Name:        wx/datectrl.h
// Purpose:     implements wxDatePickerCtrl
// Author:      Vadim Zeitlin
// Modified by:
// Created:     2005-01-09
// Copyright:   (c) 2005 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_DATECTRL_H_
#define _WX_DATECTRL_H_

#include "wx/defs.h"

#if wxUSE_DATEPICKCTRL

#include "wx/datetimectrl.h"    // the base class

#define wxDatePickerCtrlNameStr wxT("datectrl")

// wxDatePickerCtrl styles
enum
{
    // default style on this platform, either wxDP_SPIN or wxDP_DROPDOWN
    wxDP_DEFAULT = 0,

    // a spin control-like date picker (not supported in generic version)
    wxDP_SPIN = 1,

    // a combobox-like date picker (not supported in mac version)
    wxDP_DROPDOWN = 2,

    // always show century in the default date display (otherwise it depends on
    // the system date format which may include the century or not)
    wxDP_SHOWCENTURY = 4,

    // allow not having any valid date in the control (by default it always has
    // some date, today initially if no valid date specified in ctor)
    wxDP_ALLOWNONE = 8
};

// ----------------------------------------------------------------------------
// wxDatePickerCtrl: allow the user to enter the date
// ----------------------------------------------------------------------------

// The template argument must be a class deriving from wxDateTimePickerCtrlBase
// (i.e. in practice either this class itself or wxDateTimePickerCtrl).
template <typename Base>
class WXDLLIMPEXP_ADV wxDatePickerCtrlCommonBase : public Base
{
public:
    /*
       The derived classes should implement ctor and Create() method with the
       following signature:

        bool Create(wxWindow *parent,
                    wxWindowID id,
                    const wxDateTime& dt = wxDefaultDateTime,
                    const wxPoint& pos = wxDefaultPosition,
                    const wxSize& size = wxDefaultSize,
                    long style = wxDP_DEFAULT | wxDP_SHOWCENTURY,
                    const wxValidator& validator = wxDefaultValidator,
                    const wxString& name = wxDatePickerCtrlNameStr);
     */

    /*
        We inherit the methods to set/get the date from the base class.

        virtual void SetValue(const wxDateTime& dt) = 0;
        virtual wxDateTime GetValue() const = 0;
    */

    // And add methods to set/get the allowed valid range for the dates. If
    // either/both of them are invalid, there is no corresponding limit and if
    // neither is set, GetRange() returns false.
    virtual void SetRange(const wxDateTime& dt1, const wxDateTime& dt2) = 0;
    virtual bool GetRange(wxDateTime *dt1, wxDateTime *dt2) const = 0;
};

// This class is defined mostly for compatibility and is used as the base class
// by native wxDatePickerCtrl implementations.
typedef wxDatePickerCtrlCommonBase<wxDateTimePickerCtrl> wxDatePickerCtrlBase;

#if defined(__WXMSW__) && !defined(__WXUNIVERSAL__)
    #include "wx/msw/datectrl.h"

    #define wxHAS_NATIVE_DATEPICKCTRL
#elif defined(__WXOSX_COCOA__) && !defined(__WXUNIVERSAL__)
    #include "wx/osx/datectrl.h"

    #define wxHAS_NATIVE_DATEPICKCTRL
#else
    #include "wx/generic/datectrl.h"

    class WXDLLIMPEXP_ADV wxDatePickerCtrl : public wxDatePickerCtrlGeneric
    {
    public:
        wxDatePickerCtrl() { }
        wxDatePickerCtrl(wxWindow *parent,
                         wxWindowID id,
                         const wxDateTime& date = wxDefaultDateTime,
                         const wxPoint& pos = wxDefaultPosition,
                         const wxSize& size = wxDefaultSize,
                         long style = wxDP_DEFAULT | wxDP_SHOWCENTURY,
                         const wxValidator& validator = wxDefaultValidator,
                         const wxString& name = wxDatePickerCtrlNameStr)
            : wxDatePickerCtrlGeneric(parent, id, date, pos, size, style, validator, name)
        {
        }

    private:
        wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxDatePickerCtrl);
    };
#endif

#endif // wxUSE_DATEPICKCTRL

#endif // _WX_DATECTRL_H_

