/////////////////////////////////////////////////////////////////////////////
// Name:        wx/valnum.h
// Purpose:     Numeric validator classes.
// Author:      Vadim Zeitlin based on the submission of Fulvio Senore
// Created:     2010-11-06
// Copyright:   (c) 2010 wxWidgets team
//              (c) 2011 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_VALNUM_H_
#define _WX_VALNUM_H_

#include "wx/defs.h"

#if wxUSE_VALIDATORS

#include "wx/textentry.h"
#include "wx/validate.h"

// This header uses std::numeric_limits<>::min/max, but these symbols are,
// unfortunately, often defined as macros and the code here wouldn't compile in
// this case, so preventively undefine them to avoid this problem.
#undef min
#undef max

#include <limits>

// Bit masks used for numeric validator styles.
enum wxNumValidatorStyle
{
    wxNUM_VAL_DEFAULT               = 0x0,
    wxNUM_VAL_THOUSANDS_SEPARATOR   = 0x1,
    wxNUM_VAL_ZERO_AS_BLANK         = 0x2,
    wxNUM_VAL_NO_TRAILING_ZEROES    = 0x4
};

// ----------------------------------------------------------------------------
// Base class for all numeric validators.
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxNumValidatorBase : public wxValidator
{
public:
    // Change the validator style. Usually it's specified during construction.
    void SetStyle(int style) { m_style = style; }


    // Override base class method to not do anything but always return success:
    // we don't need this as we do our validation on the fly here.
    virtual bool Validate(wxWindow * WXUNUSED(parent)) wxOVERRIDE { return true; }

    // Override base class method to check that the window is a text control or
    // combobox.
    virtual void SetWindow(wxWindow *win) wxOVERRIDE;

protected:
    wxNumValidatorBase(int style)
    {
        m_style = style;
    }

    wxNumValidatorBase(const wxNumValidatorBase& other) : wxValidator(other)
    {
        m_style = other.m_style;
    }

    bool HasFlag(wxNumValidatorStyle style) const
    {
        return (m_style & style) != 0;
    }

    // Get the text entry of the associated control. Normally shouldn't ever
    // return NULL (and will assert if it does return it) but the caller should
    // still test the return value for safety.
    wxTextEntry *GetTextEntry() const;

    // Convert wxNUM_VAL_THOUSANDS_SEPARATOR and wxNUM_VAL_NO_TRAILING_ZEROES
    // bits of our style to the corresponding wxNumberFormatter::Style values.
    int GetFormatFlags() const;

    // Return the string which would result from inserting the given character
    // at the specified position.
    wxString GetValueAfterInsertingChar(wxString val, int pos, wxChar ch) const
    {
        val.insert(pos, ch);
        return val;
    }

    // Return true if this control allows negative numbers in it.
    //
    // If it doesn't, we don't allow entering "-" at all.
    virtual bool CanBeNegative() const = 0;

private:
    // Check whether the specified character can be inserted in the control at
    // the given position in the string representing the current controls
    // contents.
    //
    // Notice that the base class checks for '-' itself so it's never passed to
    // this function.
    virtual bool IsCharOk(const wxString& val, int pos, wxChar ch) const = 0;

    // NormalizeString the contents of the string if it's a valid number, return
    // empty string otherwise.
    virtual wxString NormalizeString(const wxString& s) const = 0;


    // Event handlers.
    void OnChar(wxKeyEvent& event);
    void OnKillFocus(wxFocusEvent& event);


    // Determine the current insertion point and text in the associated control.
    void GetCurrentValueAndInsertionPoint(wxString& val, int& pos) const;

    // Return true if pressing a '-' key is acceptable for the current control
    // contents and insertion point. This is used by OnChar() to handle '-' and
    // relies on CanBeNegative() implementation in the derived class.
    bool IsMinusOk(const wxString& val, int pos) const;


    // Combination of wxVAL_NUM_XXX values.
    int m_style;


    wxDECLARE_EVENT_TABLE();

    wxDECLARE_NO_ASSIGN_CLASS(wxNumValidatorBase);
};

namespace wxPrivate
{

// This is a helper class used by wxIntegerValidator and wxFloatingPointValidator
// below that implements Transfer{To,From}Window() adapted to the type of the
// variable.
//
// The template argument B is the name of the base class which must derive from
// wxNumValidatorBase and define LongestValueType type and {To,As}String()
// methods i.e. basically be one of wx{Integer,Number}ValidatorBase classes.
//
// The template argument T is just the type handled by the validator that will
// inherit from this one.
template <class B, typename T>
class wxNumValidator : public B
{
public:
    typedef B BaseValidator;
    typedef T ValueType;

    typedef typename BaseValidator::LongestValueType LongestValueType;

    wxCOMPILE_TIME_ASSERT
    (
        sizeof(ValueType) <= sizeof(LongestValueType),
        UnsupportedType
    );

    void SetMin(ValueType min)
    {
        m_min = min;
    }

    ValueType GetMin() const
    {
        return m_min;
    }

    void SetMax(ValueType max)
    {
        m_max = max;
    }

    ValueType GetMax() const
    {
        return m_max;
    }

    void SetRange(ValueType min, ValueType max)
    {
        SetMin(min);
        SetMax(max);
    }

    void GetRange(ValueType& min, ValueType& max) const
    {
        min = GetMin();
        max = GetMax();
    }

    virtual bool TransferToWindow()  wxOVERRIDE
    {
        if ( m_value )
        {
            wxTextEntry * const control = BaseValidator::GetTextEntry();
            if ( !control )
                return false;

            control->SetValue(NormalizeValue(static_cast<LongestValueType>(*m_value)));
        }

        return true;
    }

    virtual bool TransferFromWindow() wxOVERRIDE
    {
        if ( m_value )
        {
            wxTextEntry * const control = BaseValidator::GetTextEntry();
            if ( !control )
                return false;

            const wxString s(control->GetValue());
            LongestValueType value;
            if ( s.empty() && BaseValidator::HasFlag(wxNUM_VAL_ZERO_AS_BLANK) )
                value = 0;
            else if ( !BaseValidator::FromString(s, &value) )
                return false;

            if ( !this->IsInRange(value) )
                return false;

            *m_value = static_cast<ValueType>(value);
        }

        return true;
    }

protected:
    wxNumValidator(ValueType *value, int style)
        : BaseValidator(style),
          m_value(value)
    {
    }

    // Implement wxNumValidatorBase virtual method which is the same for
    // both integer and floating point numbers.
    virtual wxString NormalizeString(const wxString& s) const wxOVERRIDE
    {
        LongestValueType value;
        return BaseValidator::FromString(s, &value) ? NormalizeValue(value)
                                                    : wxString();
    }

    virtual bool CanBeNegative() const wxOVERRIDE { return m_min < 0; }


    // This member is protected because it can be useful to the derived classes
    // in their Transfer{From,To}Window() implementations.
    ValueType * const m_value;

private:
    // Just a helper which is a common part of TransferToWindow() and
    // NormalizeString(): returns string representation of a number honouring
    // wxNUM_VAL_ZERO_AS_BLANK flag.
    wxString NormalizeValue(LongestValueType value) const
    {
        // We really want to compare with the exact 0 here, so disable gcc
        // warning about doing this.
        wxGCC_WARNING_SUPPRESS(float-equal)

        wxString s;
        if ( value != 0 || !BaseValidator::HasFlag(wxNUM_VAL_ZERO_AS_BLANK) )
            s = this->ToString(value);

        wxGCC_WARNING_RESTORE(float-equal)

        return s;
    }

    // Minimal and maximal values accepted (inclusive).
    ValueType m_min, m_max;

    wxDECLARE_NO_ASSIGN_CLASS(wxNumValidator);
};

} // namespace wxPrivate

// ----------------------------------------------------------------------------
// Validators for integer numbers.
// ----------------------------------------------------------------------------

// Base class for integer numbers validator. This class contains all non
// type-dependent code of wxIntegerValidator<> and always works with values of
// type LongestValueType. It is not meant to be used directly, please use
// wxIntegerValidator<> only instead.
class WXDLLIMPEXP_CORE wxIntegerValidatorBase : public wxNumValidatorBase
{
protected:
    // Define the type we use here, it should be the maximal-sized integer type
    // we support to make it possible to base wxIntegerValidator<> for any type
    // on it.
#ifdef wxLongLong_t
    typedef wxLongLong_t LongestValueType;
    typedef wxULongLong_t ULongestValueType;
#else
    typedef long LongestValueType;
    typedef unsigned long ULongestValueType;
#endif

    wxIntegerValidatorBase(int style)
        : wxNumValidatorBase(style)
    {
        wxASSERT_MSG( !(style & wxNUM_VAL_NO_TRAILING_ZEROES),
                      "This style doesn't make sense for integers." );
    }

    // Default copy ctor is ok.

    // Provide methods for wxNumValidator use.
    wxString ToString(LongestValueType value) const;
    bool FromString(const wxString& s, LongestValueType *value) const;

    virtual bool IsInRange(LongestValueType value) const = 0;

    // Implement wxNumValidatorBase pure virtual method.
    virtual bool IsCharOk(const wxString& val, int pos, wxChar ch) const wxOVERRIDE;

private:
    wxDECLARE_NO_ASSIGN_CLASS(wxIntegerValidatorBase);
};

// Validator for integer numbers. It can actually work with any integer type
// (short, int or long and long long if supported) and their unsigned versions
// as well.
template <typename T>
class wxIntegerValidator
    : public wxPrivate::wxNumValidator<wxIntegerValidatorBase, T>
{
public:
    typedef T ValueType;

    typedef
        wxPrivate::wxNumValidator<wxIntegerValidatorBase, T> Base;
    typedef
        wxIntegerValidatorBase::LongestValueType LongestValueType;

    // Ctor for an integer validator.
    //
    // Sets the range appropriately for the type, including setting 0 as the
    // minimal value for the unsigned types.
    wxIntegerValidator(ValueType *value = NULL, int style = wxNUM_VAL_DEFAULT)
        : Base(value, style)
    {
        this->SetMin(std::numeric_limits<ValueType>::min());
        this->SetMax(std::numeric_limits<ValueType>::max());
    }

    virtual wxObject *Clone() const wxOVERRIDE { return new wxIntegerValidator(*this); }

    virtual bool IsInRange(LongestValueType value) const wxOVERRIDE
    {
        // LongestValueType is used as a container for the values of any type
        // which can be used in type-independent wxIntegerValidatorBase code,
        // but we need to use the correct type for comparisons, notably for
        // comparing unsigned values correctly, so cast to this type and check
        // that we don't lose precision while doing it.
        const ValueType valueT = static_cast<ValueType>(value);
        if ( static_cast<LongestValueType>(valueT) != value )
        {
            // The conversion wasn't lossless, so the value must not be exactly
            // representable in this type and so is definitely not in range.
            return false;
        }

        return this->GetMin() <= valueT && valueT <= this->GetMax();
    }

private:
    wxDECLARE_NO_ASSIGN_CLASS(wxIntegerValidator);
};

// Helper function for creating integer validators which allows to avoid
// explicitly specifying the type as it deduces it from its parameter.
template <typename T>
inline wxIntegerValidator<T>
wxMakeIntegerValidator(T *value, int style = wxNUM_VAL_DEFAULT)
{
    return wxIntegerValidator<T>(value, style);
}

// ----------------------------------------------------------------------------
// Validators for floating point numbers.
// ----------------------------------------------------------------------------

// Similar to wxIntegerValidatorBase, this class is not meant to be used
// directly, only wxFloatingPointValidator<> should be used in the user code.
class WXDLLIMPEXP_CORE wxFloatingPointValidatorBase : public wxNumValidatorBase
{
public:
    // Set precision i.e. the number of digits shown (and accepted on input)
    // after the decimal point. By default this is set to the maximal precision
    // supported by the type handled by the validator.
    void SetPrecision(unsigned precision) { m_precision = precision; }

    // Set multiplier applied for displaying the value, e.g. 100 if the value
    // should be displayed in percents, so that the variable containing 0.5
    // would be displayed as 50.
    void SetFactor(double factor) { m_factor = factor; }

protected:
    // Notice that we can't use "long double" here because it's not supported
    // by wxNumberFormatter yet, so restrict ourselves to just double (and
    // float).
    typedef double LongestValueType;

    wxFloatingPointValidatorBase(int style)
        : wxNumValidatorBase(style)
    {
        m_factor = 1.0;
    }

    // Default copy ctor is ok.

    // Provide methods for wxNumValidator use.
    wxString ToString(LongestValueType value) const;
    bool FromString(const wxString& s, LongestValueType *value) const;

    virtual bool IsInRange(LongestValueType value) const = 0;

    // Implement wxNumValidatorBase pure virtual method.
    virtual bool IsCharOk(const wxString& val, int pos, wxChar ch) const wxOVERRIDE;

private:
    // Maximum number of decimals digits after the decimal separator.
    unsigned m_precision;

    // Factor applied for the displayed the value.
    double m_factor;

    wxDECLARE_NO_ASSIGN_CLASS(wxFloatingPointValidatorBase);
};

// Validator for floating point numbers. It can be used with float, double or
// long double values.
template <typename T>
class wxFloatingPointValidator
    : public wxPrivate::wxNumValidator<wxFloatingPointValidatorBase, T>
{
public:
    typedef T ValueType;
    typedef wxPrivate::wxNumValidator<wxFloatingPointValidatorBase, T> Base;
    typedef wxFloatingPointValidatorBase::LongestValueType LongestValueType;


    // Ctor using implicit (maximal) precision for this type.
    wxFloatingPointValidator(ValueType *value = NULL,
                             int style = wxNUM_VAL_DEFAULT)
        : Base(value, style)
    {
        DoSetMinMax();

        this->SetPrecision(std::numeric_limits<ValueType>::digits10);
    }

    // Ctor specifying an explicit precision.
    wxFloatingPointValidator(int precision,
                      ValueType *value = NULL,
                      int style = wxNUM_VAL_DEFAULT)
        : Base(value, style)
    {
        DoSetMinMax();

        this->SetPrecision(precision);
    }

    virtual wxObject *Clone() const wxOVERRIDE
    {
        return new wxFloatingPointValidator(*this);
    }

    virtual bool IsInRange(LongestValueType value) const wxOVERRIDE
    {
        const ValueType valueT = static_cast<ValueType>(value);

        return this->GetMin() <= valueT && valueT <= this->GetMax();
    }

private:
    void DoSetMinMax()
    {
        // NB: Do not use min(), it's not the smallest representable value for
        //     the floating point types but rather the smallest representable
        //     positive value.
        this->SetMin(-std::numeric_limits<ValueType>::max());
        this->SetMax( std::numeric_limits<ValueType>::max());
    }
};

// Helper similar to wxMakeIntValidator().
//
// NB: Unfortunately we can't just have a wxMakeNumericValidator() which would
//     return either wxIntegerValidator<> or wxFloatingPointValidator<> so we
//     do need two different functions.
template <typename T>
inline wxFloatingPointValidator<T>
wxMakeFloatingPointValidator(T *value, int style = wxNUM_VAL_DEFAULT)
{
    return wxFloatingPointValidator<T>(value, style);
}

template <typename T>
inline wxFloatingPointValidator<T>
wxMakeFloatingPointValidator(int precision, T *value, int style = wxNUM_VAL_DEFAULT)
{
    return wxFloatingPointValidator<T>(precision, value, style);
}

#endif // wxUSE_VALIDATORS

#endif // _WX_VALNUM_H_
