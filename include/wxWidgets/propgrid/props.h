/////////////////////////////////////////////////////////////////////////////
// Name:        wx/propgrid/props.h
// Purpose:     wxPropertyGrid Property Classes
// Author:      Jaakko Salli
// Modified by:
// Created:     2007-03-28
// Copyright:   (c) Jaakko Salli
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PROPGRID_PROPS_H_
#define _WX_PROPGRID_PROPS_H_

#include "wx/defs.h"

#if wxUSE_PROPGRID

// -----------------------------------------------------------------------

class WXDLLIMPEXP_FWD_PROPGRID wxPGArrayEditorDialog;

#include "wx/propgrid/property.h"

#include "wx/filename.h"
#include "wx/dialog.h"
#include "wx/textctrl.h"
#include "wx/valtext.h"

// -----------------------------------------------------------------------

//
// Property class implementation helper macros.
//

#define wxPG_IMPLEMENT_PROPERTY_CLASS(NAME, UPCLASS, EDITOR) \
wxIMPLEMENT_DYNAMIC_CLASS(NAME, UPCLASS); \
wxPG_IMPLEMENT_PROPERTY_CLASS_PLAIN(NAME, EDITOR)

#if WXWIN_COMPATIBILITY_3_0
// This macro is deprecated. Use wxPG_IMPLEMENT_PROPERTY_CLASS instead.
#define WX_PG_IMPLEMENT_PROPERTY_CLASS(NAME, UPCLASS, T, T_AS_ARG, EDITOR) \
wxIMPLEMENT_DYNAMIC_CLASS(NAME, UPCLASS); \
WX_PG_IMPLEMENT_PROPERTY_CLASS_PLAIN(NAME, T, EDITOR)
#endif // WXWIN_COMPATIBILITY_3_0

// -----------------------------------------------------------------------

//
// These macros help creating DoGetValidator
#define WX_PG_DOGETVALIDATOR_ENTRY() \
    static wxValidator* s_ptr = NULL; \
    if ( s_ptr ) return s_ptr;

// Common function exit
#define WX_PG_DOGETVALIDATOR_EXIT(VALIDATOR) \
    s_ptr = VALIDATOR; \
    wxPGGlobalVars->m_arrValidators.push_back( VALIDATOR ); \
    return VALIDATOR;

// -----------------------------------------------------------------------

// Creates and manages a temporary wxTextCtrl for validation purposes.
// Uses wxPropertyGrid's current editor, if available.
class WXDLLIMPEXP_PROPGRID wxPGInDialogValidator
{
public:
    wxPGInDialogValidator()
    {
        m_textCtrl = NULL;
    }

    ~wxPGInDialogValidator()
    {
        if ( m_textCtrl )
            m_textCtrl->Destroy();
    }

    bool DoValidate( wxPropertyGrid* propGrid,
                     wxValidator* validator,
                     const wxString& value );

private:
    wxTextCtrl*         m_textCtrl;
};


// -----------------------------------------------------------------------
// Property classes
// -----------------------------------------------------------------------

#define wxPG_PROP_PASSWORD  wxPG_PROP_CLASS_SPECIFIC_2

// Basic property with string value.
// If value "<composed>" is set, then actual value is formed (or composed)
// from values of child properties.
class WXDLLIMPEXP_PROPGRID wxStringProperty : public wxPGProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxStringProperty)
public:
    wxStringProperty( const wxString& label = wxPG_LABEL,
                      const wxString& name = wxPG_LABEL,
                      const wxString& value = wxEmptyString );
    virtual ~wxStringProperty();

    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;

    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;

    // This is updated so "<composed>" special value can be handled.
    virtual void OnSetValue() wxOVERRIDE;

protected:
};

// -----------------------------------------------------------------------

// Constants used with NumericValidation<>().
enum wxPGNumericValidationConstants
{
    // Instead of modifying the value, show an error message.
    wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE      = 0,

    // Modify value, but stick with the limitations.
    wxPG_PROPERTY_VALIDATION_SATURATE           = 1,

    // Modify value, wrap around on overflow.
    wxPG_PROPERTY_VALIDATION_WRAP               = 2
};

// -----------------------------------------------------------------------

#if wxUSE_VALIDATORS

// A more comprehensive numeric validator class.
class WXDLLIMPEXP_PROPGRID wxNumericPropertyValidator : public wxTextValidator
{
public:
    enum NumericType
    {
        Signed = 0,
        Unsigned,
        Float
    };

    wxNumericPropertyValidator( NumericType numericType, int base = 10 );
    virtual ~wxNumericPropertyValidator() { }
    virtual bool Validate(wxWindow* parent) wxOVERRIDE;
};

#endif // wxUSE_VALIDATORS

// Base class for numeric properties.
// Cannot be instantiated directly.
class WXDLLIMPEXP_PROPGRID wxNumericProperty : public wxPGProperty
{
    wxDECLARE_ABSTRACT_CLASS(wxNumericProperty);
public:
    virtual ~wxNumericProperty();

    virtual bool DoSetAttribute(const wxString& name, wxVariant& value) wxOVERRIDE;

    virtual wxVariant AddSpinStepValue(long stepScale) const = 0;

    bool UseSpinMotion() const { return m_spinMotion; }

    // Common validation code - for internal use.
    template<typename T>
    bool DoNumericValidation(T& value, wxPGValidationInfo* pValidationInfo,
                             int mode, T defMin, T defMax) const;

protected:
    wxNumericProperty(const wxString& label, const wxString& name);

    wxVariant m_minVal;
    wxVariant m_maxVal;
    bool      m_spinMotion;
    wxVariant m_spinStep;
    bool      m_spinWrap;
};

// Basic property with integer value.
// Seamlessly supports 64-bit integer (wxLongLong) on overflow.
class WXDLLIMPEXP_PROPGRID wxIntProperty : public wxNumericProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxIntProperty)
public:
    wxIntProperty( const wxString& label = wxPG_LABEL,
                   const wxString& name = wxPG_LABEL,
                   long value = 0 );
    virtual ~wxIntProperty();

#if wxUSE_LONGLONG
    wxIntProperty( const wxString& label,
                   const wxString& name,
                   const wxLongLong& value );
#endif
    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool ValidateValue( wxVariant& value,
                                wxPGValidationInfo& validationInfo ) const wxOVERRIDE;
    virtual bool IntToValue( wxVariant& variant,
                             int number,
                             int argFlags = 0 ) const wxOVERRIDE;
    static wxValidator* GetClassValidator();
    virtual wxValidator* DoGetValidator() const wxOVERRIDE;
    virtual wxVariant AddSpinStepValue(long stepScale) const wxOVERRIDE;

private:
    // Validation helpers.
#if wxUSE_LONGLONG
    static bool DoValidation( const wxNumericProperty* property,
                              wxLongLong& value,
                              wxPGValidationInfo* pValidationInfo,
                              int mode =
                                wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE );

#if defined(wxLongLong_t)
    static bool DoValidation( const wxNumericProperty* property,
                              wxLongLong_t& value,
                              wxPGValidationInfo* pValidationInfo,
                              int mode =
                                wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE );
#endif // wxLongLong_t
#endif // wxUSE_LONGLONG
    static bool DoValidation(const wxNumericProperty* property,
                             long& value,
                             wxPGValidationInfo* pValidationInfo,
                             int mode =
                                wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE);
};

// -----------------------------------------------------------------------

// Basic property with unsigned integer value.
// Seamlessly supports 64-bit integer (wxULongLong) on overflow.
class WXDLLIMPEXP_PROPGRID wxUIntProperty : public wxNumericProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxUIntProperty)
public:
    wxUIntProperty( const wxString& label = wxPG_LABEL,
                    const wxString& name = wxPG_LABEL,
                    unsigned long value = 0 );
    virtual ~wxUIntProperty();
#if wxUSE_LONGLONG
    wxUIntProperty( const wxString& label,
                    const wxString& name,
                    const wxULongLong& value );
#endif
    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;
    virtual bool ValidateValue( wxVariant& value,
                                wxPGValidationInfo& validationInfo ) const wxOVERRIDE;
    virtual wxValidator* DoGetValidator () const wxOVERRIDE;
    virtual bool IntToValue( wxVariant& variant,
                             int number,
                             int argFlags = 0 ) const wxOVERRIDE;
    virtual wxVariant AddSpinStepValue(long stepScale) const wxOVERRIDE;

protected:
    wxByte      m_base;
    wxByte      m_realBase; // translated to 8,16,etc.
    wxByte      m_prefix;
private:
    void Init();

    // Validation helpers.
#if wxUSE_LONGLONG
    static bool DoValidation(const wxNumericProperty* property,
                             wxULongLong& value,
                             wxPGValidationInfo* pValidationInfo,
                             int mode =wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE);
#if defined(wxULongLong_t)
    static bool DoValidation(const wxNumericProperty* property,
                             wxULongLong_t& value,
                             wxPGValidationInfo* pValidationInfo,
                             int mode =wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE);
#endif // wxULongLong_t
#endif // wxUSE_LONGLONG
    static bool DoValidation(const wxNumericProperty* property,
                             long& value,
                             wxPGValidationInfo* pValidationInfo,
                             int mode = wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE);
};

// -----------------------------------------------------------------------

// Basic property with double-precision floating point value.
class WXDLLIMPEXP_PROPGRID wxFloatProperty : public wxNumericProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxFloatProperty)
public:
    wxFloatProperty( const wxString& label = wxPG_LABEL,
                     const wxString& name = wxPG_LABEL,
                     double value = 0.0 );
    virtual ~wxFloatProperty();

    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;

    virtual bool ValidateValue( wxVariant& value,
                                wxPGValidationInfo& validationInfo ) const wxOVERRIDE;

    static wxValidator* GetClassValidator();
    virtual wxValidator* DoGetValidator () const wxOVERRIDE;
    virtual wxVariant AddSpinStepValue(long stepScale) const wxOVERRIDE;

protected:
    int m_precision;

private:
    // Validation helper.
    static bool DoValidation(const wxNumericProperty* property,
                             double& value,
                             wxPGValidationInfo* pValidationInfo,
                             int mode = wxPG_PROPERTY_VALIDATION_ERROR_MESSAGE);
};

// -----------------------------------------------------------------------

// Basic property with boolean value.
class WXDLLIMPEXP_PROPGRID wxBoolProperty : public wxPGProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxBoolProperty)
public:
    wxBoolProperty( const wxString& label = wxPG_LABEL,
                    const wxString& name = wxPG_LABEL,
                    bool value = false );
    virtual ~wxBoolProperty();

    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool IntToValue( wxVariant& variant,
                             int number, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;
};

// -----------------------------------------------------------------------

// If set, then selection of choices is static and should not be
// changed (i.e. returns NULL in GetPropertyChoices).
#define wxPG_PROP_STATIC_CHOICES    wxPG_PROP_CLASS_SPECIFIC_1

// Represents a single selection from a list of choices
// You can derive custom properties with choices from this class.
// Updating private index is important. You can do this either by calling
// SetIndex() in IntToValue, and then letting wxBaseEnumProperty::OnSetValue
// be called (by not implementing it, or by calling super class function in
// it) -OR- you can just call SetIndex in OnSetValue.
class WXDLLIMPEXP_PROPGRID wxEnumProperty : public wxPGProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxEnumProperty)
public:

#ifndef SWIG
    wxEnumProperty( const wxString& label = wxPG_LABEL,
                    const wxString& name = wxPG_LABEL,
                    const wxChar* const* labels = NULL,
                    const long* values = NULL,
                    int value = 0 );
    wxEnumProperty( const wxString& label,
                    const wxString& name,
                    wxPGChoices& choices,
                    int value = 0 );

    // Special constructor for caching choices (used by derived class)
    wxEnumProperty( const wxString& label,
                    const wxString& name,
                    const char* const* untranslatedLabels,
                    const long* values,
                    wxPGChoices* choicesCache,
                    int value = 0 );

    wxEnumProperty( const wxString& label,
                    const wxString& name,
                    const wxArrayString& labels,
                    const wxArrayInt& values = wxArrayInt(),
                    int value = 0 );
#else
    wxEnumProperty( const wxString& label = wxPG_LABEL,
                    const wxString& name = wxPG_LABEL,
                    const wxArrayString& labels = wxArrayString(),
                    const wxArrayInt& values = wxArrayInt(),
                    int value = 0 );
#endif

    virtual ~wxEnumProperty();

    size_t GetItemCount() const { return m_choices.GetCount(); }

    virtual void OnSetValue() wxOVERRIDE;
    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool ValidateValue( wxVariant& value,
                                wxPGValidationInfo& validationInfo ) const wxOVERRIDE;

    // If wxPG_FULL_VALUE is not set in flags, then the value is interpreted
    // as index to choices list. Otherwise, it is actual value.
    virtual bool IntToValue( wxVariant& variant,
                             int number,
                             int argFlags = 0 ) const wxOVERRIDE;

    //
    // Additional virtuals

    // This must be overridden to have non-index based value
    virtual int GetIndexForValue( int value ) const;

    // GetChoiceSelection needs to overridden since m_index is
    // the true index, and various property classes derived from
    // this take advantage of it.
    virtual int GetChoiceSelection() const wxOVERRIDE { return m_index; }

protected:

    int GetIndex() const;
    void SetIndex( int index );

#if WXWIN_COMPATIBILITY_3_0
    wxDEPRECATED_MSG("use ValueFromString_(wxVariant&, int*, const wxString&, int) function instead")
    bool ValueFromString_( wxVariant& value,
                           const wxString& text,
                           int argFlags ) const
    {
        return ValueFromString_(value, NULL, text, argFlags);
    }
    wxDEPRECATED_MSG("use ValueFromInt_(wxVariant&, int*, int, int) function instead")
    bool ValueFromInt_( wxVariant& value, int intVal, int argFlags ) const
    {
        return ValueFromInt_(value, NULL, intVal, argFlags);
    }
    wxDEPRECATED_MSG("don't use ResetNextIndex() function")
    static void ResetNextIndex() { }
#endif
    // Converts text to value and returns corresponding index in the collection
    bool ValueFromString_(wxVariant& value,
                          int* pIndex,
                          const wxString& text,
                          int argFlags) const;
    // Converts number to value and returns corresponding index in the collection
    bool ValueFromInt_(wxVariant& value, int* pIndex, int intVal, int argFlags) const;

private:
    // This is private so that classes are guaranteed to use GetIndex
    // for up-to-date index value.
    int                     m_index;
};

// -----------------------------------------------------------------------

// wxEnumProperty with wxString value and writable combo box editor.
// Uses int value, similar to wxEnumProperty, unless text entered by user is
// is not in choices (in which case string value is used).
class WXDLLIMPEXP_PROPGRID wxEditEnumProperty : public wxEnumProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxEditEnumProperty)
public:

    wxEditEnumProperty( const wxString& label,
                        const wxString& name,
                        const wxChar* const* labels,
                        const long* values,
                        const wxString& value );
    wxEditEnumProperty( const wxString& label = wxPG_LABEL,
                        const wxString& name = wxPG_LABEL,
                        const wxArrayString& labels = wxArrayString(),
                        const wxArrayInt& values = wxArrayInt(),
                        const wxString& value = wxEmptyString );
    wxEditEnumProperty( const wxString& label,
                        const wxString& name,
                        wxPGChoices& choices,
                        const wxString& value = wxEmptyString );

    // Special constructor for caching choices (used by derived class)
    wxEditEnumProperty( const wxString& label,
                        const wxString& name,
                        const char* const* untranslatedLabels,
                        const long* values,
                        wxPGChoices* choicesCache,
                        const wxString& value );

    virtual ~wxEditEnumProperty();

    void OnSetValue() wxOVERRIDE;
    bool StringToValue(wxVariant& variant,
                       const wxString& text,
                       int argFlags = 0) const wxOVERRIDE;
    bool ValidateValue(wxVariant& value,
                       wxPGValidationInfo& validationInfo) const wxOVERRIDE;

protected:
};

// -----------------------------------------------------------------------

// Represents a bit set that fits in a long integer. wxBoolProperty
// sub-properties are created for editing individual bits. Textctrl is created
// to manually edit the flags as a text; a continuous sequence of spaces,
// commas and semicolons is considered as a flag id separator.
// Note: When changing "choices" (ie. flag labels) of wxFlagsProperty,
// you will need to use SetPropertyChoices - otherwise they will not get
//    updated properly.
class WXDLLIMPEXP_PROPGRID wxFlagsProperty : public wxPGProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxFlagsProperty)
public:

#ifndef SWIG
    wxFlagsProperty( const wxString& label,
                     const wxString& name,
                     const wxChar* const* labels,
                     const long* values = NULL,
                     long value = 0 );
    wxFlagsProperty( const wxString& label,
                     const wxString& name,
                     wxPGChoices& choices,
                     long value = 0 );
#endif
    wxFlagsProperty( const wxString& label = wxPG_LABEL,
                     const wxString& name = wxPG_LABEL,
                     const wxArrayString& labels = wxArrayString(),
                     const wxArrayInt& values = wxArrayInt(),
                     int value = 0 );
    virtual ~wxFlagsProperty ();

    virtual void OnSetValue() wxOVERRIDE;
    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int flags ) const wxOVERRIDE;
    virtual wxVariant ChildChanged( wxVariant& thisValue,
                                    int childIndex,
                                    wxVariant& childValue ) const wxOVERRIDE;
    virtual void RefreshChildren() wxOVERRIDE;
    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;

    // GetChoiceSelection needs to overridden since m_choices is
    // used and value is integer, but it is not index.
    virtual int GetChoiceSelection() const wxOVERRIDE { return wxNOT_FOUND; }

    // helpers
    size_t GetItemCount() const { return m_choices.GetCount(); }
    const wxString& GetLabel( size_t ind ) const
        { return m_choices.GetLabel(static_cast<unsigned int>(ind)); }

protected:
    // Used to detect if choices have been changed
    wxPGChoicesData*        m_oldChoicesData;

    // Needed to properly mark changed sub-properties
    long                    m_oldValue;

    // Converts string id to a relevant bit.
    long IdToBit( const wxString& id ) const;

    // Creates children and sets value.
    void Init();
};

// -----------------------------------------------------------------------
class WXDLLIMPEXP_PROPGRID wxEditorDialogProperty : public wxPGProperty
{
    friend class wxPGDialogAdapter;
    wxDECLARE_ABSTRACT_CLASS(wxEditorDialogProperty);

public:
    virtual ~wxEditorDialogProperty();

    virtual wxPGEditorDialogAdapter* GetEditorDialog() const wxOVERRIDE;
    virtual bool DoSetAttribute(const wxString& name, wxVariant& value) wxOVERRIDE;

protected:
    wxEditorDialogProperty(const wxString& label, const wxString& name);

    virtual bool DisplayEditorDialog(wxPropertyGrid* pg, wxVariant& value) = 0;

    wxString  m_dlgTitle;  // Title for a dialog
    long      m_dlgStyle;  // Dialog style
};

// -----------------------------------------------------------------------

// Indicates first bit usable by derived properties.
#define wxPG_PROP_SHOW_FULL_FILENAME  wxPG_PROP_CLASS_SPECIFIC_1

// Like wxLongStringProperty, but the button triggers file selector instead.
class WXDLLIMPEXP_PROPGRID wxFileProperty : public wxEditorDialogProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxFileProperty)
public:

    wxFileProperty( const wxString& label = wxPG_LABEL,
                    const wxString& name = wxPG_LABEL,
                    const wxString& value = wxEmptyString );
    virtual ~wxFileProperty ();

    virtual void OnSetValue() wxOVERRIDE;
    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;

    static wxValidator* GetClassValidator();
    virtual wxValidator* DoGetValidator() const wxOVERRIDE;

    // Returns filename to file represented by current value.
    wxFileName GetFileName() const;

protected:
    virtual bool DisplayEditorDialog(wxPropertyGrid* pg, wxVariant& value) wxOVERRIDE;

    wxString    m_wildcard;
    wxString    m_basePath; // If set, then show path relative to it
    wxString    m_initialPath; // If set, start the file dialog here
    int         m_indFilter; // index to the selected filter
};

// -----------------------------------------------------------------------

// Flag used in wxLongStringProperty to mark that edit button
// should be enabled even in the read-only mode.
#define wxPG_PROP_ACTIVE_BTN    wxPG_PROP_CLASS_SPECIFIC_1

// Like wxStringProperty, but has a button that triggers a small text
// editor dialog.
class WXDLLIMPEXP_PROPGRID wxLongStringProperty : public wxEditorDialogProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxLongStringProperty)
public:

    wxLongStringProperty( const wxString& label = wxPG_LABEL,
                          const wxString& name = wxPG_LABEL,
                          const wxString& value = wxEmptyString );
    virtual ~wxLongStringProperty();

    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;

protected:
    virtual bool DisplayEditorDialog(wxPropertyGrid* pg, wxVariant& value) wxOVERRIDE;
};

// -----------------------------------------------------------------------


// Like wxLongStringProperty, but the button triggers dir selector instead.
class WXDLLIMPEXP_PROPGRID wxDirProperty : public wxEditorDialogProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxDirProperty)
public:
    wxDirProperty( const wxString& label = wxPG_LABEL,
                   const wxString& name = wxPG_LABEL,
                   const wxString& value = wxEmptyString );
    virtual ~wxDirProperty();

    virtual wxString ValueToString(wxVariant& value, int argFlags = 0) const wxOVERRIDE;
    virtual bool StringToValue(wxVariant& variant, const wxString& text,
                               int argFlags = 0) const wxOVERRIDE;
#if WXWIN_COMPATIBILITY_3_0
    virtual bool DoSetAttribute(const wxString& name, wxVariant& value) wxOVERRIDE;
#endif // WXWIN_COMPATIBILITY_3_0
    virtual wxValidator* DoGetValidator() const wxOVERRIDE;

protected:
    virtual bool DisplayEditorDialog(wxPropertyGrid* pg, wxVariant& value) wxOVERRIDE;
};

// -----------------------------------------------------------------------

// wxBoolProperty, wxFlagsProperty specific flags
#define wxPG_PROP_USE_CHECKBOX      wxPG_PROP_CLASS_SPECIFIC_1
// DCC = Double Click Cycles
#define wxPG_PROP_USE_DCC           wxPG_PROP_CLASS_SPECIFIC_2


// -----------------------------------------------------------------------

// Property that manages a list of strings.
class WXDLLIMPEXP_PROPGRID wxArrayStringProperty : public wxEditorDialogProperty
{
    WX_PG_DECLARE_PROPERTY_CLASS(wxArrayStringProperty)
public:
    wxArrayStringProperty( const wxString& label = wxPG_LABEL,
                           const wxString& name = wxPG_LABEL,
                           const wxArrayString& value = wxArrayString() );
    virtual ~wxArrayStringProperty();

    virtual void OnSetValue() wxOVERRIDE;
    virtual wxString ValueToString( wxVariant& value, int argFlags = 0 ) const wxOVERRIDE;
    virtual bool StringToValue( wxVariant& variant,
                                const wxString& text,
                                int argFlags = 0 ) const wxOVERRIDE;
    virtual bool DoSetAttribute( const wxString& name, wxVariant& value ) wxOVERRIDE;

    // Implement in derived class for custom array-to-string conversion.
    virtual void ConvertArrayToString(const wxArrayString& arr,
                                      wxString* pString,
                                      const wxUniChar& delimiter) const;

    // Shows string editor dialog. Value to be edited should be read from
    // value, and if dialog is not cancelled, it should be stored back and true
    // should be returned if that was the case.
    virtual bool OnCustomStringEdit( wxWindow* parent, wxString& value );

#if WXWIN_COMPATIBILITY_3_0
    // Helper.
    wxDEPRECATED_MSG("OnButtonClick() function is no longer used")
    virtual bool OnButtonClick( wxPropertyGrid* propgrid,
                                wxWindow* primary,
                                const wxChar* cbt );
#endif // WXWIN_COMPATIBILITY_3_0

    // Creates wxPGArrayEditorDialog for string editing. Called in OnButtonClick.
    virtual wxPGArrayEditorDialog* CreateEditorDialog();

    enum ConversionFlags
    {
        Escape          = 0x01,
        QuoteStrings    = 0x02
    };

    // Generates contents for string dst based on the contents of
    // wxArrayString src.
    static void ArrayStringToString( wxString& dst, const wxArrayString& src,
                                     wxUniChar delimiter, int flags );

protected:
    virtual bool DisplayEditorDialog(wxPropertyGrid* pg, wxVariant& value) wxOVERRIDE;

    // Previously this was to be implemented in derived class for array-to-
    // string conversion. Now you should implement ConvertValueToString()
    // instead.
    virtual void GenerateValueAsString();

    wxString        m_display; // Cache for displayed text.
    wxUniChar       m_delimiter;
    wxString        m_customBtnText;
};

// -----------------------------------------------------------------------

#define WX_PG_DECLARE_ARRAYSTRING_PROPERTY_WITH_VALIDATOR_WITH_DECL(PROPNAME, \
                                                                    DECL) \
DECL PROPNAME : public wxArrayStringProperty \
{ \
    WX_PG_DECLARE_PROPERTY_CLASS(PROPNAME) \
public: \
    PROPNAME( const wxString& label = wxPG_LABEL, \
              const wxString& name = wxPG_LABEL, \
              const wxArrayString& value = wxArrayString() ); \
    ~PROPNAME(); \
    virtual bool OnCustomStringEdit( wxWindow* parent, wxString& value ) wxOVERRIDE; \
    virtual wxValidator* DoGetValidator() const wxOVERRIDE; \
};

#define WX_PG_DECLARE_ARRAYSTRING_PROPERTY_WITH_VALIDATOR(PROPNAM) \
WX_PG_DECLARE_ARRAYSTRING_PROPERTY_WITH_VALIDATOR(PROPNAM, class)

#define WX_PG_IMPLEMENT_ARRAYSTRING_PROPERTY_WITH_VALIDATOR(PROPNAME, \
                                                            DELIMCHAR, \
                                                            CUSTBUTTXT) \
wxPG_IMPLEMENT_PROPERTY_CLASS(PROPNAME, wxArrayStringProperty, \
                               TextCtrlAndButton) \
PROPNAME::PROPNAME( const wxString& label, \
                    const wxString& name, \
                    const wxArrayString& value ) \
    : wxArrayStringProperty(label,name,value) \
{ \
    PROPNAME::GenerateValueAsString(); \
    m_delimiter = DELIMCHAR; \
    m_customBtnText = CUSTBUTTXT; \
} \
PROPNAME::~PROPNAME() { }

#define WX_PG_DECLARE_ARRAYSTRING_PROPERTY(PROPNAME) \
WX_PG_DECLARE_ARRAYSTRING_PROPERTY_WITH_VALIDATOR(PROPNAME)

#define WX_PG_DECLARE_ARRAYSTRING_PROPERTY_WITH_DECL(PROPNAME, DECL) \
WX_PG_DECLARE_ARRAYSTRING_PROPERTY_WITH_VALIDATOR_WITH_DECL(PROPNAME, DECL)

#define WX_PG_IMPLEMENT_ARRAYSTRING_PROPERTY(PROPNAME,DELIMCHAR,CUSTBUTTXT) \
WX_PG_IMPLEMENT_ARRAYSTRING_PROPERTY_WITH_VALIDATOR(PROPNAME, \
                                                    DELIMCHAR, \
                                                    CUSTBUTTXT) \
wxValidator* PROPNAME::DoGetValidator () const \
{ return NULL; }


// -----------------------------------------------------------------------
// wxPGArrayEditorDialog
// -----------------------------------------------------------------------

#if wxUSE_EDITABLELISTBOX

#include "wx/bmpbuttn.h"
#include "wx/editlbox.h"

#define wxAEDIALOG_STYLE \
    (wxDEFAULT_DIALOG_STYLE | wxRESIZE_BORDER | wxOK | wxCANCEL | wxCENTRE)

class WXDLLIMPEXP_PROPGRID wxPGArrayEditorDialog : public wxDialog
{
public:
    wxPGArrayEditorDialog();
    virtual ~wxPGArrayEditorDialog() { }

    void Init();

    bool Create( wxWindow *parent,
                 const wxString& message,
                 const wxString& caption,
                 long style = wxAEDIALOG_STYLE,
                 const wxPoint& pos = wxDefaultPosition,
                 const wxSize& sz = wxDefaultSize );

    void EnableCustomNewAction()
    {
        m_hasCustomNewAction = true;
    }

    void SetNewButtonText(const wxString& text)
    {
        m_customNewButtonText = text;
        if ( m_elb && m_elb->GetNewButton() )
        {
            m_elb->GetNewButton()->SetToolTip(text);
        }
    }

    // Set value modified by dialog.
    virtual void SetDialogValue( const wxVariant& WXUNUSED(value) )
    {
        wxFAIL_MSG(wxS("re-implement this member function in derived class"));
    }

    // Return value modified by dialog.
    virtual wxVariant GetDialogValue() const
    {
        wxFAIL_MSG(wxS("re-implement this member function in derived class"));
        return wxVariant();
    }

    // Override to return wxValidator to be used with the wxTextCtrl
    // in dialog. Note that the validator is used in the standard
    // wx way, i.e. it immediately prevents user from entering invalid
    // input.
    // Note: Dialog frees the validator.
    virtual wxValidator* GetTextCtrlValidator() const
    {
        return NULL;
    }

    // Returns true if array was actually modified
    bool IsModified() const { return m_modified; }

    // wxEditableListBox utilities
    int GetSelection() const;

    // implementation from now on
    void OnAddClick(wxCommandEvent& event);
    void OnDeleteClick(wxCommandEvent& event);
    void OnUpClick(wxCommandEvent& event);
    void OnDownClick(wxCommandEvent& event);
    void OnEndLabelEdit(wxListEvent& event);
    void OnBeginLabelEdit(wxListEvent& evt);
    void OnIdle(wxIdleEvent& event);

protected:
    wxEditableListBox*  m_elb;

    // These are used for focus repair
    wxWindow*           m_elbSubPanel;
    wxWindow*           m_lastFocused;

    // A new item, edited by user, is pending at this index.
    // It will be committed once list ctrl item editing is done.
    int             m_itemPendingAtIndex;

    bool            m_modified;
    bool            m_hasCustomNewAction;
    wxString        m_customNewButtonText;

    // These must be overridden - must return true on success.
    virtual wxString ArrayGet( size_t index ) = 0;
    virtual size_t ArrayGetCount() = 0;
    virtual bool ArrayInsert( const wxString& str, int index ) = 0;
    virtual bool ArraySet( size_t index, const wxString& str ) = 0;
    virtual void ArrayRemoveAt( int index ) = 0;
    virtual void ArraySwap( size_t first, size_t second ) = 0;
    virtual bool OnCustomNewAction(wxString* WXUNUSED(resString))
    {
        return false;
    }

private:
    wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxPGArrayEditorDialog);
    wxDECLARE_EVENT_TABLE();
};

#endif // wxUSE_EDITABLELISTBOX

// -----------------------------------------------------------------------
// wxPGArrayStringEditorDialog
// -----------------------------------------------------------------------

class WXDLLIMPEXP_PROPGRID
    wxPGArrayStringEditorDialog : public wxPGArrayEditorDialog
{
public:
    wxPGArrayStringEditorDialog();
    virtual ~wxPGArrayStringEditorDialog() { }

    void Init();

    virtual void SetDialogValue( const wxVariant& value ) wxOVERRIDE
    {
        m_array = value.GetArrayString();
    }

    virtual wxVariant GetDialogValue() const wxOVERRIDE
    {
        return m_array;
    }

    void SetCustomButton( const wxString& custBtText,
                          wxArrayStringProperty* pcc )
    {
        if ( !custBtText.empty() )
        {
            EnableCustomNewAction();
            SetNewButtonText(custBtText);
            m_pCallingClass = pcc;
        }
    }

    virtual bool OnCustomNewAction(wxString* resString) wxOVERRIDE;

protected:
    wxArrayString   m_array;

    wxArrayStringProperty*     m_pCallingClass;

    virtual wxString ArrayGet( size_t index ) wxOVERRIDE;
    virtual size_t ArrayGetCount() wxOVERRIDE;
    virtual bool ArrayInsert( const wxString& str, int index ) wxOVERRIDE;
    virtual bool ArraySet( size_t index, const wxString& str ) wxOVERRIDE;
    virtual void ArrayRemoveAt( int index ) wxOVERRIDE;
    virtual void ArraySwap( size_t first, size_t second ) wxOVERRIDE;

private:
    wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxPGArrayStringEditorDialog);
    wxDECLARE_EVENT_TABLE();
};

// -----------------------------------------------------------------------

#endif // wxUSE_PROPGRID

#endif // _WX_PROPGRID_PROPS_H_
