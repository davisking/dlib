/////////////////////////////////////////////////////////////////////////////
// Name:        wx/valtext.h
// Purpose:     wxTextValidator class
// Author:      Julian Smart
// Modified by: Francesco Montorsi
// Created:     29/01/98
// Copyright:   (c) 1998 Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_VALTEXT_H_
#define _WX_VALTEXT_H_

#include "wx/defs.h"

#if wxUSE_VALIDATORS && (wxUSE_TEXTCTRL || wxUSE_COMBOBOX)

class WXDLLIMPEXP_FWD_CORE wxTextEntry;

#include "wx/validate.h"

enum wxTextValidatorStyle
{
    wxFILTER_NONE = 0x0,
    wxFILTER_EMPTY = 0x1,
    wxFILTER_ASCII = 0x2,
    wxFILTER_ALPHA = 0x4,
    wxFILTER_ALPHANUMERIC = 0x8,
    wxFILTER_DIGITS = 0x10,
    wxFILTER_NUMERIC = 0x20,
    wxFILTER_INCLUDE_LIST = 0x40,
    wxFILTER_INCLUDE_CHAR_LIST = 0x80,
    wxFILTER_EXCLUDE_LIST = 0x100,
    wxFILTER_EXCLUDE_CHAR_LIST = 0x200,
    wxFILTER_XDIGITS = 0x400,
    wxFILTER_SPACE = 0x800,

    // filter char class (for internal use only)
    wxFILTER_CC = wxFILTER_SPACE|wxFILTER_ASCII|wxFILTER_NUMERIC|
                  wxFILTER_ALPHANUMERIC|wxFILTER_ALPHA|
                  wxFILTER_DIGITS|wxFILTER_XDIGITS
};

// ----------------------------------------------------------------------------
// wxTextValidator
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxTextValidator: public wxValidator
{
public:
    wxTextValidator(long style = wxFILTER_NONE, wxString *val = NULL);
    wxTextValidator(const wxTextValidator& val);

    virtual ~wxTextValidator(){}

    // Make a clone of this validator (or return NULL) - currently necessary
    // if you're passing a reference to a validator.
    // Another possibility is to always pass a pointer to a new validator
    // (so the calling code can use a copy constructor of the relevant class).
    virtual wxObject *Clone() const wxOVERRIDE { return new wxTextValidator(*this); }
    bool Copy(const wxTextValidator& val);

    // Called when the value in the window must be validated.
    // This function can pop up an error message.
    virtual bool Validate(wxWindow *parent) wxOVERRIDE;

    // Called to transfer data to the window
    virtual bool TransferToWindow() wxOVERRIDE;

    // Called to transfer data from the window
    virtual bool TransferFromWindow() wxOVERRIDE;

    // Filter keystrokes
    void OnChar(wxKeyEvent& event);

    // ACCESSORS
    inline long GetStyle() const { return m_validatorStyle; }
    void SetStyle(long style);

    wxTextEntry *GetTextEntry();

    // strings & chars inclusions:
    // ---------------------------

    void SetCharIncludes(const wxString& chars);
    void AddCharIncludes(const wxString& chars);

    void SetIncludes(const wxArrayString& includes);
    void AddInclude(const wxString& include);

    const wxArrayString& GetIncludes() const { return m_includes; }
    wxString GetCharIncludes() const { return m_charIncludes; }

    // strings & chars exclusions:
    // ---------------------------

    void SetCharExcludes(const wxString& chars);
    void AddCharExcludes(const wxString& chars);

    void SetExcludes(const wxArrayString& excludes);
    void AddExclude(const wxString& exclude);

    const wxArrayString& GetExcludes() const { return m_excludes; }
    wxString GetCharExcludes() const { return m_charExcludes; }

    bool HasFlag(wxTextValidatorStyle style) const
        { return (m_validatorStyle & style) != 0; }

    // implementation only
    // --------------------

    // returns the error message if the contents of 'str' are invalid
    virtual wxString IsValid(const wxString& str) const;

protected:

    bool IsCharIncluded(const wxUniChar& c) const
    {
        return m_charIncludes.find(c) != wxString::npos;
    }

    bool IsCharExcluded(const wxUniChar& c) const
    {
        return m_charExcludes.find(c) != wxString::npos;
    }

    bool IsIncluded(const wxString& str) const
    {
        if ( HasFlag(wxFILTER_INCLUDE_LIST) )
            return m_includes.Index(str) != wxNOT_FOUND;

        // m_includes should be ignored (i.e. return true)
        // if the style is not set.
        return true;
    }

    bool IsExcluded(const wxString& str) const
    {
        return m_excludes.Index(str) != wxNOT_FOUND;
    }

    // returns false if the character is invalid
    bool IsValidChar(const wxUniChar& c) const;

    // These two functions (undocumented now) are kept for compatibility reasons.
    bool ContainsOnlyIncludedCharacters(const wxString& val) const;
    bool ContainsExcludedCharacters(const wxString& val) const;

protected:
    long                 m_validatorStyle;
    wxString*            m_stringValue;
    wxString             m_charIncludes;
    wxString             m_charExcludes;
    wxArrayString        m_includes;
    wxArrayString        m_excludes;

private:
    wxDECLARE_NO_ASSIGN_CLASS(wxTextValidator);
    wxDECLARE_DYNAMIC_CLASS(wxTextValidator);
    wxDECLARE_EVENT_TABLE();
};

#endif
  // wxUSE_VALIDATORS && (wxUSE_TEXTCTRL || wxUSE_COMBOBOX)

#endif // _WX_VALTEXT_H_
