///////////////////////////////////////////////////////////////////////////////
// Name:        wx/secretstore.h
// Purpose:     Storing and retrieving secrets using OS-provided facilities.
// Author:      Vadim Zeitlin
// Created:     2016-05-27
// Copyright:   (c) 2016 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_SECRETSTORE_H_
#define _WX_SECRETSTORE_H_

#include "wx/defs.h"

#include "wx/string.h"

#if wxUSE_SECRETSTORE

// Initial version of wxSecretStore required passing user name to Load(), which
// didn't make much sense without support for multiple usernames per service,
// so the API was changed to load the username too. Test for this symbol to
// distinguish between the old and the new API, it wasn't defined before the
// API change.
#define wxHAS_SECRETSTORE_LOAD_USERNAME

class wxSecretStoreImpl;
class wxSecretValueImpl;

// ----------------------------------------------------------------------------
// Represents a secret value, e.g. a password string.
// ----------------------------------------------------------------------------

// This is an immutable value-like class which tries to ensure that the secret
// value will be wiped out from memory once it's not needed any more.
class WXDLLIMPEXP_BASE wxSecretValue
{
public:
    // Creates an empty secret value (not the same as an empty password).
    wxSecretValue() : m_impl(NULL) { }

    // Creates a secret value from the given data.
    wxSecretValue(size_t size, const void *data)
        : m_impl(NewImpl(size, data))
    {
    }

    // Creates a secret value from string.
    explicit wxSecretValue(const wxString& secret)
    {
        const wxScopedCharBuffer buf(secret.utf8_str());
        m_impl = NewImpl(buf.length(), buf.data());
    }

    wxSecretValue(const wxSecretValue& other);
    wxSecretValue& operator=(const wxSecretValue& other);

    ~wxSecretValue();

    // Check if a secret is not empty.
    bool IsOk() const { return m_impl != NULL; }

    // Compare with another secret.
    bool operator==(const wxSecretValue& other) const;
    bool operator!=(const wxSecretValue& other) const
    {
        return !(*this == other);
    }

    // Get the size, in bytes, of the secret data.
    size_t GetSize() const;

    // Get read-only access to the secret data.
    //
    // Don't assume it is NUL-terminated, use GetSize() instead.
    const void *GetData() const;

    // Get the secret data as a string.
    //
    // Notice that you may want to overwrite the string contents after using it
    // by calling WipeString().
    wxString GetAsString(const wxMBConv& conv = wxConvWhateverWorks) const;

    // Erase the given area of memory overwriting its presumably sensitive
    // content.
    static void Wipe(size_t size, void *data);

    // Overwrite the contents of the given wxString.
    static void WipeString(wxString& str);

private:
    // This method is implemented in platform-specific code and must return a
    // new heap-allocated object initialized with the given data.
    static wxSecretValueImpl* NewImpl(size_t size, const void *data);

    // This ctor is only used by wxSecretStore and takes ownership of the
    // provided existing impl pointer.
    explicit wxSecretValue(wxSecretValueImpl* impl) : m_impl(impl) { }

    wxSecretValueImpl* m_impl;

    friend class wxSecretStore;
};

// ----------------------------------------------------------------------------
// A collection of secrets, sometimes called a key chain.
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxSecretStore
{
public:
    // Returns the default secrets collection to use.
    //
    // Currently this is the only way to create a secret store object. In the
    // future we could add more factory functions to e.g. create non-persistent
    // stores or allow creating stores corresponding to the native facilities
    // being used (e.g. specify schema name under Linux or a SecKeychainRef
    // under OS X).
    static wxSecretStore GetDefault();

    // This class has no default ctor, use GetDefault() instead.

    // But it can be copied, a copy refers to the same store as the original.
    wxSecretStore(const wxSecretStore& store);

    // Dtor is not virtual, this class is not supposed to be derived from.
    ~wxSecretStore();


    // Check if this object is valid, i.e. can be used, and optionally fill in
    // the provided error message string if it isn't.
    bool IsOk(wxString* errmsg = NULL) const;


    // Store a username/password combination.
    //
    // The service name should be user readable and unique.
    //
    // If a secret with the same service name already exists, it will be
    // overwritten with the new value.
    //
    // Returns false after logging an error message if an error occurs,
    // otherwise returns true indicating that the secret has been stored.
    bool Save(const wxString& service,
              const wxString& username,
              const wxSecretValue& password);

    // Look up the username/password for the given service.
    //
    // If no username/password is found for the given service, false is
    // returned.
    //
    // Otherwise the function returns true and updates the provided user name
    // and password arguments.
    bool Load(const wxString& service,
              wxString& username,
              wxSecretValue& password) const;

    // Delete a previously stored username/password combination.
    //
    // If anything was deleted, returns true. Otherwise returns false and
    // logs an error if any error other than not finding any matches occurred.
    bool Delete(const wxString& service);

private:
    // Ctor takes ownership of the passed pointer.
    explicit wxSecretStore(wxSecretStoreImpl* impl) : m_impl(impl) { }

    wxSecretStoreImpl* const m_impl;
};

#else // !wxUSE_SECRETSTORE

// Provide stand in for wxSecretValue allowing to use it without having #if
// wxUSE_SECRETSTORE checks everywhere. Unlike the real version, this class
// doesn't provide any added security.
class wxSecretValue
{
public:
    wxSecretValue() { m_valid = false; }

    wxSecretValue(size_t size, const void *data)
    {
        Init(size, data);
    }

    explicit wxSecretValue(const wxString& secret)
    {
        const wxScopedCharBuffer buf(secret.utf8_str());
        Init(buf.length(), buf.data());
    }

    bool IsOk() const { return m_valid; }

    bool operator==(const wxSecretValue& other) const
    {
        return m_valid == other.m_valid && m_data == other.m_data;
    }

    bool operator!=(const wxSecretValue& other) const
    {
        return !(*this == other);
    }

    size_t GetSize() const { return m_data.utf8_str().length(); }

    const void *GetData() const { return m_data.utf8_str().data(); }

    wxString GetAsString(const wxMBConv& conv = wxConvWhateverWorks) const
    {
        wxUnusedVar(conv);
        return m_data;
    }

    static void Wipe(size_t size, void *data) { memset(data, 0, size); }
    static void WipeString(wxString& str)
    {
        str.assign(str.length(), '*');
        str.clear();
    }

private:
    void Init(size_t size, const void *data)
    {
        m_data = wxString::From8BitData(static_cast<const char*>(data), size);
    }

    wxString m_data;
    bool m_valid;
};

#endif // wxUSE_SECRETSTORE/!wxUSE_SECRETSTORE

// Helper class ensuring WipeString() is called.
//
// It should only be used as a local variable and never polymorphically.
class wxSecretString : public wxString
{
public:
    wxSecretString()
    {
    }

    wxSecretString(const wxString& value)
        : wxString(value)
    {
    }

    explicit wxSecretString(const wxSecretValue& value)
        : wxString(value.GetAsString())
    {
    }

    ~wxSecretString()
    {
        wxSecretValue::WipeString(*this);
    }
};

#endif // _WX_SECRETSTORE_H_
