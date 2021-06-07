/////////////////////////////////////////////////////////////////////////////
// Name:        wx/translation.h
// Purpose:     Internationalization and localisation for wxWidgets
// Author:      Vadim Zeitlin, Vaclav Slavik,
//              Michael N. Filippov <michael@idisys.iae.nsk.su>
// Created:     2010-04-23
// Copyright:   (c) 1998 Vadim Zeitlin <zeitlin@dptmaths.ens-cachan.fr>
//              (c) 2010 Vaclav Slavik <vslavik@fastmail.fm>
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_TRANSLATION_H_
#define _WX_TRANSLATION_H_

#include "wx/defs.h"
#include "wx/string.h"

#if wxUSE_INTL

#include "wx/buffer.h"
#include "wx/language.h"
#include "wx/hashmap.h"
#include "wx/strconv.h"
#include "wx/scopedptr.h"

// ============================================================================
// global decls
// ============================================================================

// ----------------------------------------------------------------------------
// macros
// ----------------------------------------------------------------------------

// gettext() style macros (notice that xgettext should be invoked with
// --keyword="_" --keyword="wxPLURAL:1,2" options
// to extract the strings from the sources)
#ifndef WXINTL_NO_GETTEXT_MACRO
#ifndef wxNO_IMPLICIT_WXSTRING_ENCODING
    #define _(s)                               wxGetTranslation((s))
#else
    #define _(s)                               wxGetTranslation(wxASCII_STR(s))
#endif
    #define wxPLURAL(sing, plur, n)            wxGetTranslation((sing), (plur), n)
#endif

// wx-specific macro for translating strings in the given context: if you use
// them, you need to also add
// --keyword="wxGETTEXT_IN_CONTEXT:1c,2" --keyword="wxGETTEXT_IN_CONTEXT_PLURAL:1c,2,3"
// options to xgettext invocation.
#ifndef wxNO_IMPLICIT_WXSTRING_ENCODING
#define wxGETTEXT_IN_CONTEXT(c, s) \
    wxGetTranslation((s), wxString(), c)
#else
#define wxGETTEXT_IN_CONTEXT(c, s) \
    wxGetTranslation(wxASCII_STR(s), wxString(), c)
#endif
#define wxGETTEXT_IN_CONTEXT_PLURAL(c, sing, plur, n) \
    wxGetTranslation((sing), (plur), n, wxString(), c)

// another one which just marks the strings for extraction, but doesn't
// perform the translation (use -kwxTRANSLATE with xgettext!)
#define wxTRANSLATE(str) str

// ----------------------------------------------------------------------------
// forward decls
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_FWD_BASE wxArrayString;
class WXDLLIMPEXP_FWD_BASE wxTranslationsLoader;
class WXDLLIMPEXP_FWD_BASE wxLocale;

class wxPluralFormsCalculator;
wxDECLARE_SCOPED_PTR(wxPluralFormsCalculator, wxPluralFormsCalculatorPtr)

// ----------------------------------------------------------------------------
// wxMsgCatalog corresponds to one loaded message catalog.
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxMsgCatalog
{
public:
    // Ctor is protected, because CreateFromXXX functions must be used,
    // but destruction should be unrestricted
#if !wxUSE_UNICODE
    ~wxMsgCatalog();
#endif

    // load the catalog from disk or from data; caller is responsible for
    // deleting them if not NULL
    static wxMsgCatalog *CreateFromFile(const wxString& filename,
                                        const wxString& domain);

    static wxMsgCatalog *CreateFromData(const wxScopedCharBuffer& data,
                                        const wxString& domain);

    // get name of the catalog
    wxString GetDomain() const { return m_domain; }

    // get the translated string: returns NULL if not found
    const wxString *GetString(const wxString& sz, unsigned n = UINT_MAX, const wxString& ct = wxEmptyString) const;

protected:
    wxMsgCatalog(const wxString& domain)
        : m_pNext(NULL), m_domain(domain)
#if !wxUSE_UNICODE
        , m_conv(NULL)
#endif
    {}

private:
    // variable pointing to the next element in a linked list (or NULL)
    wxMsgCatalog *m_pNext;
    friend class wxTranslations;

    wxStringToStringHashMap m_messages; // all messages in the catalog
    wxString                m_domain;   // name of the domain

#if !wxUSE_UNICODE
    // the conversion corresponding to this catalog charset if we installed it
    // as the global one
    wxCSConv *m_conv;
#endif

    wxPluralFormsCalculatorPtr m_pluralFormsCalculator;
};

// ----------------------------------------------------------------------------
// wxTranslations: message catalogs
// ----------------------------------------------------------------------------

// this class allows to get translations for strings
class WXDLLIMPEXP_BASE wxTranslations
{
public:
    wxTranslations();
    ~wxTranslations();

    // returns current translations object, may return NULL
    static wxTranslations *Get();
    // sets current translations object (takes ownership; may be NULL)
    static void Set(wxTranslations *t);

    // changes loader to non-default one; takes ownership of 'loader'
    void SetLoader(wxTranslationsLoader *loader);

    void SetLanguage(wxLanguage lang);
    void SetLanguage(const wxString& lang);

    // get languages available for this app
    wxArrayString GetAvailableTranslations(const wxString& domain) const;

    // find best translation language for given domain
    wxString GetBestTranslation(const wxString& domain, wxLanguage msgIdLanguage);
    wxString GetBestTranslation(const wxString& domain,
                                const wxString& msgIdLanguage = wxASCII_STR("en"));

    // add standard wxWidgets catalog ("wxstd")
    bool AddStdCatalog();

    // add catalog with given domain name and language, looking it up via
    // wxTranslationsLoader
    bool AddCatalog(const wxString& domain,
                    wxLanguage msgIdLanguage = wxLANGUAGE_ENGLISH_US);
#if !wxUSE_UNICODE
    bool AddCatalog(const wxString& domain,
                    wxLanguage msgIdLanguage,
                    const wxString& msgIdCharset);
#endif

    // check if the given catalog is loaded
    bool IsLoaded(const wxString& domain) const;

    // access to translations
    const wxString *GetTranslatedString(const wxString& origString,
                                        const wxString& domain = wxEmptyString,
                                        const wxString& context = wxEmptyString) const;
    const wxString *GetTranslatedString(const wxString& origString,
                                        unsigned n,
                                        const wxString& domain = wxEmptyString,
                                        const wxString& context = wxEmptyString) const;

    wxString GetHeaderValue(const wxString& header,
                            const wxString& domain = wxEmptyString) const;

    // this is hack to work around a problem with wxGetTranslation() which
    // returns const wxString& and not wxString, so when it returns untranslated
    // string, it needs to have a copy of it somewhere
    static const wxString& GetUntranslatedString(const wxString& str);

private:
    // perform loading of the catalog via m_loader
    bool LoadCatalog(const wxString& domain, const wxString& lang, const wxString& msgIdLang);

    // find catalog by name in a linked list, return NULL if !found
    wxMsgCatalog *FindCatalog(const wxString& domain) const;

    // same as Set(), without taking ownership; only for wxLocale
    static void SetNonOwned(wxTranslations *t);
    friend class wxLocale;

private:
    wxString m_lang;
    wxTranslationsLoader *m_loader;

    wxMsgCatalog *m_pMsgCat; // pointer to linked list of catalogs

    // In addition to keeping all the catalogs in the linked list, we also
    // store them in a hash map indexed by the domain name to allow finding
    // them by name efficiently.
    WX_DECLARE_HASH_MAP(wxString, wxMsgCatalog *, wxStringHash, wxStringEqual, wxMsgCatalogMap);
    wxMsgCatalogMap m_catalogMap;
};


// abstraction of translations discovery and loading
class WXDLLIMPEXP_BASE wxTranslationsLoader
{
public:
    wxTranslationsLoader() {}
    virtual ~wxTranslationsLoader() {}

    virtual wxMsgCatalog *LoadCatalog(const wxString& domain,
                                      const wxString& lang) = 0;

    virtual wxArrayString GetAvailableTranslations(const wxString& domain) const = 0;
};


// standard wxTranslationsLoader implementation, using filesystem
class WXDLLIMPEXP_BASE wxFileTranslationsLoader
    : public wxTranslationsLoader
{
public:
    static void AddCatalogLookupPathPrefix(const wxString& prefix);

    virtual wxMsgCatalog *LoadCatalog(const wxString& domain,
                                      const wxString& lang) wxOVERRIDE;

    virtual wxArrayString GetAvailableTranslations(const wxString& domain) const wxOVERRIDE;
};


#ifdef __WINDOWS__
// loads translations from win32 resources
class WXDLLIMPEXP_BASE wxResourceTranslationsLoader
    : public wxTranslationsLoader
{
public:
    virtual wxMsgCatalog *LoadCatalog(const wxString& domain,
                                      const wxString& lang) wxOVERRIDE;

    virtual wxArrayString GetAvailableTranslations(const wxString& domain) const wxOVERRIDE;

protected:
    // returns resource type to use for translations
    virtual wxString GetResourceType() const { return wxASCII_STR("MOFILE"); }

    // returns module to load resources from
    virtual WXHINSTANCE GetModule() const { return NULL; }
};
#endif // __WINDOWS__


// ----------------------------------------------------------------------------
// global functions
// ----------------------------------------------------------------------------

// get the translation of the string in the current locale
inline const wxString& wxGetTranslation(const wxString& str,
                                        const wxString& domain = wxString(),
                                        const wxString& context = wxString())
{
    wxTranslations *trans = wxTranslations::Get();
    const wxString *transStr = trans ? trans->GetTranslatedString(str, domain, context)
                                     : NULL;
    if ( transStr )
        return *transStr;
    else
        // NB: this function returns reference to a string, so we have to keep
        //     a copy of it somewhere
        return wxTranslations::GetUntranslatedString(str);
}

inline const wxString& wxGetTranslation(const wxString& str1,
                                        const wxString& str2,
                                        unsigned n,
                                        const wxString& domain = wxString(),
                                        const wxString& context = wxString())
{
    wxTranslations *trans = wxTranslations::Get();
    const wxString *transStr = trans ? trans->GetTranslatedString(str1, n, domain, context)
                                     : NULL;
    if ( transStr )
        return *transStr;
    else
        // NB: this function returns reference to a string, so we have to keep
        //     a copy of it somewhere
        return n == 1
               ? wxTranslations::GetUntranslatedString(str1)
               : wxTranslations::GetUntranslatedString(str2);
}

#ifdef wxNO_IMPLICIT_WXSTRING_ENCODING

/*
 * It must always be possible to call wxGetTranslation() with const
 * char* arguments.
 */
inline const wxString& wxGetTranslation(const char *str,
                                        const char *domain = "",
                                        const char *context = "") {
    const wxMBConv &conv = wxConvWhateverWorks;
    return wxGetTranslation(wxString(str, conv), wxString(domain, conv),
                            wxString(context, conv));
}

inline const wxString& wxGetTranslation(const char *str1,
                                        const char *str2,
                                        unsigned n,
                                        const char *domain = "",
                                        const char *context = "") {
    const wxMBConv &conv = wxConvWhateverWorks;
    return wxGetTranslation(wxString(str1, conv), wxString(str2, conv), n,
                            wxString(domain, conv),
                            wxString(context, conv));
}

#endif // wxNO_IMPLICIT_WXSTRING_ENCODING

#else // !wxUSE_INTL

// the macros should still be defined - otherwise compilation would fail

#if !defined(WXINTL_NO_GETTEXT_MACRO)
    #if !defined(_)
#ifndef wxNO_IMPLICIT_WXSTRING_ENCODING
        #define _(s)                 (s)
#else
        #define _(s)                 wxASCII_STR(s)
#endif
    #endif
    #define wxPLURAL(sing, plur, n)  ((n) == 1 ? (sing) : (plur))
    #define wxGETTEXT_IN_CONTEXT(c, s)                     (s)
    #define wxGETTEXT_IN_CONTEXT_PLURAL(c, sing, plur, n)  wxPLURAL(sing, plur, n)
#endif

#define wxTRANSLATE(str) str

// NB: we use a template here in order to avoid using
//     wxLocale::GetUntranslatedString() above, which would be required if
//     we returned const wxString&; this way, the compiler should be able to
//     optimize wxGetTranslation() away

template<typename TString>
inline TString wxGetTranslation(TString str)
    { return str; }

template<typename TString, typename TDomain>
inline TString wxGetTranslation(TString str, TDomain WXUNUSED(domain))
    { return str; }

template<typename TString, typename TDomain, typename TContext>
inline TString wxGetTranslation(TString str, TDomain WXUNUSED(domain), TContext WXUNUSED(context))
    { return str; }

template<typename TString, typename TDomain>
inline TString wxGetTranslation(TString str1, TString str2, size_t n)
    { return n == 1 ? str1 : str2; }

template<typename TString, typename TDomain>
inline TString wxGetTranslation(TString str1, TString str2, size_t n,
                                TDomain WXUNUSED(domain))
    { return n == 1 ? str1 : str2; }

template<typename TString, typename TDomain, typename TContext>
inline TString wxGetTranslation(TString str1, TString str2, size_t n,
                                TDomain WXUNUSED(domain),
                                TContext WXUNUSED(context))
    { return n == 1 ? str1 : str2; }

#endif // wxUSE_INTL/!wxUSE_INTL

// define this one just in case it occurs somewhere (instead of preferred
// wxTRANSLATE) too
#if !defined(WXINTL_NO_GETTEXT_MACRO)
    #if !defined(gettext_noop)
        #define gettext_noop(str) (str)
    #endif
    #if !defined(N_)
        #define N_(s)             (s)
    #endif
#endif

#endif // _WX_TRANSLATION_H_
