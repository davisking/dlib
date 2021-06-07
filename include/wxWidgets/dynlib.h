/////////////////////////////////////////////////////////////////////////////
// Name:        wx/dynlib.h
// Purpose:     Dynamic library loading classes
// Author:      Guilhem Lavaux, Vadim Zeitlin, Vaclav Slavik
// Modified by:
// Created:     20/07/98
// Copyright:   (c) 1998 Guilhem Lavaux
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_DYNLIB_H__
#define _WX_DYNLIB_H__

#include "wx/defs.h"

#if wxUSE_DYNLIB_CLASS

#include "wx/string.h"
#include "wx/dynarray.h"

class WXDLLIMPEXP_FWD_BASE wxDynamicLibraryDetailsCreator;

// ----------------------------------------------------------------------------
// conditional compilation
// ----------------------------------------------------------------------------

#if defined(__WINDOWS__)
    typedef WXHMODULE           wxDllType;
#elif defined(HAVE_DLOPEN)
    #include <dlfcn.h>
    typedef void               *wxDllType;
#elif defined(__WXMAC__)
    #include <CodeFragments.h>
    typedef CFragConnectionID   wxDllType;
#else
    #error "Dynamic Loading classes can't be compiled on this platform, sorry."
#endif

// ----------------------------------------------------------------------------
// constants
// ----------------------------------------------------------------------------

enum wxDLFlags
{
    wxDL_LAZY       = 0x00000001,   // resolve undefined symbols at first use
                                    // (only works on some Unix versions)
    wxDL_NOW        = 0x00000002,   // resolve undefined symbols on load
                                    // (default, always the case under Win32)
    wxDL_GLOBAL     = 0x00000004,   // export extern symbols to subsequently
                                    // loaded libs.
    wxDL_VERBATIM   = 0x00000008,   // attempt to load the supplied library
                                    // name without appending the usual dll
                                    // filename extension.

    // this flag is obsolete, don't use
    wxDL_NOSHARE    = 0x00000010,   // load new DLL, don't reuse already loaded
                                    // (only for wxPluginManager)

    wxDL_QUIET      = 0x00000020,   // don't log an error if failed to load

    // this flag is dangerous, for internal use of wxMSW only, don't use at all
    // and especially don't use directly, use wxLoadedDLL instead if you really
    // do need it
    wxDL_GET_LOADED = 0x00000040,   // Win32 only: return handle of already
                                    // loaded DLL or NULL otherwise; Unload()
                                    // should not be called so don't forget to
                                    // Detach() if you use this function

    wxDL_DEFAULT    = wxDL_NOW      // default flags correspond to Win32
};

enum wxDynamicLibraryCategory
{
    wxDL_LIBRARY,       // standard library
    wxDL_MODULE         // loadable module/plugin
};

enum wxPluginCategory
{
    wxDL_PLUGIN_GUI,    // plugin that uses GUI classes
    wxDL_PLUGIN_BASE    // wxBase-only plugin
};

// ----------------------------------------------------------------------------
// macros
// ----------------------------------------------------------------------------

// when loading a function from a DLL you always have to cast the returned
// "void *" pointer to the correct type and, even more annoyingly, you have to
// repeat this type twice if you want to declare and define a function pointer
// all in one line
//
// this macro makes this slightly less painful by allowing you to specify the
// type only once, as the first parameter, and creating a variable of this type
// called "pfn<name>" initialized with the "name" from the "dynlib"
#define wxDYNLIB_FUNCTION(type, name, dynlib) \
    type pfn ## name = (type)(dynlib).GetSymbol(wxT(#name))


// a more convenient function replacing wxDYNLIB_FUNCTION above
//
// it uses the convention that the type of the function is its name suffixed
// with "_t" but it doesn't define a variable but just assigns the loaded value
// to it and also allows to pass it the prefix to be used instead of hardcoding
// "pfn" (the prefix can be "m_" or "gs_pfn" or whatever)
//
// notice that this function doesn't generate error messages if the symbol
// couldn't be loaded, the caller should generate the appropriate message
#define wxDL_INIT_FUNC(pfx, name, dynlib) \
    pfx ## name = (name ## _t)(dynlib).RawGetSymbol(#name)

#ifdef __WINDOWS__

// same as wxDL_INIT_FUNC() but appends 'A' or 'W' to the function name, see
// wxDynamicLibrary::GetSymbolAorW()
#define wxDL_INIT_FUNC_AW(pfx, name, dynlib) \
    pfx ## name = (name ## _t)(dynlib).GetSymbolAorW(#name)

#endif // __WINDOWS__

// the following macros can be used to redirect a whole library to a class and
// check at run-time if the library is present and contains all required
// methods
//
// notice that they are supposed to be used inside a class which has "m_ok"
// member variable indicating if the library had been successfully loaded

// helper macros constructing the name of the variable storing the function
// pointer and the name of its type from the function name
#define wxDL_METHOD_NAME(name) m_pfn ## name
#define wxDL_METHOD_TYPE(name) name ## _t

// parameters are:
//  - rettype: return type of the function, e.g. "int"
//  - name: name of the function, e.g. "foo"
//  - args: function signature in parentheses, e.g. "(int x, int y)"
//  - argnames: the names of the parameters in parentheses, e.g. "(x, y)"
//  - defret: the value to return if the library wasn't successfully loaded
#define wxDL_METHOD_DEFINE( rettype, name, args, argnames, defret ) \
    typedef rettype (* wxDL_METHOD_TYPE(name)) args ; \
    wxDL_METHOD_TYPE(name) wxDL_METHOD_NAME(name); \
    rettype name args \
        { return m_ok ? wxDL_METHOD_NAME(name) argnames : defret; }

#define wxDL_VOIDMETHOD_DEFINE( name, args, argnames ) \
    typedef void (* wxDL_METHOD_TYPE(name)) args ; \
    wxDL_METHOD_TYPE(name) wxDL_METHOD_NAME(name); \
    void name args \
        { if ( m_ok ) wxDL_METHOD_NAME(name) argnames ; }

#define wxDL_METHOD_LOAD(lib, name) \
    wxDL_METHOD_NAME(name) = \
        (wxDL_METHOD_TYPE(name)) lib.GetSymbol(#name, &m_ok); \
    if ( !m_ok ) return false

// ----------------------------------------------------------------------------
// wxDynamicLibraryDetails: contains details about a loaded wxDynamicLibrary
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxDynamicLibraryDetails
{
public:
    // ctor, normally never used as these objects are only created by
    // wxDynamicLibrary::ListLoaded()
    wxDynamicLibraryDetails() { m_address = NULL; m_length = 0; }

    // get the (base) name
    wxString GetName() const { return m_name; }

    // get the full path of this object
    wxString GetPath() const { return m_path; }

    // get the load address and the extent, return true if this information is
    // available
    bool GetAddress(void **addr, size_t *len) const
    {
        if ( !m_address )
            return false;

        if ( addr )
            *addr = m_address;
        if ( len )
            *len = m_length;

        return true;
    }

    // return the version of the DLL (may be empty if no version info)
    wxString GetVersion() const
    {
        return m_version;
    }

private:
    wxString m_name,
             m_path,
             m_version;

    void *m_address;
    size_t m_length;

    friend class wxDynamicLibraryDetailsCreator;
};

WX_DECLARE_USER_EXPORTED_OBJARRAY(wxDynamicLibraryDetails,
                                  wxDynamicLibraryDetailsArray,
                                  WXDLLIMPEXP_BASE);

// ----------------------------------------------------------------------------
// wxDynamicLibrary: represents a handle to a DLL/shared object
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxDynamicLibrary
{
public:
    // return a valid handle for the main program itself or NULL if back
    // linking is not supported by the current platform (e.g. Win32)
    static wxDllType         GetProgramHandle();

    // return the platform standard DLL extension (with leading dot)
    static wxString GetDllExt(wxDynamicLibraryCategory cat = wxDL_LIBRARY);

    wxDynamicLibrary() : m_handle(NULL) { }
    wxDynamicLibrary(const wxString& libname, int flags = wxDL_DEFAULT)
        : m_handle(NULL)
    {
        Load(libname, flags);
    }

    // NOTE: this class is (deliberately) not virtual, do not attempt
    //       to use it polymorphically.
    ~wxDynamicLibrary() { Unload(); }

    // return true if the library was loaded successfully
    bool IsLoaded() const { return m_handle != NULL; }

    // load the library with the given name (full or not), return true if ok
    bool Load(const wxString& libname, int flags = wxDL_DEFAULT);

    // raw function for loading dynamic libs: always behaves as if
    // wxDL_VERBATIM were specified and doesn't log error message if the
    // library couldn't be loaded but simply returns NULL
    static wxDllType RawLoad(const wxString& libname, int flags = wxDL_DEFAULT);

    // attach to an existing handle
    void Attach(wxDllType h) { Unload(); m_handle = h; }

    // detach the library object from its handle, i.e. prevent the object from
    // unloading the library in its dtor -- the caller is now responsible for
    // doing this
    wxDllType Detach() { wxDllType h = m_handle; m_handle = NULL; return h; }

    // unload the given library handle (presumably returned by Detach() before)
    static void Unload(wxDllType handle);

    // unload the library, also done automatically in dtor
    void Unload() { if ( IsLoaded() ) { Unload(m_handle); m_handle = NULL; } }

    // Return the raw handle from dlopen and friends.
    wxDllType GetLibHandle() const { return m_handle; }

    // check if the given symbol is present in the library, useful to verify if
    // a loadable module is our plugin, for example, without provoking error
    // messages from GetSymbol()
    bool HasSymbol(const wxString& name) const
    {
        bool ok;
        DoGetSymbol(name, &ok);
        return ok;
    }

    // resolve a symbol in a loaded DLL, such as a variable or function name.
    // 'name' is the (possibly mangled) name of the symbol. (use extern "C" to
    // export unmangled names)
    //
    // Since it is perfectly valid for the returned symbol to actually be NULL,
    // that is not always indication of an error.  Pass and test the parameter
    // 'success' for a true indication of success or failure to load the
    // symbol.
    //
    // Returns a pointer to the symbol on success, or NULL if an error occurred
    // or the symbol wasn't found.
    void *GetSymbol(const wxString& name, bool *success = NULL) const;

    // low-level version of GetSymbol()
    static void *RawGetSymbol(wxDllType handle, const wxString& name);
    void *RawGetSymbol(const wxString& name) const
    {
        return RawGetSymbol(m_handle, name);
    }

#ifdef __WINDOWS__
    // this function is useful for loading functions from the standard Windows
    // DLLs: such functions have an 'A' (in ANSI build) or 'W' (in Unicode, or
    // wide character build) suffix if they take string parameters
    static void *RawGetSymbolAorW(wxDllType handle, const wxString& name)
    {
        return RawGetSymbol
               (
                handle,
                name +
#if wxUSE_UNICODE
                L'W'
#else
                'A'
#endif
               );
    }

    void *GetSymbolAorW(const wxString& name) const
    {
        return RawGetSymbolAorW(m_handle, name);
    }
#endif // __WINDOWS__

    // return all modules/shared libraries in the address space of this process
    //
    // returns an empty array if not implemented or an error occurred
    static wxDynamicLibraryDetailsArray ListLoaded();

    // return platform-specific name of dynamic library with proper extension
    // and prefix (e.g. "foo.dll" on Windows or "libfoo.so" on Linux)
    static wxString CanonicalizeName(const wxString& name,
                                     wxDynamicLibraryCategory cat = wxDL_LIBRARY);

    // return name of wxWidgets plugin (adds compiler and version info
    // to the filename):
    static wxString
    CanonicalizePluginName(const wxString& name,
                           wxPluginCategory cat = wxDL_PLUGIN_GUI);

    // return plugin directory on platforms where it makes sense and empty
    // string on others:
    static wxString GetPluginsDirectory();

    // Return the load address of the module containing the given address or
    // NULL if not found.
    //
    // If path output parameter is non-NULL, fill it with the full path to this
    // module disk file on success.
    static void* GetModuleFromAddress(const void* addr, wxString* path = NULL);

#ifdef __WINDOWS__
    // return the handle (HMODULE/HINSTANCE) of the DLL with the given name
    // and/or containing the specified address: for XP and later systems only
    // the address is used and the name is ignored but for the previous systems
    // only the name (which may be either a full path to the DLL or just its
    // base name, possibly even without extension) is used
    //
    // the returned handle reference count is not incremented so it doesn't
    // need to be freed using FreeLibrary() but it also means that it can
    // become invalid if the DLL is unloaded
    static WXHMODULE MSWGetModuleHandle(const wxString& name, void *addr);
#endif // __WINDOWS__

protected:
    // common part of GetSymbol() and HasSymbol()
    void* DoGetSymbol(const wxString& name, bool* success = NULL) const;

    // log the error after an OS dynamic library function failure
    static void ReportError(const wxString& msg,
                            const wxString& name = wxString());

    // the handle to DLL or NULL
    wxDllType m_handle;

    // no copy ctor/assignment operators (or we'd try to unload the library
    // twice)
    wxDECLARE_NO_COPY_CLASS(wxDynamicLibrary);
};

#ifdef __WINDOWS__

// ----------------------------------------------------------------------------
// wxLoadedDLL is a MSW-only internal helper class allowing to dynamically bind
// to a DLL already loaded into the project address space
// ----------------------------------------------------------------------------

class wxLoadedDLL : public wxDynamicLibrary
{
public:
    wxLoadedDLL(const wxString& dllname)
        : wxDynamicLibrary(dllname, wxDL_GET_LOADED | wxDL_VERBATIM | wxDL_QUIET)
    {
    }

    ~wxLoadedDLL()
    {
        Detach();
    }
};

#endif // __WINDOWS__

// ----------------------------------------------------------------------------
// Interesting defines
// ----------------------------------------------------------------------------

#define WXDLL_ENTRY_FUNCTION() \
extern "C" WXEXPORT const wxClassInfo *wxGetClassFirst(); \
const wxClassInfo *wxGetClassFirst() { \
  return wxClassInfo::GetFirst(); \
}

#endif // wxUSE_DYNLIB_CLASS

#endif // _WX_DYNLIB_H__
