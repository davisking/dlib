///////////////////////////////////////////////////////////////////////////////
// Name:        wx/stdpaths.h
// Purpose:     declaration of wxStandardPaths class
// Author:      Vadim Zeitlin
// Modified by:
// Created:     2004-10-17
// Copyright:   (c) 2004 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_STDPATHS_H_
#define _WX_STDPATHS_H_

#include "wx/defs.h"

#include "wx/string.h"
#include "wx/filefn.h"

class WXDLLIMPEXP_FWD_BASE wxStandardPaths;

// ----------------------------------------------------------------------------
// wxStandardPaths returns the standard locations in the file system
// ----------------------------------------------------------------------------

// NB: This is always compiled in, wxUSE_STDPATHS=0 only disables native
//     wxStandardPaths class, but a minimal version is always available
class WXDLLIMPEXP_BASE wxStandardPathsBase
{
public:
    // possible resources categories
    enum ResourceCat
    {
        // no special category
        ResourceCat_None,

        // message catalog resources
        ResourceCat_Messages,

        // end of enum marker
        ResourceCat_Max
    };

    // what should we use to construct paths unique to this application:
    // (AppInfo_AppName and AppInfo_VendorName can be combined together)
    enum
    {
        AppInfo_None       = 0,  // nothing
        AppInfo_AppName    = 1,  // the application name
        AppInfo_VendorName = 2   // the vendor name
    };

    enum Dir
    {
        Dir_Cache,
        Dir_Documents,
        Dir_Desktop,
        Dir_Downloads,
        Dir_Music,
        Dir_Pictures,
        Dir_Videos
    };

    // Layout to use for user config/data files under Unix.
    enum FileLayout
    {
        FileLayout_Classic,     // Default: use home directory.
        FileLayout_XDG          // Recommended: use XDG specification.
    };

    // Naming convention for the config files under Unix.
    enum ConfigFileConv
    {
        ConfigFileConv_Dot,     // Classic Unix dot-file convention.
        ConfigFileConv_Ext      // Use .conf extension.
    };


    // return the global standard paths object
    static wxStandardPaths& Get();

    // return the path (directory+filename) of the running executable or
    // wxEmptyString if it couldn't be determined.
    // The path is returned as an absolute path whenever possible.
    // Default implementation only try to use wxApp->argv[0].
    virtual wxString GetExecutablePath() const;

    // return the directory with system config files:
    // /etc under Unix, c:\Documents and Settings\All Users\Application Data
    // under Windows, /Library/Preferences for Mac
    virtual wxString GetConfigDir() const = 0;

    // return the directory for the user config files:
    // $HOME under Unix, c:\Documents and Settings\username under Windows,
    // ~/Library/Preferences under Mac
    //
    // only use this if you have a single file to put there, otherwise
    // GetUserDataDir() is more appropriate
    virtual wxString GetUserConfigDir() const = 0;

    // return the location of the applications global, i.e. not user-specific,
    // data files
    //
    // prefix/share/appname under Unix, c:\Program Files\appname under Windows,
    // appname.app/Contents/SharedSupport app bundle directory under Mac
    virtual wxString GetDataDir() const = 0;

    // return the location for application data files which are host-specific
    //
    // same as GetDataDir() except under Unix where it is /etc/appname
    virtual wxString GetLocalDataDir() const;

    // return the directory for the user-dependent application data files
    //
    // $HOME/.appname under Unix,
    // c:\Documents and Settings\username\Application Data\appname under Windows
    // and ~/Library/Application Support/appname under Mac
    virtual wxString GetUserDataDir() const = 0;

    // return the directory for user data files which shouldn't be shared with
    // the other machines
    //
    // same as GetUserDataDir() for all platforms except Windows where it is
    // the "Local Settings\Application Data\appname" directory
    virtual wxString GetUserLocalDataDir() const;

    // return the directory where the loadable modules (plugins) live
    //
    // prefix/lib/appname under Unix, program directory under Windows and
    // Contents/Plugins app bundle subdirectory under Mac
    virtual wxString GetPluginsDir() const = 0;

    // get resources directory: resources are auxiliary files used by the
    // application and include things like image and sound files
    //
    // same as GetDataDir() for all platforms except Mac where it returns
    // Contents/Resources subdirectory of the app bundle
    virtual wxString GetResourcesDir() const { return GetDataDir(); }

    // get localized resources directory containing the resource files of the
    // specified category for the given language
    //
    // in general this is just GetResourcesDir()/lang under Windows and Unix
    // and GetResourcesDir()/lang.lproj under Mac but is something quite
    // different under Unix for message catalog category (namely the standard
    // prefix/share/locale/lang/LC_MESSAGES)
    virtual wxString
    GetLocalizedResourcesDir(const wxString& lang,
                             ResourceCat WXUNUSED(category)
                                = ResourceCat_None) const
    {
        return GetResourcesDir() + wxFILE_SEP_PATH + lang;
    }

    // return the "Documents" directory for the current user
    //
    // C:\Documents and Settings\username\My Documents under Windows,
    // $HOME under Unix and ~/Documents under Mac
    virtual wxString GetDocumentsDir() const
    {
        return GetUserDir(Dir_Documents);
    }

    // return the directory for the documents files used by this application:
    // it's a subdirectory of GetDocumentsDir() constructed using the
    // application name/vendor if it exists or just GetDocumentsDir() otherwise
    virtual wxString GetAppDocumentsDir() const;

    // return the temporary directory for the current user
    virtual wxString GetTempDir() const;

    virtual wxString GetUserDir(Dir userDir) const;

    virtual wxString
    MakeConfigFileName(const wxString& basename,
                       ConfigFileConv conv = ConfigFileConv_Ext) const = 0;

    // virtual dtor for the base class
    virtual ~wxStandardPathsBase();

    // Information used by AppendAppInfo
    void UseAppInfo(int info)
    {
        m_usedAppInfo = info;
    }

    bool UsesAppInfo(int info) const { return (m_usedAppInfo & info) != 0; }

    void SetFileLayout(FileLayout layout)
    {
        m_fileLayout = layout;
    }

    FileLayout GetFileLayout() const
    {
        return m_fileLayout;
    }

protected:
    // Ctor is protected as this is a base class which should never be created
    // directly.
    wxStandardPathsBase();

    // append the path component, with a leading path separator if a
    // path separator or dot (.) is not already at the end of dir
    static wxString AppendPathComponent(const wxString& dir, const wxString& component);

    // append application information determined by m_usedAppInfo to dir
    wxString AppendAppInfo(const wxString& dir) const;


    // combination of AppInfo_XXX flags used by AppendAppInfo()
    int m_usedAppInfo;

    // The file layout to use, currently only used under Unix.
    FileLayout m_fileLayout;
};

#if wxUSE_STDPATHS
    #if defined(__WINDOWS__)
        #include "wx/msw/stdpaths.h"
        #define wxHAS_NATIVE_STDPATHS
    #elif defined(__WXOSX_COCOA__) || defined(__WXOSX_IPHONE__) || defined(__DARWIN__)
        #include "wx/osx/cocoa/stdpaths.h"
        #define wxHAS_NATIVE_STDPATHS
    #elif defined(__UNIX__)
        #include "wx/unix/stdpaths.h"
        #define wxHAS_NATIVE_STDPATHS
        #define wxHAS_STDPATHS_INSTALL_PREFIX
    #endif
#endif

// ----------------------------------------------------------------------------
// Minimal generic implementation
// ----------------------------------------------------------------------------

// NB: Note that this minimal implementation is compiled in even if
//     wxUSE_STDPATHS=0, so that our code can still use wxStandardPaths.

#ifndef wxHAS_NATIVE_STDPATHS
#define wxHAS_STDPATHS_INSTALL_PREFIX
class WXDLLIMPEXP_BASE wxStandardPaths : public wxStandardPathsBase
{
public:
    void SetInstallPrefix(const wxString& prefix) { m_prefix = prefix; }
    wxString GetInstallPrefix() const { return m_prefix; }

    virtual wxString GetExecutablePath() const { return m_prefix; }
    virtual wxString GetConfigDir() const { return m_prefix; }
    virtual wxString GetUserConfigDir() const { return m_prefix; }
    virtual wxString GetDataDir() const { return m_prefix; }
    virtual wxString GetLocalDataDir() const { return m_prefix; }
    virtual wxString GetUserDataDir() const { return m_prefix; }
    virtual wxString GetPluginsDir() const { return m_prefix; }
    virtual wxString GetUserDir(Dir WXUNUSED(userDir)) const { return m_prefix; }
    virtual wxString
    MakeConfigFileName(const wxString& basename,
                       ConfigFileConv WXUNUSED(conv) = ConfigFileConv_Ext) const
    {
        return m_prefix + wxS("/") + basename;
    }

protected:
    // Ctor is protected because wxStandardPaths::Get() should always be used
    // to access the global wxStandardPaths object of the correct type instead
    // of creating one of a possibly wrong type yourself.
    wxStandardPaths() { }

private:
    wxString m_prefix;
};
#endif // !wxHAS_NATIVE_STDPATHS

#endif // _WX_STDPATHS_H_

