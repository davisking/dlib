/////////////////////////////////////////////////////////////////////////////
// Name:        wx/module.h
// Purpose:     Modules handling
// Author:      Wolfram Gloger/adapted by Guilhem Lavaux
// Modified by:
// Created:     04/11/98
// Copyright:   (c) Wolfram Gloger and Guilhem Lavaux
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_MODULE_H_
#define _WX_MODULE_H_

#include "wx/object.h"
#include "wx/vector.h"

class wxModule;

typedef wxVector<wxModule*> wxModuleList;

// declaring a class derived from wxModule will automatically create an
// instance of this class on program startup, call its OnInit() method and call
// OnExit() on program termination (but only if OnInit() succeeded)
class WXDLLIMPEXP_BASE wxModule : public wxObject
{
public:
    wxModule() {}
    virtual ~wxModule() {}

    // if module init routine returns false the application
    // will fail to startup

    bool Init() { return OnInit(); }
    void Exit() { OnExit(); }

    // Override both of these

        // called on program startup

    virtual bool OnInit() = 0;

        // called just before program termination, but only if OnInit()
        // succeeded

    virtual void OnExit() = 0;

    static void RegisterModule(wxModule *module);
    static void RegisterModules();
    static bool InitializeModules();
    static void CleanUpModules();
    static bool AreInitialized() { return ms_areInitialized; }

    // used by wxObjectLoader when unloading shared libs's

    static void UnregisterModule(wxModule *module);

protected:
    static wxModuleList ms_modules;

    static bool ms_areInitialized;

    // the function to call from constructor of a deriving class add module
    // dependency which will be initialized before the module and unloaded
    // after that
    void AddDependency(wxClassInfo *dep)
    {
        wxCHECK_RET( dep, wxT("NULL module dependency") );

        m_dependencies.push_back(dep);
    }

    // same as the version above except it will look up wxClassInfo by name on
    // its own. Note that className must be ASCII
    void AddDependency(const char *className)
    {
        m_namedDependencies.push_back(wxASCII_STR(className));
    }


private:
    // initialize module and Append it to initializedModules list recursively
    // calling itself to satisfy module dependencies if needed
    static bool
    DoInitializeModule(wxModule *module, wxModuleList &initializedModules);

    // cleanup the modules in the specified list (which may not contain all
    // modules if we're called during initialization because not all modules
    // could be initialized) and also empty ms_modules itself
    static void DoCleanUpModules(const wxModuleList& modules);

    // resolve all named dependencies and add them to the normal m_dependencies
    bool ResolveNamedDependencies();


    // module dependencies: contains wxClassInfo pointers for all modules which
    // must be initialized before this one
    typedef wxVector<wxClassInfo*> wxArrayClassInfo;
    wxArrayClassInfo m_dependencies;

    // and the named dependencies: those will be resolved during run-time and
    // added to m_dependencies
    wxVector<wxString> m_namedDependencies;

    // used internally while initializing/cleaning up modules
    enum
    {
        State_Registered,   // module registered but not initialized yet
        State_Initializing, // we're initializing this module but not done yet
        State_Initialized   // module initialized successfully
    } m_state;


    wxDECLARE_CLASS(wxModule);
};

#endif // _WX_MODULE_H_
