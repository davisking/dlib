///////////////////////////////////////////////////////////////////////////////
// Name:        wx/unix/glx11.h
// Purpose:     class common for all X11-based wxGLCanvas implementations
// Author:      Vadim Zeitlin
// Created:     2007-04-15
// Copyright:   (c) 2007 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_UNIX_GLX11_H_
#define _WX_UNIX_GLX11_H_

#include <GL/gl.h>

typedef struct __GLXcontextRec* GLXContext;
typedef struct __GLXFBConfigRec* GLXFBConfig;

// ----------------------------------------------------------------------------
// wxGLContext
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_GL wxGLContext : public wxGLContextBase
{
public:
    wxGLContext(wxGLCanvas *win,
                const wxGLContext *other = NULL,
                const wxGLContextAttrs *ctxAttrs = NULL);
    virtual ~wxGLContext();

    virtual bool SetCurrent(const wxGLCanvas& win) const wxOVERRIDE;

private:
    GLXContext m_glContext;

    wxDECLARE_CLASS(wxGLContext);
};

// ----------------------------------------------------------------------------
// wxGLCanvasX11
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_GL wxGLCanvasX11 : public wxGLCanvasBase
{
public:
    // initialization and dtor
    // -----------------------

    // default ctor doesn't do anything, InitVisual() must be called
    wxGLCanvasX11();

    // initializes GLXFBConfig and XVisualInfo corresponding to the given attributes
    bool InitVisual(const wxGLAttributes& dispAttrs);

    // frees XVisualInfo info
    virtual ~wxGLCanvasX11();


    // implement wxGLCanvasBase methods
    // --------------------------------

    virtual bool SwapBuffers() wxOVERRIDE;


    // X11-specific methods
    // --------------------

    // return GLX version: 13 means 1.3 &c
    static int GetGLXVersion();

    // return true if multisample extension is available
    static bool IsGLXMultiSampleAvailable();

    // get the X11 handle of this window
    virtual unsigned long GetXWindow() const = 0;


    // GLX-specific methods
    // --------------------

    // override some wxWindow methods
    // ------------------------------

    // return true only if the window is realized: OpenGL context can't be
    // created until we are
    virtual bool IsShownOnScreen() const wxOVERRIDE;


    // implementation only from now on
    // -------------------------------

    // get the GLXFBConfig/XVisualInfo we use
    GLXFBConfig *GetGLXFBConfig() const { return m_fbc; }
    void* GetXVisualInfo() const { return m_vi; }

    // initialize the global default GL visual, return false if matching visual
    // not found
    static bool InitDefaultVisualInfo(const int *attribList);

private:
    GLXFBConfig *m_fbc;
    void* m_vi;
};

// ----------------------------------------------------------------------------
// wxGLApp
// ----------------------------------------------------------------------------

// this is used in wx/glcanvas.h, prevent it from defining a generic wxGLApp
#define wxGL_APP_DEFINED

class WXDLLIMPEXP_GL wxGLApp : public wxGLAppBase
{
public:
    virtual bool InitGLVisual(const int *attribList) wxOVERRIDE;

    // This method is not currently used by the library itself, but remains for
    // backwards compatibility and also because wxGTK has it we could start
    // using it for the same purpose in wxX11 too some day.
    virtual void* GetXVisualInfo() wxOVERRIDE;

    // and override this wxApp method to clean up
    virtual int OnExit() wxOVERRIDE;

private:
    wxDECLARE_DYNAMIC_CLASS(wxGLApp);
};

#endif // _WX_UNIX_GLX11_H_

