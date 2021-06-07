/////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/menu.h
// Purpose:
// Author:      Robert Roebling
// Copyright:   (c) 1998 Robert Roebling, Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTKMENU_H_
#define _WX_GTKMENU_H_

//-----------------------------------------------------------------------------
// wxMenuBar
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxMenuBar : public wxMenuBarBase
{
public:
    // ctors
    wxMenuBar();
    wxMenuBar(long style);
    wxMenuBar(size_t n, wxMenu *menus[], const wxString titles[], long style = 0);
    ~wxMenuBar();

    // implement base class (pure) virtuals
    virtual bool Append( wxMenu *menu, const wxString &title ) wxOVERRIDE;
    virtual bool Insert(size_t pos, wxMenu *menu, const wxString& title) wxOVERRIDE;
    virtual wxMenu *Replace(size_t pos, wxMenu *menu, const wxString& title) wxOVERRIDE;
    virtual wxMenu *Remove(size_t pos) wxOVERRIDE;

    virtual int FindMenuItem(const wxString& menuString,
                             const wxString& itemString) const wxOVERRIDE;
    virtual wxMenuItem* FindItem( int id, wxMenu **menu = NULL ) const wxOVERRIDE;

    virtual void EnableTop( size_t pos, bool flag ) wxOVERRIDE;
    virtual bool IsEnabledTop(size_t pos) const wxOVERRIDE;
    virtual void SetMenuLabel( size_t pos, const wxString& label ) wxOVERRIDE;
    virtual wxString GetMenuLabel( size_t pos ) const wxOVERRIDE;

    void SetLayoutDirection(wxLayoutDirection dir) wxOVERRIDE;
    wxLayoutDirection GetLayoutDirection() const wxOVERRIDE;

    virtual void Attach(wxFrame *frame) wxOVERRIDE;
    virtual void Detach() wxOVERRIDE;

private:
    // common part of Append and Insert
    void GtkAppend(wxMenu* menu, const wxString& title, int pos = -1);

    void Init(size_t n, wxMenu *menus[], const wxString titles[], long style);

    // wxMenuBar is not a top level window but it still doesn't need a parent
    // window
    virtual bool GTKNeedsParent() const wxOVERRIDE { return false; }

    GtkWidget* m_menubar;

    wxDECLARE_DYNAMIC_CLASS(wxMenuBar);
};

//-----------------------------------------------------------------------------
// wxMenu
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxMenu : public wxMenuBase
{
public:
    // ctors & dtor
    wxMenu(const wxString& title, long style = 0)
        : wxMenuBase(title, style) { Init(); }

    wxMenu(long style = 0) : wxMenuBase(style) { Init(); }

    virtual ~wxMenu();

    void Attach(wxMenuBarBase *menubar) wxOVERRIDE;

    void SetLayoutDirection(wxLayoutDirection dir);
    wxLayoutDirection GetLayoutDirection() const;

    // Returns the title, with mnemonics translated to wx format
    wxString GetTitle() const;

    // Sets the title, with mnemonics translated to gtk format
    virtual void SetTitle(const wxString& title) wxOVERRIDE;

    // implementation GTK only
    GtkWidget       *m_menu;  // GtkMenu
    GtkWidget       *m_owner;
    GtkAccelGroup   *m_accel;
    bool m_popupShown;

protected:
    virtual wxMenuItem* DoAppend(wxMenuItem *item) wxOVERRIDE;
    virtual wxMenuItem* DoInsert(size_t pos, wxMenuItem *item) wxOVERRIDE;
    virtual wxMenuItem* DoRemove(wxMenuItem *item) wxOVERRIDE;

private:
    // common code for all constructors:
    void Init();

    // common part of Append (if pos == -1)  and Insert
    void GtkAppend(wxMenuItem* item, int pos = -1);


    wxDECLARE_DYNAMIC_CLASS(wxMenu);
};

#endif
    // _WX_GTKMENU_H_
