/////////////////////////////////////////////////////////////////////////////
// Name:        wx/propgrid/manager.h
// Purpose:     wxPropertyGridManager
// Author:      Jaakko Salli
// Modified by:
// Created:     2005-01-14
// Copyright:   (c) Jaakko Salli
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PROPGRID_MANAGER_H_
#define _WX_PROPGRID_MANAGER_H_

#include "wx/defs.h"

#if wxUSE_PROPGRID

#include "wx/propgrid/propgrid.h"

// -----------------------------------------------------------------------

#ifndef SWIG
extern WXDLLIMPEXP_DATA_PROPGRID(const char) wxPropertyGridManagerNameStr[];
#endif

// Holder of property grid page information. You can subclass this and
// give instance in wxPropertyGridManager::AddPage. It inherits from
// wxEvtHandler and can be used to process events specific to this
// page (id of events will still be same as manager's). If you don't
// want to use it to process all events of the page, you need to
// return false in the derived wxPropertyGridPage::IsHandlingAllEvents.
//
// Please note that wxPropertyGridPage lacks many non-const property
// manipulation functions found in wxPropertyGridManager. Please use
// parent manager (m_manager member variable) when needed.
//
// Please note that most member functions are inherited and as such not
// documented on this page. This means you will probably also want to read
// wxPropertyGridInterface class reference.
//
// wxPropertyGridPage receives events emitted by its wxPropertyGridManager, but
// only those events that are specific to that page. If
// wxPropertyGridPage::IsHandlingAllEvents returns false, then unhandled
// events are sent to the manager's parent, as usual.
class WXDLLIMPEXP_PROPGRID wxPropertyGridPage : public wxEvtHandler,
                                                public wxPropertyGridInterface,
                                                public wxPropertyGridPageState
{
    friend class wxPropertyGridManager;
    wxDECLARE_CLASS(wxPropertyGridPage);
public:

    wxPropertyGridPage();
    virtual ~wxPropertyGridPage();

    // Deletes all properties on page.
    virtual void Clear() wxOVERRIDE;

    // Reduces column sizes to minimum possible that contents are still
    // visibly (naturally some margin space will be applied as well).
    // Returns minimum size for the page to still display everything.
    // This function only works properly if size of containing grid was
    // already fairly large.
    // Note that you can also get calculated column widths by calling
    // GetColumnWidth() immediately after this function returns.
    wxSize FitColumns();

    // Returns page index in manager;
    inline int GetIndex() const;

    // Returns x-coordinate position of splitter on a page.
    int GetSplitterPosition( int col = 0 ) const
        { return GetStatePtr()->DoGetSplitterPosition(col); }

    // Returns "root property". It does not have name, etc. and it is not
    // visible. It is only useful for accessing its children.
    wxPGProperty* GetRoot() const { return GetStatePtr()->DoGetRoot(); }

    // Returns pointer to contained property grid state.
    wxPropertyGridPageState* GetStatePtr()
    {
        return this;
    }

    // Returns pointer to contained property grid state.
    const wxPropertyGridPageState* GetStatePtr() const
    {
        return this;
    }

    // Returns id of the tool bar item that represents this page on
    // wxPropertyGridManager's wxToolBar.
    int GetToolId() const
    {
        return m_toolId;
    }

    // Do any member initialization in this method.
    // Notes:
    // - Called every time the page is added into a manager.
    // - You can add properties to the page here.
    virtual void Init() {}

    // Return false here to indicate unhandled events should be
    // propagated to manager's parent, as normal.
    virtual bool IsHandlingAllEvents() const { return true; }

    // Called every time page is about to be shown.
    // Useful, for instance, creating properties just-in-time.
    virtual void OnShow();

    //  Refreshes given property on page.
    virtual void RefreshProperty( wxPGProperty* p ) wxOVERRIDE;

    // Sets splitter position on page.
    // Splitter position cannot exceed grid size, and therefore setting it
    // during form creation may fail as initial grid size is often smaller
    // than desired splitter position, especially when sizers are being used.
    void SetSplitterPosition( int splitterPos, int col = 0 );

#if WXWIN_COMPATIBILITY_3_0
    // To avoid ambiguity between functions inherited
    // from both wxPropertyGridInterface and wxPropertyGridPageState
    using wxPropertyGridInterface::GetPropertyByLabel;
#endif // WXWIN_COMPATIBILITY_3_0

protected:

    // Propagate to other pages.
    virtual void DoSetSplitterPosition( int pos,
                                        int splitterColumn = 0,
                                        int flags = wxPG_SPLITTER_REFRESH ) wxOVERRIDE;

    // Page label (may be referred as name in some parts of documentation).
    // Can be set in constructor, or passed in
    // wxPropertyGridManager::AddPage(), but *not* in both.
    wxString                m_label;

    //virtual bool ProcessEvent( wxEvent& event );

    wxPropertyGridManager*  m_manager;

    // Toolbar tool id. Note that this is only valid when the tool bar
    // exists.
    int                     m_toolId;

private:
    bool                    m_isDefault; // is this base page object?

    wxDECLARE_EVENT_TABLE();
};

// -----------------------------------------------------------------------

#if wxUSE_TOOLBAR
class WXDLLIMPEXP_FWD_CORE wxToolBar;
#endif
#if wxUSE_HEADERCTRL
class wxPGHeaderCtrl;
#endif
class WXDLLIMPEXP_FWD_CORE wxStaticText;

// wxPropertyGridManager is an efficient multi-page version of wxPropertyGrid,
// which can optionally have toolbar for mode and page selection, and help
// text box.
// Use window flags to select components to include.
class WXDLLIMPEXP_PROPGRID
    wxPropertyGridManager : public wxPanel, public wxPropertyGridInterface
{
    wxDECLARE_CLASS(wxPropertyGridManager);
    friend class wxPropertyGridPage;
public:

#ifndef SWIG
    // Two step constructor.
    // Call Create when this constructor is called to build up the
    // wxPropertyGridManager.
    wxPropertyGridManager();
#endif

    // The default constructor. The styles to be used are styles valid for
    // the wxWindow.
    wxPropertyGridManager( wxWindow *parent, wxWindowID id = wxID_ANY,
                           const wxPoint& pos = wxDefaultPosition,
                           const wxSize& size = wxDefaultSize,
                           long style = wxPGMAN_DEFAULT_STYLE,
                           const wxString& name = wxASCII_STR(wxPropertyGridManagerNameStr) );

    // Destructor.
    virtual ~wxPropertyGridManager();

    // Creates new property page. Note that the first page is not created
    // automatically.
    // label - A label for the page. This may be shown as a toolbar tooltip etc.
    // bmp - Bitmap image for toolbar. If wxNullBitmap is used, then a built-in
    //   default image is used.
    // pageObj - wxPropertyGridPage instance. Manager will take ownership of this object.
    // NULL indicates that a default page instance should be created.
    // Returns pointer to created page.
    // If toolbar is used, it is highly recommended that the pages are
    // added when the toolbar is not turned off using window style flag
    // switching.
    wxPropertyGridPage* AddPage( const wxString& label = wxEmptyString,
                                 const wxBitmap& bmp = wxNullBitmap,
                                 wxPropertyGridPage* pageObj = NULL )
    {
        return InsertPage(-1, label, bmp, pageObj);
    }

    // Deletes all all properties and all pages.
    virtual void Clear() wxOVERRIDE;

    // Deletes all properties on given page.
    void ClearPage( int page );

    // Forces updating the value of property from the editor control.
    // Returns true if DoPropertyChanged was actually called.
    bool CommitChangesFromEditor( wxUint32 flags = 0 )
    {
        return m_pPropGrid->CommitChangesFromEditor(flags);
    }

    // Two step creation.
    // Whenever the control is created without any parameters, use Create to
    // actually create it. Don't access the control's public methods before
    // this is called.
    bool Create( wxWindow *parent, wxWindowID id = wxID_ANY,
                 const wxPoint& pos = wxDefaultPosition,
                 const wxSize& size = wxDefaultSize,
                 long style = wxPGMAN_DEFAULT_STYLE,
                 const wxString& name = wxASCII_STR(wxPropertyGridManagerNameStr) );

    // Enables or disables (shows/hides) categories according to parameter
    // enable.
    // Calling this may not properly update toolbar buttons.
    bool EnableCategories( bool enable )
    {
        long fl = m_windowStyle | wxPG_HIDE_CATEGORIES;
        if ( enable ) fl = m_windowStyle & ~(wxPG_HIDE_CATEGORIES);
        SetWindowStyleFlag(fl);
        return true;
    }

    // Selects page, scrolls and/or expands items to ensure that the
    // given item is visible. Returns true if something was actually done.
    bool EnsureVisible( wxPGPropArg id );

    // Returns number of columns on given page. By the default,
    // returns number of columns on current page.
    int GetColumnCount( int page = -1 ) const;

    // Returns height of the description text box.
    int GetDescBoxHeight() const;

    // Returns pointer to the contained wxPropertyGrid. This does not change
    // after wxPropertyGridManager has been created, so you can safely obtain
    // pointer once and use it for the entire lifetime of the manager instance.
    wxPropertyGrid* GetGrid()
    {
        wxASSERT(m_pPropGrid);
        return m_pPropGrid;
    }

    const wxPropertyGrid* GetGrid() const
    {
        wxASSERT(m_pPropGrid);
        return m_pPropGrid;
    }

    // Returns iterator class instance.
    // Calling this method in wxPropertyGridManager causes run-time assertion
    // failure. Please only iterate through individual pages or use
    // CreateVIterator().
    wxPropertyGridIterator GetIterator( int flags = wxPG_ITERATE_DEFAULT,
                                        wxPGProperty* firstProp = NULL )
    {
        wxFAIL_MSG( wxS("Please only iterate through individual pages ")
                    wxS("or use CreateVIterator()") );
        return wxPropertyGridInterface::GetIterator( flags, firstProp );
    }

    wxPropertyGridConstIterator
    GetIterator(int flags = wxPG_ITERATE_DEFAULT,
                wxPGProperty* firstProp = NULL) const
    {
        wxFAIL_MSG( wxS("Please only iterate through individual pages ")
                    wxS("or use CreateVIterator()") );
        return wxPropertyGridInterface::GetIterator( flags, firstProp );
    }

    // Returns iterator class instance.
    // Calling this method in wxPropertyGridManager causes run-time assertion
    // failure. Please only iterate through individual pages or use
    // CreateVIterator().
    wxPropertyGridIterator GetIterator( int flags, int startPos )
    {
        wxFAIL_MSG( wxS("Please only iterate through individual pages ")
                    wxS("or use CreateVIterator()") );

        return wxPropertyGridInterface::GetIterator( flags, startPos );
    }

    wxPropertyGridConstIterator GetIterator( int flags, int startPos ) const
    {
        wxFAIL_MSG( wxS("Please only iterate through individual pages ")
                    wxS("or use CreateVIterator()") );
        return wxPropertyGridInterface::GetIterator( flags, startPos );
    }

    // Similar to GetIterator, but instead returns wxPGVIterator instance,
    // which can be useful for forward-iterating through arbitrary property
    // containers.
    virtual wxPGVIterator GetVIterator( int flags ) const wxOVERRIDE;

    // Returns currently selected page.
    wxPropertyGridPage* GetCurrentPage() const
    {
        return GetPage(m_selPage);
    }

    // Returns page object for given page index.
    wxPropertyGridPage* GetPage( unsigned int ind ) const
    {
        return m_arrPages[ind];
    }

    // Returns page object for given page name.
    wxPropertyGridPage* GetPage( const wxString& name ) const
    {
        return GetPage(GetPageByName(name));
    }

    // Returns index for a page name.
    // If no match is found, wxNOT_FOUND is returned.
    int GetPageByName( const wxString& name ) const;

    // Returns index for a relevant propertygrid state.
    // If no match is found, wxNOT_FOUND is returned.
    int GetPageByState( const wxPropertyGridPageState* pstate ) const;

protected:
    // Returns wxPropertyGridPageState of given page, current page's for -1.
    virtual wxPropertyGridPageState* GetPageState( int page ) const wxOVERRIDE;

public:
    // Returns number of managed pages.
    size_t GetPageCount() const;

    // Returns name of given page.
    const wxString& GetPageName( int index ) const;

    // Returns "root property" of the given page. It does not have name, etc.
    // and it is not visible. It is only useful for accessing its children.
    wxPGProperty* GetPageRoot( int index ) const;

    // Returns index to currently selected page.
    int GetSelectedPage() const { return m_selPage; }

    // Alias for GetSelection().
    wxPGProperty* GetSelectedProperty() const
    {
        return GetSelection();
    }

    // Shortcut for GetGrid()->GetSelection().
    wxPGProperty* GetSelection() const
    {
        return m_pPropGrid->GetSelection();
    }

#if wxUSE_TOOLBAR
    // Returns a pointer to the toolbar currently associated with the
    // wxPropertyGridManager (if any).
    wxToolBar* GetToolBar() const { return m_pToolbar; }
#endif // wxUSE_TOOLBAR

    // Creates new property page. Note that the first page is not created
    // automatically.
    // index - Add to this position. -1 will add as the last item.
    // label - A label for the page. This may be shown as a toolbar tooltip etc.
    // bmp - Bitmap image for toolbar. If wxNullBitmap is used, then a built-in
    //   default image is used.
    // pageObj - wxPropertyGridPage instance. Manager will take ownership of this object.
    //   If NULL, default page object is constructed.
    // Returns pointer to created page.
    virtual wxPropertyGridPage* InsertPage( int index,
                                            const wxString& label,
                                            const wxBitmap& bmp = wxNullBitmap,
                                            wxPropertyGridPage* pageObj = NULL );

    // Returns true if any property on any page has been modified by the user.
    bool IsAnyModified() const;

    // Returns true if any property on given page has been modified by the
    // user.
    bool IsPageModified( size_t index ) const;

    // Returns true if property is selected. Since selection is page
    // based, this function checks every page in the manager.
    virtual bool IsPropertySelected( wxPGPropArg id ) const;

    virtual void Refresh( bool eraseBackground = true,
                          const wxRect* rect = (const wxRect*) NULL ) wxOVERRIDE;

    // Removes a page.
    // Returns false if it was not possible to remove page in question.
    virtual bool RemovePage( int page );

    // Select and displays a given page.
    // index - Index of page being selected. Can be -1 to select nothing.
    void SelectPage( int index );

    // Select and displays a given page (by label).
    void SelectPage( const wxString& label )
    {
        int index = GetPageByName(label);
        wxCHECK_RET( index >= 0, wxS("No page with such name") );
        SelectPage( index );
    }

    // Select and displays a given page.
    void SelectPage( wxPropertyGridPage* ptr )
    {
        SelectPage( GetPageByState(ptr) );
    }

    // Select a property.
    bool SelectProperty( wxPGPropArg id, bool focus = false )
    {
        wxPG_PROP_ARG_CALL_PROLOG_RETVAL(false)
        unsigned int flags = wxPG_SEL_DONT_SEND_EVENT;
        if ( focus )
            flags |= wxPG_SEL_FOCUS;

        return p->GetParentState()->DoSelectProperty(p, flags);
    }

#if wxUSE_HEADERCTRL
    // Sets a column title. Default title for column 0 is "Property",
    // and "Value" for column 1.
    // If header is not shown yet, then calling this
    // member function will make it visible.
    void SetColumnTitle( int idx, const wxString& title );
#endif // wxUSE_HEADERCTRL

    // Sets number of columns on given page (default is current page).
    // If you use header, then you should always use this
    // member function to set the column count, instead of
    // ones present in wxPropertyGrid or wxPropertyGridPage.
    void SetColumnCount( int colCount, int page = -1 );

    // Sets label and text in description box.
    void SetDescription( const wxString& label, const wxString& content );

    // Sets y coordinate of the description box splitter.
    void SetDescBoxHeight( int ht, bool refresh = true );

    // Moves splitter as left as possible, while still allowing all
    // labels to be shown in full.
    // subProps - If false, will still allow sub-properties (ie. properties which
    //   parent is not root or category) to be cropped.
    // allPages - If true, takes labels on all pages into account.
    void SetSplitterLeft( bool subProps = false, bool allPages = true );

    // Moves splitter as left as possible on an individual page, while still allowing all
    // labels to be shown in full.
    void SetPageSplitterLeft(int page, bool subProps = false);

    // Sets splitter position on individual page.
    // If you use header, then you should always use this
    // member function to set the splitter position, instead of
    // ones present in wxPropertyGrid or wxPropertyGridPage.
    void SetPageSplitterPosition( int page, int pos, int column = 0 );

    // Sets splitter position for all pages.
    // Splitter position cannot exceed grid size, and therefore
    // setting it during form creation may fail as initial grid
    // size is often smaller than desired splitter position,
    // especially when sizers are being used.
    // If you use header, then you should always use this
    // member function to set the splitter position, instead of
    // ones present in wxPropertyGrid or wxPropertyGridPage.
    void SetSplitterPosition( int pos, int column = 0 );

#if wxUSE_HEADERCTRL
    // Show or hide the property grid header control. It is hidden
    // by the default.
    // Grid may look better if you use wxPG_NO_INTERNAL_BORDER
    // window style when showing a header.
    void ShowHeader(bool show = true);
#endif

protected:

    //
    // Subclassing helpers
    //

    // Creates property grid for the manager. Reimplement in derived class to
    // use subclassed wxPropertyGrid. However, if you do this then you
    // must also use the two-step construction (i.e. default constructor and
    // Create() instead of constructor with arguments) when creating the
    // manager.
    virtual wxPropertyGrid* CreatePropertyGrid() const;

public:
    virtual void RefreshProperty( wxPGProperty* p ) wxOVERRIDE;

    //
    // Overridden functions - no documentation required.
    //

    void SetId( wxWindowID winid ) wxOVERRIDE;
    virtual void SetExtraStyle ( long exStyle ) wxOVERRIDE;
    virtual bool SetFont ( const wxFont& font ) wxOVERRIDE;
    virtual void SetWindowStyleFlag ( long style ) wxOVERRIDE;
    virtual bool Reparent( wxWindowBase *newParent ) wxOVERRIDE;

protected:
    virtual wxSize DoGetBestSize() const wxOVERRIDE;

    virtual void DoFreeze() wxOVERRIDE;
    virtual void DoThaw() wxOVERRIDE;

    //
    // Event handlers
    //
    void OnMouseMove( wxMouseEvent &event );
    void OnMouseClick( wxMouseEvent &event );
    void OnMouseUp( wxMouseEvent &event );
    void OnMouseEntry( wxMouseEvent &event );

    void OnPaint( wxPaintEvent &event );

#if wxUSE_TOOLBAR
    void OnToolbarClick( wxCommandEvent &event );
#endif
    void OnResize( wxSizeEvent& event );
    void OnPropertyGridSelect( wxPropertyGridEvent& event );
    void OnPGScrollH(wxPropertyGridEvent& evt);
    void OnColWidthsChanged(wxPropertyGridEvent& evt);


    wxPropertyGrid* m_pPropGrid;

    wxVector<wxPropertyGridPage*>   m_arrPages;

#if wxUSE_TOOLBAR
    wxToolBar*      m_pToolbar;
#endif
#if wxUSE_HEADERCTRL
    wxPGHeaderCtrl* m_pHeaderCtrl;
#endif
    wxStaticText*   m_pTxtHelpCaption;
    wxStaticText*   m_pTxtHelpContent;

    wxPropertyGridPage*     m_emptyPage;

    long            m_iFlags;

    // Selected page index.
    int             m_selPage;

    int             m_width;

    int             m_height;

    int             m_extraHeight;

    int             m_splitterY;

    int             m_splitterHeight;

    int             m_dragOffset;

    wxCursor        m_cursorSizeNS;

    int             m_nextDescBoxSize;

    // Toolbar tool ids for categorized and alphabetic mode selectors.
    int             m_categorizedModeToolId;
    int             m_alphabeticModeToolId;

    unsigned char   m_dragStatus;

    bool            m_onSplitter;

    bool            m_showHeader;

    virtual wxPGProperty* DoGetPropertyByName( const wxString& name ) const wxOVERRIDE;

    // Select and displays a given page.
    virtual bool DoSelectPage( int index ) wxOVERRIDE;

    // Sets some members to defaults.
    void Init1();

    // Initializes some members.
    void Init2( int style );

/*#ifdef __WXMSW__
    virtual WXDWORD MSWGetStyle(long flags, WXDWORD *exstyle) const;
#endif*/

    virtual bool ProcessEvent( wxEvent& event ) wxOVERRIDE;

    // Recalculates new positions for components, according to the
    // given size.
    void RecalculatePositions( int width, int height );

    // (Re)creates/destroys controls, according to the window style bits.
    void RecreateControls();

    void UpdateDescriptionBox( int new_splittery, int new_width, int new_height );

    void RepaintDescBoxDecorations( wxDC& dc,
                                    int newSplitterY,
                                    int newWidth,
                                    int newHeight );

    void SetDescribedProperty( wxPGProperty* p );

    // Reimplement these to handle "descboxheight" state item
    virtual bool SetEditableStateItem( const wxString& name, wxVariant value ) wxOVERRIDE;
    virtual wxVariant GetEditableStateItem( const wxString& name ) const wxOVERRIDE;

    // Reconnect propgrid event handlers.
    void ReconnectEventHandlers(wxWindowID oldId, wxWindowID newId);
private:
    wxDECLARE_EVENT_TABLE();
};

// -----------------------------------------------------------------------

inline int wxPropertyGridPage::GetIndex() const
{
    if ( !m_manager )
        return wxNOT_FOUND;
    return m_manager->GetPageByState(this);
}

// -----------------------------------------------------------------------

#endif // wxUSE_PROPGRID

#endif // _WX_PROPGRID_MANAGER_H_
