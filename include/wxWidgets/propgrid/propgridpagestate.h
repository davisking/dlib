/////////////////////////////////////////////////////////////////////////////
// Name:        wx/propgrid/propgridpagestate.h
// Purpose:     wxPropertyGridPageState class
// Author:      Jaakko Salli
// Modified by:
// Created:     2008-08-24
// Copyright:   (c) Jaakko Salli
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_PROPGRID_PROPGRIDPAGESTATE_H_
#define _WX_PROPGRID_PROPGRIDPAGESTATE_H_

#include "wx/defs.h"

#if wxUSE_PROPGRID

#include "wx/propgrid/property.h"

// -----------------------------------------------------------------------

// A return value from wxPropertyGrid::HitTest(),
// contains all you need to know about an arbitrary location on the grid.
class WXDLLIMPEXP_PROPGRID wxPropertyGridHitTestResult
{
    friend class wxPropertyGridPageState;
public:
    wxPropertyGridHitTestResult()
    {
        m_property = NULL;
        m_column = -1;
        m_splitter = -1;
        m_splitterHitOffset = 0;
    }

    ~wxPropertyGridHitTestResult()
    {
    }

    // Returns column hit. -1 for margin.
    int GetColumn() const { return m_column; }

    // Returns property hit. NULL if empty space below
    // properties was hit instead.
    wxPGProperty* GetProperty() const
    {
        return m_property;
    }

    // Returns index of splitter hit, -1 for none.
    int GetSplitter() const { return m_splitter; }

    // If splitter hit, then this member function
    // returns offset to the exact splitter position.
    int GetSplitterHitOffset() const { return m_splitterHitOffset; }

private:
    // Property. NULL if empty space below properties was hit.
    wxPGProperty*   m_property;

    // Column. -1 for margin.
    int             m_column;

    // Index of splitter hit, -1 for none.
    int             m_splitter;

    // If splitter hit, offset to that.
    int             m_splitterHitOffset;
};

// -----------------------------------------------------------------------

#define wxPG_IT_CHILDREN(A)         ((A)<<16)

// NOTES: At lower 16-bits, there are flags to check if item will be included.
// At higher 16-bits, there are same flags, but to instead check if children
// will be included.
enum wxPG_ITERATOR_FLAGS
{

// Iterate through 'normal' property items (does not include children of
// aggregate or hidden items by default).
wxPG_ITERATE_PROPERTIES = wxPG_PROP_PROPERTY |
                          wxPG_PROP_MISC_PARENT |
                          wxPG_PROP_AGGREGATE |
                          wxPG_PROP_COLLAPSED |
                          wxPG_IT_CHILDREN(wxPG_PROP_MISC_PARENT) |
                          wxPG_IT_CHILDREN(wxPG_PROP_CATEGORY),

// Iterate children of collapsed parents, and individual items that are hidden.
wxPG_ITERATE_HIDDEN = wxPG_PROP_HIDDEN |
                      wxPG_IT_CHILDREN(wxPG_PROP_COLLAPSED),

// Iterate children of parent that is an aggregate property (ie has fixed
// children).
wxPG_ITERATE_FIXED_CHILDREN = wxPG_IT_CHILDREN(wxPG_PROP_AGGREGATE) |
                              wxPG_ITERATE_PROPERTIES,

// Iterate categories.
// Note that even without this flag, children of categories are still iterated
// through.
wxPG_ITERATE_CATEGORIES = wxPG_PROP_CATEGORY |
                          wxPG_IT_CHILDREN(wxPG_PROP_CATEGORY) |
                          wxPG_PROP_COLLAPSED,

wxPG_ITERATE_ALL_PARENTS = wxPG_PROP_MISC_PARENT |
                           wxPG_PROP_AGGREGATE |
                           wxPG_PROP_CATEGORY,

wxPG_ITERATE_ALL_PARENTS_RECURSIVELY = wxPG_ITERATE_ALL_PARENTS |
                                       wxPG_IT_CHILDREN(
                                                wxPG_ITERATE_ALL_PARENTS),

wxPG_ITERATOR_FLAGS_ALL = wxPG_PROP_PROPERTY |
                          wxPG_PROP_MISC_PARENT |
                          wxPG_PROP_AGGREGATE |
                          wxPG_PROP_HIDDEN |
                          wxPG_PROP_CATEGORY |
                          wxPG_PROP_COLLAPSED,

wxPG_ITERATOR_MASK_OP_ITEM = wxPG_ITERATOR_FLAGS_ALL,

// (wxPG_PROP_MISC_PARENT|wxPG_PROP_AGGREGATE|wxPG_PROP_CATEGORY)
wxPG_ITERATOR_MASK_OP_PARENT = wxPG_ITERATOR_FLAGS_ALL,

// Combines all flags needed to iterate through visible properties
// (ie. hidden properties and children of collapsed parents are skipped).
wxPG_ITERATE_VISIBLE = wxPG_ITERATE_PROPERTIES |
                       wxPG_PROP_CATEGORY |
                       wxPG_IT_CHILDREN(wxPG_PROP_AGGREGATE),

// Iterate all items.
wxPG_ITERATE_ALL = wxPG_ITERATE_VISIBLE |
                   wxPG_ITERATE_HIDDEN,

// Iterate through individual properties (ie categories and children of
// aggregate properties are skipped).
wxPG_ITERATE_NORMAL = wxPG_ITERATE_PROPERTIES |
                      wxPG_ITERATE_HIDDEN,

// Default iterator flags.
wxPG_ITERATE_DEFAULT = wxPG_ITERATE_NORMAL

};


#define wxPG_ITERATOR_CREATE_MASKS(FLAGS, A, B) \
    A = (FLAGS ^ wxPG_ITERATOR_MASK_OP_ITEM) & \
        wxPG_ITERATOR_MASK_OP_ITEM & 0xFFFF; \
    B = ((FLAGS>>16) ^ wxPG_ITERATOR_MASK_OP_PARENT) & \
        wxPG_ITERATOR_MASK_OP_PARENT & 0xFFFF;


// Macro to test if children of PWC should be iterated through
#define wxPG_ITERATOR_PARENTEXMASK_TEST(PWC, PARENTMASK) \
        ( \
        !PWC->HasFlag(PARENTMASK) && \
        PWC->GetChildCount() \
        )


// Base for wxPropertyGridIterator classes.
class WXDLLIMPEXP_PROPGRID wxPropertyGridIteratorBase
{
public:
    wxPropertyGridIteratorBase()
    {
    }

    void Assign( const wxPropertyGridIteratorBase& it );

    bool AtEnd() const { return m_property == NULL; }

    // Get current property.
    wxPGProperty* GetProperty() const { return m_property; }

    void Init( wxPropertyGridPageState* state,
               int flags,
               wxPGProperty* property,
               int dir = 1 );

    void Init( wxPropertyGridPageState* state,
               int flags,
               int startPos = wxTOP,
               int dir = 0 );

    // Iterate to the next property.
    void Next( bool iterateChildren = true );

    // Iterate to the previous property.
    void Prev();

    // Set base parent, i.e. a property when, in which iteration returns, it
    // ends.
    // Default base parent is the root of the used wxPropertyGridPageState.
    void SetBaseParent( wxPGProperty* baseParent )
        { m_baseParent = baseParent; }

protected:

    wxPGProperty*               m_property;

private:
    wxPropertyGridPageState*        m_state;
    wxPGProperty*               m_baseParent;

    // Masks are used to quickly exclude items
    wxPGProperty::FlagType      m_itemExMask;
    wxPGProperty::FlagType      m_parentExMask;
};

template <typename PROPERTY, typename STATE>
class wxPGIterator : public wxPropertyGridIteratorBase
{
public:
    wxPGIterator(STATE* state, int flags = wxPG_ITERATE_DEFAULT,
                 PROPERTY* property = NULL, int dir = 1)
        : wxPropertyGridIteratorBase()
    {
        Init(const_cast<wxPropertyGridPageState*>(state), flags, const_cast<wxPGProperty*>(property), dir);
    }
    wxPGIterator(STATE* state, int flags, int startPos, int dir = 0)
        : wxPropertyGridIteratorBase()
    {
        Init(const_cast<wxPropertyGridPageState*>(state), flags, startPos, dir);
    }
    wxPGIterator()
        : wxPropertyGridIteratorBase()
    {
        m_property = NULL;
    }
    wxPGIterator(const wxPGIterator& it)
        : wxPropertyGridIteratorBase()
    {
        Assign(it);
    }
    ~wxPGIterator()
    {
    }
    wxPGIterator& operator=(const wxPGIterator& it)
    {
        if ( this != &it )
            Assign(it);
        return *this;
    }

    wxPGIterator& operator++() { Next(); return *this; }
    wxPGIterator operator++(int) { wxPGIterator it = *this; Next(); return it; }
    wxPGIterator& operator--() { Prev(); return *this; }
    wxPGIterator operator--(int) { wxPGIterator it = *this; Prev(); return it; }
    PROPERTY* operator *() const { return const_cast<PROPERTY*>(m_property); }
    static PROPERTY* OneStep(STATE* state, int flags = wxPG_ITERATE_DEFAULT,
                             PROPERTY* property = NULL, int dir = 1)
    {
        wxPGIterator it(state, flags, property, dir);
        if ( property )
        {
            if ( dir == 1 )
                it.Next();
            else
                it.Prev();
        }
        return *it;
    }
};

// Preferable way to iterate through contents of wxPropertyGrid,
// wxPropertyGridManager, and wxPropertyGridPage.
// See wxPropertyGridInterface::GetIterator() for more information about usage.
typedef wxPGIterator<wxPGProperty, wxPropertyGridPageState> wxPropertyGridIterator;
typedef wxPGIterator<const wxPGProperty, const wxPropertyGridPageState> wxPropertyGridConstIterator;

// -----------------------------------------------------------------------

// Base class to derive new viterators.
class WXDLLIMPEXP_PROPGRID wxPGVIteratorBase : public wxObjectRefData
{
    friend class wxPGVIterator;
public:
    wxPGVIteratorBase() { }
    virtual void Next() = 0;
protected:
    virtual ~wxPGVIteratorBase() { }

    wxPropertyGridIterator  m_it;
};

// Abstract implementation of a simple iterator. Can only be used
// to iterate in forward order, and only through the entire container.
// Used to have functions dealing with all properties work with both
// wxPropertyGrid and wxPropertyGridManager.
class WXDLLIMPEXP_PROPGRID wxPGVIterator
{
public:
    wxPGVIterator() { m_pIt = NULL; }
    wxPGVIterator( wxPGVIteratorBase* obj ) { m_pIt = obj; }
    ~wxPGVIterator() { UnRef(); }
    void UnRef() { if (m_pIt) m_pIt->DecRef(); }
    wxPGVIterator( const wxPGVIterator& it )
    {
        m_pIt = it.m_pIt;
        m_pIt->IncRef();
    }
    const wxPGVIterator& operator=( const wxPGVIterator& it )
    {
        if (this != &it)
        {
            UnRef();
            m_pIt = it.m_pIt;
            m_pIt->IncRef();
        }
        return *this;
    }
    void Next() { m_pIt->Next(); }
    bool AtEnd() const { return m_pIt->m_it.AtEnd(); }
    wxPGProperty* GetProperty() const { return m_pIt->m_it.GetProperty(); }
protected:
    wxPGVIteratorBase*  m_pIt;
};

// -----------------------------------------------------------------------

// Contains low-level property page information (properties, column widths,
// etc.) of a single wxPropertyGrid or single wxPropertyGridPage. Generally you
// should not use this class directly, but instead member functions in
// wxPropertyGridInterface, wxPropertyGrid, wxPropertyGridPage, and
// wxPropertyGridManager.
// - Currently this class is not implemented in wxPython.
class WXDLLIMPEXP_PROPGRID wxPropertyGridPageState
{
    friend class wxPropertyGrid;
    friend class wxPropertyGridInterface;
    friend class wxPropertyGridPage;
    friend class wxPropertyGridManager;
public:

    // Default constructor.
    wxPropertyGridPageState();

    // Destructor.
    virtual ~wxPropertyGridPageState();

    // Makes sure all columns have minimum width.
    void CheckColumnWidths( int widthChange = 0 );

    // Override this member function to add custom behaviour on property
    // deletion.
    virtual void DoDelete( wxPGProperty* item, bool doDelete = true );

    wxSize DoFitColumns( bool allowGridResize = false );

    wxPGProperty* DoGetItemAtY( int y ) const;

    // Override this member function to add custom behaviour on property
    // insertion.
    virtual wxPGProperty* DoInsert( wxPGProperty* parent,
                                    int index,
                                    wxPGProperty* property );

    // This needs to be overridden in grid used the manager so that splitter
    // changes can be propagated to other pages.
    virtual void DoSetSplitterPosition( int pos,
                                        int splitterColumn = 0,
                                        int flags = 0 );

    bool EnableCategories( bool enable );

    // Make sure virtual height is up-to-date.
    void EnsureVirtualHeight()
    {
        if ( m_vhCalcPending )
        {
            RecalculateVirtualHeight();
            m_vhCalcPending = false;
        }
    }

    // Returns (precalculated) height of contained visible properties.
    unsigned int GetVirtualHeight() const
    {
        wxASSERT( !m_vhCalcPending );
        return m_virtualHeight;
    }

    // Returns (precalculated) height of contained visible properties.
    unsigned int GetVirtualHeight()
    {
        EnsureVirtualHeight();
        return m_virtualHeight;
    }

    // Returns actual height of contained visible properties.
    // Mostly used for internal diagnostic purposes.
    unsigned int GetActualVirtualHeight() const;

    unsigned int GetColumnCount() const
    {
        return (unsigned int) m_colWidths.size();
    }

    int GetColumnMinWidth( int column ) const;

    int GetColumnWidth( unsigned int column ) const
    {
        return m_colWidths[column];
    }

    wxPropertyGrid* GetGrid() const { return m_pPropGrid; }

    // Returns last item which could be iterated using given flags.
    wxPGProperty* GetLastItem( int flags = wxPG_ITERATE_DEFAULT );

    const wxPGProperty* GetLastItem( int flags = wxPG_ITERATE_DEFAULT ) const
    {
        return const_cast<wxPropertyGridPageState*>(this)->GetLastItem(flags);
    }

    // Returns currently selected property.
    wxPGProperty* GetSelection() const
    {
        return m_selection.empty()? NULL: m_selection[0];
    }

    void DoSetSelection( wxPGProperty* prop )
    {
        m_selection.clear();
        if ( prop )
            m_selection.push_back(prop);
    }

    bool DoClearSelection()
    {
        return DoSelectProperty(NULL);
    }

    void DoRemoveFromSelection( wxPGProperty* prop );

    void DoSetColumnProportion( unsigned int column, int proportion );

    int DoGetColumnProportion( unsigned int column ) const
    {
        return m_columnProportions[column];
    }

    void ResetColumnSizes( int setSplitterFlags );

    wxPropertyCategory* GetPropertyCategory( const wxPGProperty* p ) const;

#if WXWIN_COMPATIBILITY_3_0
    wxDEPRECATED_MSG("don't refer directly to wxPropertyGridPageState::GetPropertyByLabel")
    wxPGProperty* GetPropertyByLabel( const wxString& name,
                                      wxPGProperty* parent = NULL ) const;
#endif // WXWIN_COMPATIBILITY_3_0

    wxVariant DoGetPropertyValues( const wxString& listname,
                                   wxPGProperty* baseparent,
                                   long flags ) const;

    wxPGProperty* DoGetRoot() const { return m_properties; }

    void DoSetPropertyName( wxPGProperty* p, const wxString& newName );

    // Returns combined width of margin and all the columns
    int GetVirtualWidth() const
    {
        return m_width;
    }

    // Returns minimal width for given column so that all images and texts
    // will fit entirely.
    // Used by SetSplitterLeft() and DoFitColumns().
    int GetColumnFitWidth(const wxDC& dc,
                          wxPGProperty* pwc,
                          unsigned int col,
                          bool subProps) const;

    int GetColumnFullWidth(const wxDC& dc, wxPGProperty* p, unsigned int col);

    // Returns information about arbitrary position in the grid.
    // pt - Logical coordinates in the virtual grid space. Use
    //   wxScrolled<T>::CalcUnscrolledPosition() if you need to
    //   translate a scrolled position into a logical one.
    wxPropertyGridHitTestResult HitTest( const wxPoint& pt ) const;

    // Returns true if page is visibly displayed.
    bool IsDisplayed() const;

    bool IsInNonCatMode() const { return (bool)(m_properties == m_abcArray); }

    void DoLimitPropertyEditing( wxPGProperty* p, bool limit = true )
    {
        p->SetFlagRecursively(wxPG_PROP_NOEDITOR, limit);
    }

    bool DoSelectProperty( wxPGProperty* p, unsigned int flags = 0 );

    // widthChange is non-client.
    void OnClientWidthChange( int newWidth,
                              int widthChange,
                              bool fromOnResize = false );

    // Recalculates m_virtualHeight.
    void RecalculateVirtualHeight()
    {
        m_virtualHeight = GetActualVirtualHeight();
    }

    void SetColumnCount( int colCount );

    void PropagateColSizeDec( int column, int decrease, int dir );

    bool DoHideProperty( wxPGProperty* p, bool hide, int flags = wxPG_RECURSE );

    bool DoSetPropertyValueString( wxPGProperty* p, const wxString& value );

    bool DoSetPropertyValue( wxPGProperty* p, wxVariant& value );

    bool DoSetPropertyValueWxObjectPtr( wxPGProperty* p, wxObject* value );
    void DoSetPropertyValues( const wxVariantList& list,
                              wxPGProperty* default_category );

    void SetSplitterLeft( bool subProps = false );

    // Set virtual width for this particular page.
    void SetVirtualWidth( int width );

    void DoSortChildren( wxPGProperty* p, int flags = 0 );
    void DoSort( int flags = 0 );

    bool PrepareAfterItemsAdded();

    // Called after virtual height needs to be recalculated.
    void VirtualHeightChanged()
    {
        m_vhCalcPending = true;
    }

    // Base append.
    wxPGProperty* DoAppend( wxPGProperty* property );

    // Returns property by its name.
    wxPGProperty* BaseGetPropertyByName( const wxString& name ) const;

    // Called in, for example, wxPropertyGrid::Clear.
    void DoClear();

    bool DoIsPropertySelected( wxPGProperty* prop ) const;

    bool DoCollapse( wxPGProperty* p );

    bool DoExpand( wxPGProperty* p );

    void CalculateFontAndBitmapStuff( int vspacing );

protected:

    // Utility to check if two properties are visibly next to each other
    bool ArePropertiesAdjacent( wxPGProperty* prop1,
                                wxPGProperty* prop2,
                                int iterFlags = wxPG_ITERATE_VISIBLE ) const;

    int DoGetSplitterPosition( int splitterIndex = 0 ) const;

    // Returns column at x coordinate (in GetGrid()->GetPanel()).
    // pSplitterHit - Give pointer to int that receives index to splitter that is at x.
    // pSplitterHitOffset - Distance from said splitter.
    int HitTestH( int x, int* pSplitterHit, int* pSplitterHitOffset ) const;

    bool PrepareToAddItem( wxPGProperty* property,
                           wxPGProperty* scheduledParent );

    // Returns property by its label.
    wxPGProperty* BaseGetPropertyByLabel( const wxString& label,
                                      const wxPGProperty* parent = NULL ) const;

    // Unselect sub-properties.
    void DoRemoveChildrenFromSelection(wxPGProperty* p, bool recursive,
                                       int selFlags);

    // Mark sub-properties as being deleted.
    void DoMarkChildrenAsDeleted(wxPGProperty* p, bool recursive);

    // Rename the property
    // so it won't remain in the way of the user code.
    void DoInvalidatePropertyName(wxPGProperty* p);

    // Rename sub-properties
    // so it won't remain in the way of the user code.
    void DoInvalidateChildrenNames(wxPGProperty* p, bool recursive);

    // Check if property contains given sub-category.
    bool IsChildCategory(wxPGProperty* p,
                         wxPropertyCategory* cat, bool recursive);

    // If visible, then this is pointer to wxPropertyGrid.
    // This shall *never* be NULL to indicate that this state is not visible.
    wxPropertyGrid*             m_pPropGrid;

    // Pointer to currently used array.
    wxPGProperty*               m_properties;

    // Array for categoric mode.
    wxPGRootProperty            m_regularArray;

    // Array for root of non-categoric mode.
    wxPGRootProperty*           m_abcArray;

    // Dictionary for name-based access.
    wxPGHashMapS2P              m_dictName;

    // List of column widths (first column does not include margin).
    wxVector<int>               m_colWidths;

    // List of indices of columns the user can edit by clicking it.
    wxVector<int>               m_editableColumns;

    // Column proportions.
    wxVector<int>               m_columnProportions;

    double                      m_fSplitterX;

    // Most recently added category.
    wxPropertyCategory*         m_currentCategory;

    // Array of selected property.
    wxArrayPGProperty           m_selection;

    // Virtual width.
    int                         m_width;

    // Indicates total virtual height of visible properties.
    unsigned int                m_virtualHeight;

#if WXWIN_COMPATIBILITY_3_0
    // 1 items appended/inserted, so stuff needs to be done before drawing;
    // If m_virtualHeight == 0, then calcylatey's must be done.
    // Otherwise just sort.
    unsigned char               m_itemsAdded;

    // 1 if any value is modified.
    unsigned char               m_anyModified;

    unsigned char               m_vhCalcPending;
#else
    // True: items appended/inserted, so stuff needs to be done before drawing;
    // If m_virtualHeight == 0, then calcylatey's must be done.
    // Otherwise just sort.
    bool                        m_itemsAdded;

    // True if any value is modified.
    bool                        m_anyModified;

    bool                        m_vhCalcPending;
#endif // WXWIN_COMPATIBILITY_3_0

    // True if splitter has been pre-set by the application.
    bool                        m_isSplitterPreSet;

    // Used to (temporarily) disable splitter centering.
    bool                        m_dontCenterSplitter;

private:
    // Only inits arrays, doesn't migrate things or such.
    void InitNonCatMode();
};

// -----------------------------------------------------------------------

#endif // wxUSE_PROPGRID

#endif // _WX_PROPGRID_PROPGRIDPAGESTATE_H_
