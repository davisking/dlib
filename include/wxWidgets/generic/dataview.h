/////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/dataview.h
// Purpose:     wxDataViewCtrl generic implementation header
// Author:      Robert Roebling
// Modified By: Bo Yang
// Copyright:   (c) 1998 Robert Roebling
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef __GENERICDATAVIEWCTRLH__
#define __GENERICDATAVIEWCTRLH__

#include "wx/defs.h"
#include "wx/object.h"
#include "wx/control.h"
#include "wx/scrolwin.h"
#include "wx/icon.h"
#include "wx/vector.h"
#if wxUSE_ACCESSIBILITY
    #include "wx/access.h"
#endif // wxUSE_ACCESSIBILITY

class WXDLLIMPEXP_FWD_CORE wxDataViewMainWindow;
class WXDLLIMPEXP_FWD_CORE wxDataViewHeaderWindow;
#if wxUSE_ACCESSIBILITY
class WXDLLIMPEXP_FWD_CORE wxDataViewCtrlAccessible;
#endif // wxUSE_ACCESSIBILITY

// ---------------------------------------------------------
// wxDataViewColumn
// ---------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewColumn : public wxDataViewColumnBase
{
public:
    wxDataViewColumn(const wxString& title,
                     wxDataViewRenderer *renderer,
                     unsigned int model_column,
                     int width = wxDVC_DEFAULT_WIDTH,
                     wxAlignment align = wxALIGN_CENTER,
                     int flags = wxDATAVIEW_COL_RESIZABLE)
        : wxDataViewColumnBase(renderer, model_column),
          m_title(title)
    {
        Init(width, align, flags);
    }

    wxDataViewColumn(const wxBitmap& bitmap,
                     wxDataViewRenderer *renderer,
                     unsigned int model_column,
                     int width = wxDVC_DEFAULT_WIDTH,
                     wxAlignment align = wxALIGN_CENTER,
                     int flags = wxDATAVIEW_COL_RESIZABLE)
        : wxDataViewColumnBase(bitmap, renderer, model_column)
    {
        Init(width, align, flags);
    }

    // implement wxHeaderColumnBase methods
    virtual void SetTitle(const wxString& title) wxOVERRIDE
    {
        m_title = title;
        UpdateWidth();
    }
    virtual wxString GetTitle() const wxOVERRIDE
    {
        return m_title;
    }

    virtual void SetWidth(int width) wxOVERRIDE
    {
        // Call the actual update method, used for both automatic and "manual"
        // width changes.
        WXUpdateWidth(width);

        // Do remember the last explicitly set width: this is used to prevent
        // UpdateColumnSizes() from resizing the last column to be smaller than
        // this size.
        m_manuallySetWidth = width;
    }
    virtual int GetWidth() const wxOVERRIDE;

    virtual void SetMinWidth(int minWidth) wxOVERRIDE
    {
        m_minWidth = minWidth;
        UpdateWidth();
    }
    virtual int GetMinWidth() const wxOVERRIDE
    {
        return m_minWidth;
    }

    virtual void SetAlignment(wxAlignment align) wxOVERRIDE
    {
        m_align = align;
        UpdateDisplay();
    }
    virtual wxAlignment GetAlignment() const wxOVERRIDE
    {
        return m_align;
    }

    virtual void SetFlags(int flags) wxOVERRIDE
    {
        m_flags = flags;
        UpdateDisplay();
    }
    virtual int GetFlags() const wxOVERRIDE
    {
        return m_flags;
    }

    virtual bool IsSortKey() const wxOVERRIDE
    {
        return m_sort;
    }

    virtual void UnsetAsSortKey() wxOVERRIDE;

    virtual void SetSortOrder(bool ascending) wxOVERRIDE;

    virtual bool IsSortOrderAscending() const wxOVERRIDE
    {
        return m_sortAscending;
    }

    virtual void SetBitmap( const wxBitmap& bitmap ) wxOVERRIDE
    {
        wxDataViewColumnBase::SetBitmap(bitmap);
        UpdateWidth();
    }

    // This method is specific to the generic implementation and is used only
    // by wxWidgets itself.
    void WXUpdateWidth(int width)
    {
        if ( width == m_width )
            return;

        m_width = width;
        UpdateWidth();
    }

    // This method is also internal and called when the column is resized by
    // user interactively.
    void WXOnResize(int width);

    virtual int WXGetSpecifiedWidth() const wxOVERRIDE;

private:
    // common part of all ctors
    void Init(int width, wxAlignment align, int flags);

    // These methods forward to wxDataViewCtrl::OnColumnChange() and
    // OnColumnWidthChange() respectively, i.e. the latter is stronger than the
    // former.
    void UpdateDisplay();
    void UpdateWidth();

    // Return the effective value corresponding to the given width, handling
    // its negative values such as wxCOL_WIDTH_DEFAULT.
    int DoGetEffectiveWidth(int width) const;


    wxString m_title;
    int m_width,
        m_manuallySetWidth,
        m_minWidth;
    wxAlignment m_align;
    int m_flags;
    bool m_sort,
         m_sortAscending;

    friend class wxDataViewHeaderWindowBase;
    friend class wxDataViewHeaderWindow;
    friend class wxDataViewHeaderWindowMSW;
};

// ---------------------------------------------------------
// wxDataViewCtrl
// ---------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewCtrl : public wxDataViewCtrlBase,
                                       public wxScrollHelper
{
    friend class wxDataViewMainWindow;
    friend class wxDataViewHeaderWindowBase;
    friend class wxDataViewHeaderWindow;
    friend class wxDataViewHeaderWindowMSW;
    friend class wxDataViewColumn;
#if wxUSE_ACCESSIBILITY
    friend class wxDataViewCtrlAccessible;
#endif // wxUSE_ACCESSIBILITY

public:
    wxDataViewCtrl() : wxScrollHelper(this)
    {
        Init();
    }

    wxDataViewCtrl( wxWindow *parent, wxWindowID id,
           const wxPoint& pos = wxDefaultPosition,
           const wxSize& size = wxDefaultSize, long style = 0,
           const wxValidator& validator = wxDefaultValidator,
           const wxString& name = wxASCII_STR(wxDataViewCtrlNameStr) )
             : wxScrollHelper(this)
    {
        Create(parent, id, pos, size, style, validator, name);
    }

    virtual ~wxDataViewCtrl();

    void Init();

    bool Create(wxWindow *parent, wxWindowID id,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize, long style = 0,
                const wxValidator& validator = wxDefaultValidator,
                const wxString& name = wxASCII_STR(wxDataViewCtrlNameStr));

    virtual bool AssociateModel( wxDataViewModel *model ) wxOVERRIDE;

    virtual bool AppendColumn( wxDataViewColumn *col ) wxOVERRIDE;
    virtual bool PrependColumn( wxDataViewColumn *col ) wxOVERRIDE;
    virtual bool InsertColumn( unsigned int pos, wxDataViewColumn *col ) wxOVERRIDE;

    virtual void DoSetExpanderColumn() wxOVERRIDE;
    virtual void DoSetIndent() wxOVERRIDE;

    virtual unsigned int GetColumnCount() const wxOVERRIDE;
    virtual wxDataViewColumn* GetColumn( unsigned int pos ) const wxOVERRIDE;
    virtual bool DeleteColumn( wxDataViewColumn *column ) wxOVERRIDE;
    virtual bool ClearColumns() wxOVERRIDE;
    virtual int GetColumnPosition( const wxDataViewColumn *column ) const wxOVERRIDE;

    virtual wxDataViewColumn *GetSortingColumn() const wxOVERRIDE;
    virtual wxVector<wxDataViewColumn *> GetSortingColumns() const wxOVERRIDE;

    virtual wxDataViewItem GetTopItem() const wxOVERRIDE;
    virtual int GetCountPerPage() const wxOVERRIDE;

    virtual int GetSelectedItemsCount() const wxOVERRIDE;
    virtual int GetSelections( wxDataViewItemArray & sel ) const wxOVERRIDE;
    virtual void SetSelections( const wxDataViewItemArray & sel ) wxOVERRIDE;
    virtual void Select( const wxDataViewItem & item ) wxOVERRIDE;
    virtual void Unselect( const wxDataViewItem & item ) wxOVERRIDE;
    virtual bool IsSelected( const wxDataViewItem & item ) const wxOVERRIDE;

    virtual void SelectAll() wxOVERRIDE;
    virtual void UnselectAll() wxOVERRIDE;

    virtual void EnsureVisible( const wxDataViewItem & item,
                                const wxDataViewColumn *column = NULL ) wxOVERRIDE;
    virtual void HitTest( const wxPoint & point, wxDataViewItem & item,
                          wxDataViewColumn* &column ) const wxOVERRIDE;
    virtual wxRect GetItemRect( const wxDataViewItem & item,
                                const wxDataViewColumn *column = NULL ) const wxOVERRIDE;

    virtual bool SetRowHeight( int rowHeight ) wxOVERRIDE;

    virtual void Collapse( const wxDataViewItem & item ) wxOVERRIDE;
    virtual bool IsExpanded( const wxDataViewItem & item ) const wxOVERRIDE;

    virtual void SetFocus() wxOVERRIDE;

    virtual bool SetFont(const wxFont & font) wxOVERRIDE;

#if wxUSE_ACCESSIBILITY
    virtual bool Show(bool show = true) wxOVERRIDE;
    virtual void SetName(const wxString &name) wxOVERRIDE;
    virtual bool Reparent(wxWindowBase *newParent) wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY
    virtual bool Enable(bool enable = true) wxOVERRIDE;

    virtual bool AllowMultiColumnSort(bool allow) wxOVERRIDE;
    virtual bool IsMultiColumnSortAllowed() const wxOVERRIDE { return m_allowMultiColumnSort; }
    virtual void ToggleSortByColumn(int column) wxOVERRIDE;

#if wxUSE_DRAG_AND_DROP
    virtual bool EnableDragSource( const wxDataFormat &format ) wxOVERRIDE;
    virtual bool EnableDropTarget( const wxDataFormat &format ) wxOVERRIDE;
#endif // wxUSE_DRAG_AND_DROP

    virtual wxBorder GetDefaultBorder() const wxOVERRIDE;

    virtual void EditItem(const wxDataViewItem& item, const wxDataViewColumn *column) wxOVERRIDE;

    virtual bool SetHeaderAttr(const wxItemAttr& attr) wxOVERRIDE;

    virtual bool SetAlternateRowColour(const wxColour& colour) wxOVERRIDE;

    // This method is specific to generic wxDataViewCtrl implementation and
    // should not be used in portable code.
    wxColour GetAlternateRowColour() const { return m_alternateRowColour; }

    // The returned pointer is null if the control has wxDV_NO_HEADER style.
    //
    // This method is only available in the generic versions.
    wxHeaderCtrl* GenericGetHeader() const;

protected:
    void EnsureVisibleRowCol( int row, int column );

    // Notice that row here may be invalid (i.e. >= GetRowCount()), this is not
    // an error and this function simply returns an invalid item in this case.
    wxDataViewItem GetItemByRow( unsigned int row ) const;
    int GetRowByItem( const wxDataViewItem & item ) const;

    // Mark the column as being used or not for sorting.
    void UseColumnForSorting(int idx);
    void DontUseColumnForSorting(int idx);

    // Return true if the given column is sorted
    bool IsColumnSorted(int idx) const;

    // Reset all columns currently used for sorting.
    void ResetAllSortColumns();

    virtual void DoEnableSystemTheme(bool enable, wxWindow* window) wxOVERRIDE;

    void OnDPIChanged(wxDPIChangedEvent& event);

public:     // utility functions not part of the API

    // returns the "best" width for the idx-th column
    unsigned int GetBestColumnWidth(int idx) const;

    // called by header window after reorder
    void ColumnMoved( wxDataViewColumn* col, unsigned int new_pos );

    // update the display after a change to an individual column
    void OnColumnChange(unsigned int idx);

    // update after the column width changes due to interactive resizing
    void OnColumnResized();

    // update after the column width changes because of e.g. title or bitmap
    // change, invalidates the column best width and calls OnColumnChange()
    void OnColumnWidthChange(unsigned int idx);

    // update after a change to the number of columns
    void OnColumnsCountChanged();

    wxWindow *GetMainWindow() { return (wxWindow*) m_clientArea; }

    // return the index of the given column in m_cols
    int GetColumnIndex(const wxDataViewColumn *column) const;

    // Return the index of the column having the given model index.
    int GetModelColumnIndex(unsigned int model_column) const;

    // return the column displayed at the given position in the control
    wxDataViewColumn *GetColumnAt(unsigned int pos) const;

    virtual wxDataViewColumn *GetCurrentColumn() const wxOVERRIDE;

    virtual void OnInternalIdle() wxOVERRIDE;

#if wxUSE_ACCESSIBILITY
    virtual wxAccessible* CreateAccessible() wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY

private:
    virtual wxDataViewItem DoGetCurrentItem() const wxOVERRIDE;
    virtual void DoSetCurrentItem(const wxDataViewItem& item) wxOVERRIDE;

    virtual void DoExpand(const wxDataViewItem& item, bool expandChildren) wxOVERRIDE;

    void InvalidateColBestWidths();
    void InvalidateColBestWidth(int idx);
    void UpdateColWidths();

    void DoClearColumns();

    wxVector<wxDataViewColumn*> m_cols;
    // cached column best widths information, values are for
    // respective columns from m_cols and the arrays have same size
    struct CachedColWidthInfo
    {
        CachedColWidthInfo() : width(0), dirty(true) {}
        int width;  // cached width or 0 if not computed
        bool dirty; // column was invalidated, header needs updating
    };
    wxVector<CachedColWidthInfo> m_colsBestWidths;
    // This indicates that at least one entry in m_colsBestWidths has 'dirty'
    // flag set. It's cheaper to check one flag in OnInternalIdle() than to
    // iterate over m_colsBestWidths to check if anything needs to be done.
    bool                      m_colsDirty;

    wxDataViewModelNotifier  *m_notifier;
    wxDataViewMainWindow     *m_clientArea;
    wxDataViewHeaderWindow   *m_headerArea;

    // user defined color to draw row lines, may be invalid
    wxColour m_alternateRowColour;

    // columns indices used for sorting, empty if nothing is sorted
    wxVector<int> m_sortingColumnIdxs;

    // if true, allow sorting by more than one column
    bool m_allowMultiColumnSort;

private:
    void OnSize( wxSizeEvent &event );
    virtual wxSize GetSizeAvailableForScrollTarget(const wxSize& size) wxOVERRIDE;

    // we need to return a special WM_GETDLGCODE value to process just the
    // arrows but let the other navigation characters through
#ifdef __WXMSW__
    virtual WXLRESULT MSWWindowProc(WXUINT nMsg, WXWPARAM wParam, WXLPARAM lParam) wxOVERRIDE;
#endif // __WXMSW__

    WX_FORWARD_TO_SCROLL_HELPER()

private:
    wxDECLARE_DYNAMIC_CLASS(wxDataViewCtrl);
    wxDECLARE_NO_COPY_CLASS(wxDataViewCtrl);
    wxDECLARE_EVENT_TABLE();
};

#if wxUSE_ACCESSIBILITY
//-----------------------------------------------------------------------------
// wxDataViewCtrlAccessible
//-----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewCtrlAccessible: public wxWindowAccessible
{
public:
    wxDataViewCtrlAccessible(wxDataViewCtrl* win);
    virtual ~wxDataViewCtrlAccessible() {}

    virtual wxAccStatus HitTest(const wxPoint& pt, int* childId,
                                wxAccessible** childObject) wxOVERRIDE;

    virtual wxAccStatus GetLocation(wxRect& rect, int elementId) wxOVERRIDE;

    virtual wxAccStatus Navigate(wxNavDir navDir, int fromId,
                                 int* toId, wxAccessible** toObject) wxOVERRIDE;

    virtual wxAccStatus GetName(int childId, wxString* name) wxOVERRIDE;

    virtual wxAccStatus GetChildCount(int* childCount) wxOVERRIDE;

    virtual wxAccStatus GetChild(int childId, wxAccessible** child) wxOVERRIDE;

    // wxWindowAccessible::GetParent() implementation is enough.
    // virtual wxAccStatus GetParent(wxAccessible** parent) wxOVERRIDE;

    virtual wxAccStatus DoDefaultAction(int childId) wxOVERRIDE;

    virtual wxAccStatus GetDefaultAction(int childId, wxString* actionName) wxOVERRIDE;

    virtual wxAccStatus GetDescription(int childId, wxString* description) wxOVERRIDE;

    virtual wxAccStatus GetHelpText(int childId, wxString* helpText) wxOVERRIDE;

    virtual wxAccStatus GetKeyboardShortcut(int childId, wxString* shortcut) wxOVERRIDE;

    virtual wxAccStatus GetRole(int childId, wxAccRole* role) wxOVERRIDE;

    virtual wxAccStatus GetState(int childId, long* state) wxOVERRIDE;

    virtual wxAccStatus GetValue(int childId, wxString* strValue) wxOVERRIDE;

    virtual wxAccStatus Select(int childId, wxAccSelectionFlags selectFlags) wxOVERRIDE;

    virtual wxAccStatus GetFocus(int* childId, wxAccessible** child) wxOVERRIDE;

    virtual wxAccStatus GetSelections(wxVariant* selections) wxOVERRIDE;
};
#endif // wxUSE_ACCESSIBILITY

#endif // __GENERICDATAVIEWCTRLH__
