/////////////////////////////////////////////////////////////////////////////
// Name:        wx/generic/notebook.h
// Purpose:     wxNotebook class (a.k.a. property sheet, tabbed dialog)
// Author:      Julian Smart
// Modified by:
// Copyright:   (c) Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_NOTEBOOK_H_
#define _WX_NOTEBOOK_H_

// ----------------------------------------------------------------------------
// headers
// ----------------------------------------------------------------------------
#include "wx/event.h"
#include "wx/control.h"

// ----------------------------------------------------------------------------
// types
// ----------------------------------------------------------------------------

// fwd declarations
class WXDLLIMPEXP_FWD_CORE wxImageList;
class WXDLLIMPEXP_FWD_CORE wxWindow;
class WXDLLIMPEXP_FWD_CORE wxTabView;

// ----------------------------------------------------------------------------
// wxNotebook
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxNotebook : public wxNotebookBase
{
public:
  // ctors
  // -----
    // default for dynamic class
  wxNotebook();
    // the same arguments as for wxControl (@@@ any special styles?)
  wxNotebook(wxWindow *parent,
             wxWindowID id,
             const wxPoint& pos = wxDefaultPosition,
             const wxSize& size = wxDefaultSize,
             long style = 0,
             const wxString& name = wxASCII_STR(wxNotebookNameStr));
    // Create() function
  bool Create(wxWindow *parent,
              wxWindowID id,
              const wxPoint& pos = wxDefaultPosition,
              const wxSize& size = wxDefaultSize,
              long style = 0,
              const wxString& name = wxASCII_STR(wxNotebookNameStr));
    // dtor
  virtual ~wxNotebook();

  // accessors
  // ---------
  // Find the position of the wxNotebookPage, wxNOT_FOUND if not found.
  int FindPagePosition(wxNotebookPage* page) const;

    // set the currently selected page, return the index of the previously
    // selected one (or wxNOT_FOUND on error)
    // NB: this function will _not_ generate wxEVT_NOTEBOOK_PAGE_xxx events
  int SetSelection(size_t nPage);
    // cycle thru the tabs
  //  void AdvanceSelection(bool bForward = true);

    // changes selected page without sending events
  int ChangeSelection(size_t nPage);

    // set/get the title of a page
  bool SetPageText(size_t nPage, const wxString& strText);
  wxString GetPageText(size_t nPage) const;

  // get the number of rows for a control with wxNB_MULTILINE style (not all
  // versions support it - they will always return 1 then)
  virtual int GetRowCount() const ;

    // sets/returns item's image index in the current image list
  int  GetPageImage(size_t nPage) const;
  bool SetPageImage(size_t nPage, int nImage);

  // control the appearance of the notebook pages
    // set the size (the same for all pages)
  void SetPageSize(const wxSize& size);
    // set the padding between tabs (in pixels)
  void SetPadding(const wxSize& padding);

    // Sets the size of the tabs (assumes all tabs are the same size)
  void SetTabSize(const wxSize& sz);

  // operations
  // ----------
    // remove one page from the notebook, and delete the page.
  bool DeletePage(size_t nPage);
  bool DeletePage(wxNotebookPage* page);
    // remove one page from the notebook, without deleting the page.
  bool RemovePage(size_t nPage);
  bool RemovePage(wxNotebookPage* page);
  virtual wxWindow* DoRemovePage(size_t nPage);

    // remove all pages
  bool DeleteAllPages();
    // the same as AddPage(), but adds it at the specified position
  bool InsertPage(size_t nPage,
                  wxNotebookPage *pPage,
                  const wxString& strText,
                  bool bSelect = false,
                  int imageId = NO_IMAGE);

  // callbacks
  // ---------
  void OnSize(wxSizeEvent& event);
  void OnInternalIdle();
  void OnSelChange(wxBookCtrlEvent& event);
  void OnSetFocus(wxFocusEvent& event);
  void OnNavigationKey(wxNavigationKeyEvent& event);

  // base class virtuals
  // -------------------
  virtual void Command(wxCommandEvent& event);
  virtual void SetConstraintSizes(bool recurse = true);
  virtual bool DoPhase(int nPhase);

  virtual wxSize CalcSizeFromPage(const wxSize& sizePage) const;

  // Implementation

  // wxNotebook on Motif uses a generic wxTabView to implement itself.
  wxTabView *GetTabView() const { return m_tabView; }
  void SetTabView(wxTabView *v) { m_tabView = v; }

  void OnMouseEvent(wxMouseEvent& event);
  void OnPaint(wxPaintEvent& event);

  virtual wxRect GetAvailableClientSize();

  // Implementation: calculate the layout of the view rect
  // and resize the children if required
  bool RefreshLayout(bool force = true);

protected:
  // common part of all ctors
  void Init();

  // helper functions
  void ChangePage(int nOldSel, int nSel); // change pages

  wxTabView*   m_tabView;

  wxDECLARE_DYNAMIC_CLASS(wxNotebook);
  wxDECLARE_EVENT_TABLE();
};

#endif // _WX_NOTEBOOK_H_
