/////////////////////////////////////////////////////////////////////////////
// Name:        wx/richtext/richtextsymboldlg.h
// Purpose:     Declares the symbol picker dialog.
// Author:      Julian Smart
// Modified by:
// Created:     10/5/2006 3:11:58 PM
// Copyright:   (c) Julian Smart
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _RICHTEXTSYMBOLDLG_H_
#define _RICHTEXTSYMBOLDLG_H_

/*!
 * Includes
 */

#include "wx/richtext/richtextuicustomization.h"
#include "wx/dialog.h"
#include "wx/vscroll.h"

/*!
 * Forward declarations
 */

class WXDLLIMPEXP_FWD_CORE wxStaticText;
class WXDLLIMPEXP_FWD_CORE wxComboBox;
class WXDLLIMPEXP_FWD_CORE wxTextCtrl;

////@begin forward declarations
class wxSymbolListCtrl;
class wxStdDialogButtonSizer;
////@end forward declarations

// __UNICODE__ is a symbol used by DialogBlocks-generated code.
#ifndef __UNICODE__
#if wxUSE_UNICODE
#define __UNICODE__
#endif
#endif

/*!
 * Symbols
 */

#define SYMBOL_WXSYMBOLPICKERDIALOG_STYLE (wxDEFAULT_DIALOG_STYLE|wxRESIZE_BORDER|wxCLOSE_BOX)
#define SYMBOL_WXSYMBOLPICKERDIALOG_TITLE wxGetTranslation("Symbols")
#define SYMBOL_WXSYMBOLPICKERDIALOG_IDNAME ID_SYMBOLPICKERDIALOG
#define SYMBOL_WXSYMBOLPICKERDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_WXSYMBOLPICKERDIALOG_POSITION wxDefaultPosition

/*!
 * wxSymbolPickerDialog class declaration
 */

class WXDLLIMPEXP_RICHTEXT wxSymbolPickerDialog: public wxDialog
{
    wxDECLARE_DYNAMIC_CLASS(wxSymbolPickerDialog);
    wxDECLARE_EVENT_TABLE();
    DECLARE_HELP_PROVISION()

public:
    /// Constructors
    wxSymbolPickerDialog( );
    wxSymbolPickerDialog( const wxString& symbol, const wxString& fontName, const wxString& normalTextFont,
        wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& caption = SYMBOL_WXSYMBOLPICKERDIALOG_TITLE, const wxPoint& pos = SYMBOL_WXSYMBOLPICKERDIALOG_POSITION, const wxSize& size = SYMBOL_WXSYMBOLPICKERDIALOG_SIZE, long style = SYMBOL_WXSYMBOLPICKERDIALOG_STYLE );

    /// Creation
    bool Create( const wxString& symbol, const wxString& fontName, const wxString& normalTextFont,
        wxWindow* parent, wxWindowID id = wxID_ANY, const wxString& caption = SYMBOL_WXSYMBOLPICKERDIALOG_TITLE, const wxPoint& pos = SYMBOL_WXSYMBOLPICKERDIALOG_POSITION, const wxSize& size = SYMBOL_WXSYMBOLPICKERDIALOG_SIZE, long style = SYMBOL_WXSYMBOLPICKERDIALOG_STYLE );

    /// Initialises members variables
    void Init();

    /// Creates the controls and sizers
    void CreateControls();

    /// Update the display
    void UpdateSymbolDisplay(bool updateSymbolList = true, bool showAtSubset = true);

    /// Respond to symbol selection
    void OnSymbolSelected( wxCommandEvent& event );

    /// Set Unicode mode
    void SetUnicodeMode(bool unicodeMode);

    /// Show at the current subset selection
    void ShowAtSubset();

    /// Get the selected symbol character
    int GetSymbolChar() const;

    /// Is there a selection?
    bool HasSelection() const { return !m_symbol.IsEmpty(); }

    /// Specifying normal text?
    bool UseNormalFont() const { return m_fontName.IsEmpty(); }

    /// Should we show tooltips?
    static bool ShowToolTips() { return sm_showToolTips; }

    /// Determines whether tooltips will be shown
    static void SetShowToolTips(bool show) { sm_showToolTips = show; }

    /// Data transfer
    virtual bool TransferDataToWindow() wxOVERRIDE;

////@begin wxSymbolPickerDialog event handler declarations

    /// wxEVT_COMBOBOX event handler for ID_SYMBOLPICKERDIALOG_FONT
    void OnFontCtrlSelected( wxCommandEvent& event );

#if defined(__UNICODE__)
    /// wxEVT_COMBOBOX event handler for ID_SYMBOLPICKERDIALOG_SUBSET
    void OnSubsetSelected( wxCommandEvent& event );

    /// wxEVT_UPDATE_UI event handler for ID_SYMBOLPICKERDIALOG_SUBSET
    void OnSymbolpickerdialogSubsetUpdate( wxUpdateUIEvent& event );

#endif
#if defined(__UNICODE__)
    /// wxEVT_COMBOBOX event handler for ID_SYMBOLPICKERDIALOG_FROM
    void OnFromUnicodeSelected( wxCommandEvent& event );

#endif
    /// wxEVT_UPDATE_UI event handler for wxID_OK
    void OnOkUpdate( wxUpdateUIEvent& event );

    /// wxEVT_BUTTON event handler for wxID_HELP
    void OnHelpClick( wxCommandEvent& event );

    /// wxEVT_UPDATE_UI event handler for wxID_HELP
    void OnHelpUpdate( wxUpdateUIEvent& event );

////@end wxSymbolPickerDialog event handler declarations

////@begin wxSymbolPickerDialog member function declarations

    wxString GetFontName() const { return m_fontName ; }
    void SetFontName(const wxString& value) { m_fontName = value; }

    bool GetFromUnicode() const { return m_fromUnicode ; }
    void SetFromUnicode(bool value) { m_fromUnicode = value ; }

    wxString GetNormalTextFontName() const { return m_normalTextFontName ; }
    void SetNormalTextFontName(const wxString& value) { m_normalTextFontName = value; }

    wxString GetSymbol() const { return m_symbol ; }
    void SetSymbol(const wxString& value) { m_symbol = value; }

    /// Retrieves bitmap resources
    wxBitmap GetBitmapResource( const wxString& name );

    /// Retrieves icon resources
    wxIcon GetIconResource( const wxString& name );
////@end wxSymbolPickerDialog member function declarations

////@begin wxSymbolPickerDialog member variables
    wxComboBox* m_fontCtrl;
#if defined(__UNICODE__)
    wxComboBox* m_subsetCtrl;
#endif
    wxSymbolListCtrl* m_symbolsCtrl;
    wxStaticText* m_symbolStaticCtrl;
    wxTextCtrl* m_characterCodeCtrl;
#if defined(__UNICODE__)
    wxComboBox* m_fromUnicodeCtrl;
#endif
    wxStdDialogButtonSizer* m_stdButtonSizer;
    wxString m_fontName;
    bool m_fromUnicode;
    wxString m_normalTextFontName;
    wxString m_symbol;
    /// Control identifiers
    enum {
        ID_SYMBOLPICKERDIALOG = 10600,
        ID_SYMBOLPICKERDIALOG_FONT = 10602,
        ID_SYMBOLPICKERDIALOG_SUBSET = 10605,
        ID_SYMBOLPICKERDIALOG_LISTCTRL = 10608,
        ID_SYMBOLPICKERDIALOG_CHARACTERCODE = 10601,
        ID_SYMBOLPICKERDIALOG_FROM = 10603
    };
////@end wxSymbolPickerDialog member variables

    bool m_dontUpdate;
    static bool             sm_showToolTips;
};

/*!
 * The scrolling symbol list.
 */

class WXDLLIMPEXP_RICHTEXT wxSymbolListCtrl : public wxVScrolledWindow
{
public:
    // constructors and such
    // ---------------------

    // default constructor, you must call Create() later
    wxSymbolListCtrl() { Init(); }

    // normal constructor which calls Create() internally
    wxSymbolListCtrl(wxWindow *parent,
               wxWindowID id = wxID_ANY,
               const wxPoint& pos = wxDefaultPosition,
               const wxSize& size = wxDefaultSize,
               long style = 0,
               const wxString& name = wxASCII_STR(wxPanelNameStr))
    {
        Init();

        (void)Create(parent, id, pos, size, style, name);
    }

    // really creates the control and sets the initial number of items in it
    // (which may be changed later with SetItemCount())
    //
    // returns true on success or false if the control couldn't be created
    bool Create(wxWindow *parent,
                wxWindowID id = wxID_ANY,
                const wxPoint& pos = wxDefaultPosition,
                const wxSize& size = wxDefaultSize,
                long style = 0,
                const wxString& name = wxASCII_STR(wxPanelNameStr));

    // dtor does some internal cleanup
    virtual ~wxSymbolListCtrl();


    // accessors
    // ---------

    // set the current font
    virtual bool SetFont(const wxFont& font) wxOVERRIDE;

    // set Unicode/ASCII mode
    void SetUnicodeMode(bool unicodeMode);

    // get the index of the currently selected item or wxNOT_FOUND if there is no selection
    int GetSelection() const;

    // is this item selected?
    bool IsSelected(int item) const;

    // is this item the current one?
    bool IsCurrentItem(int item) const { return item == m_current; }

    // get the margins around each cell
    wxPoint GetMargins() const { return m_ptMargins; }

    // get the background colour of selected cells
    const wxColour& GetSelectionBackground() const { return m_colBgSel; }

    // operations
    // ----------

    // set the selection to the specified item, if it is wxNOT_FOUND the
    // selection is unset
    void SetSelection(int selection);

    // make this item visible
    void EnsureVisible(int item);

    // set the margins: horizontal margin is the distance between the window
    // border and the item contents while vertical margin is half of the
    // distance between items
    //
    // by default both margins are 0
    void SetMargins(const wxPoint& pt);
    void SetMargins(wxCoord x, wxCoord y) { SetMargins(wxPoint(x, y)); }

    // set the cell size
    void SetCellSize(const wxSize& sz) { m_cellSize = sz; }
    const wxSize& GetCellSize() const { return m_cellSize; }

    // change the background colour of the selected cells
    void SetSelectionBackground(const wxColour& col);

    virtual wxVisualAttributes GetDefaultAttributes() const wxOVERRIDE
    {
        return GetClassDefaultAttributes(GetWindowVariant());
    }

    static wxVisualAttributes
    GetClassDefaultAttributes(wxWindowVariant variant = wxWINDOW_VARIANT_NORMAL);

    // Get min/max symbol values
    int GetMinSymbolValue() const { return m_minSymbolValue; }
    int GetMaxSymbolValue() const { return m_maxSymbolValue; }

    // Respond to size change
    void OnSize(wxSizeEvent& event);

protected:

    // draws a line of symbols
    virtual void OnDrawItem(wxDC& dc, const wxRect& rect, size_t n) const;

    // gets the line height
    virtual wxCoord OnGetRowHeight(size_t line) const wxOVERRIDE;

    // event handlers
    void OnPaint(wxPaintEvent& event);
    void OnKeyDown(wxKeyEvent& event);
    void OnLeftDown(wxMouseEvent& event);
    void OnLeftDClick(wxMouseEvent& event);

    // common part of all ctors
    void Init();

    // send the wxEVT_LISTBOX event
    void SendSelectedEvent();

    // change the current item (in single selection listbox it also implicitly
    // changes the selection); current may be wxNOT_FOUND in which case there
    // will be no current item any more
    //
    // return true if the current item changed, false otherwise
    bool DoSetCurrent(int current);

    // flags for DoHandleItemClick
    enum
    {
        ItemClick_Shift = 1,        // item shift-clicked
        ItemClick_Ctrl  = 2,        //       ctrl
        ItemClick_Kbd   = 4         // item selected from keyboard
    };

    // common part of keyboard and mouse handling processing code
    void DoHandleItemClick(int item, int flags);

    // calculate line number from symbol value
    int SymbolValueToLineNumber(int item);

    // initialise control from current min/max values
    void SetupCtrl(bool scrollToSelection = true);

    // hit testing
    int HitTest(const wxPoint& pt);

private:
    // the current item or wxNOT_FOUND
    int m_current;

    // margins
    wxPoint     m_ptMargins;

    // the selection bg colour
    wxColour    m_colBgSel;

    // double buffer
    wxBitmap*   m_doubleBuffer;

    // cell size
    wxSize      m_cellSize;

    // minimum and maximum symbol value
    int         m_minSymbolValue;

    // minimum and maximum symbol value
    int         m_maxSymbolValue;

    // number of items per line
    int         m_symbolsPerLine;

    // Unicode/ASCII mode
    bool        m_unicodeMode;

    wxDECLARE_EVENT_TABLE();
    wxDECLARE_NO_COPY_CLASS(wxSymbolListCtrl);
    wxDECLARE_ABSTRACT_CLASS(wxSymbolListCtrl);
};

#endif
    // _RICHTEXTSYMBOLDLG_H_
