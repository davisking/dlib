/////////////////////////////////////////////////////////////////////////////
// Name:        wx/html/htmlcell.h
// Purpose:     wxHtmlCell class is used by wxHtmlWindow/wxHtmlWinParser
//              as a basic visual element of HTML page
// Author:      Vaclav Slavik
// Copyright:   (c) 1999-2003 Vaclav Slavik
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_HTMLCELL_H_
#define _WX_HTMLCELL_H_

#include "wx/defs.h"

#if wxUSE_HTML

#include "wx/html/htmltag.h"
#include "wx/html/htmldefs.h"
#include "wx/window.h"
#include "wx/brush.h"


class WXDLLIMPEXP_FWD_HTML wxHtmlWindowInterface;
class WXDLLIMPEXP_FWD_HTML wxHtmlLinkInfo;
class WXDLLIMPEXP_FWD_HTML wxHtmlCell;
class WXDLLIMPEXP_FWD_HTML wxHtmlContainerCell;


// wxHtmlSelection is data holder with information about text selection.
// Selection is defined by two positions (beginning and end of the selection)
// and two leaf(!) cells at these positions.
class WXDLLIMPEXP_HTML wxHtmlSelection
{
public:
    wxHtmlSelection()
        : m_fromPos(wxDefaultPosition), m_toPos(wxDefaultPosition),
          m_fromCharacterPos(-1), m_toCharacterPos(-1),
          m_fromCell(NULL), m_toCell(NULL),
          m_extBeforeSel(0), m_extBeforeSelEnd(0) {}

    // this version is used for the user selection defined with the mouse
    void Set(const wxPoint& fromPos, const wxHtmlCell *fromCell,
             const wxPoint& toPos, const wxHtmlCell *toCell);
    void Set(const wxHtmlCell *fromCell, const wxHtmlCell *toCell);

    const wxHtmlCell *GetFromCell() const { return m_fromCell; }
    const wxHtmlCell *GetToCell() const { return m_toCell; }

    // these values are in absolute coordinates:
    const wxPoint& GetFromPos() const { return m_fromPos; }
    const wxPoint& GetToPos() const { return m_toPos; }

    // these are From/ToCell's private data
    void ClearFromToCharacterPos() { m_toCharacterPos = m_fromCharacterPos = -1; }
    bool AreFromToCharacterPosSet() const { return m_toCharacterPos != -1 && m_fromCharacterPos != -1; }

    void SetFromCharacterPos (wxCoord pos) { m_fromCharacterPos = pos; }
    void SetToCharacterPos (wxCoord pos) { m_toCharacterPos = pos; }
    wxCoord GetFromCharacterPos () const { return m_fromCharacterPos; }
    wxCoord GetToCharacterPos () const { return m_toCharacterPos; }

    void SetExtentBeforeSelection(unsigned ext) { m_extBeforeSel = ext; }
    void SetExtentBeforeSelectionEnd(unsigned ext) { m_extBeforeSelEnd = ext; }
    unsigned GetExtentBeforeSelection() const { return m_extBeforeSel; }
    unsigned GetExtentBeforeSelectionEnd() const { return m_extBeforeSelEnd; }

    bool IsEmpty() const
        { return m_fromPos == wxDefaultPosition &&
                 m_toPos == wxDefaultPosition; }

private:
    wxPoint m_fromPos, m_toPos;
    wxCoord m_fromCharacterPos, m_toCharacterPos;
    const wxHtmlCell *m_fromCell, *m_toCell;

    // Extent of the text before selection start.
    unsigned m_extBeforeSel;

    // Extent of the text from the beginning to the selection end.
    unsigned m_extBeforeSelEnd;
};



enum wxHtmlSelectionState
{
    wxHTML_SEL_OUT,     // currently rendered cell is outside the selection
    wxHTML_SEL_IN,      // ... is inside selection
    wxHTML_SEL_CHANGING // ... is the cell on which selection state changes
};

// Selection state is passed to wxHtmlCell::Draw so that it can render itself
// differently e.g. when inside text selection or outside it.
class WXDLLIMPEXP_HTML wxHtmlRenderingState
{
public:
    wxHtmlRenderingState() : m_selState(wxHTML_SEL_OUT) { m_bgMode = wxBRUSHSTYLE_SOLID; }

    void SetSelectionState(wxHtmlSelectionState s) { m_selState = s; }
    wxHtmlSelectionState GetSelectionState() const { return m_selState; }

    void SetFgColour(const wxColour& c) { m_fgColour = c; }
    const wxColour& GetFgColour() const { return m_fgColour; }
    void SetBgColour(const wxColour& c) { m_bgColour = c; }
    const wxColour& GetBgColour() const { return m_bgColour; }
    void SetBgMode(int m) { m_bgMode = m; }
    int GetBgMode() const { return m_bgMode; }

private:
    wxHtmlSelectionState  m_selState;
    wxColour              m_fgColour, m_bgColour;
    int                   m_bgMode;
};


// HTML rendering customization. This class is used when rendering wxHtmlCells
// as a callback:
class WXDLLIMPEXP_HTML wxHtmlRenderingStyle
{
public:
    virtual ~wxHtmlRenderingStyle() {}
    virtual wxColour GetSelectedTextColour(const wxColour& clr) = 0;
    virtual wxColour GetSelectedTextBgColour(const wxColour& clr) = 0;
};

// Standard style:
class WXDLLIMPEXP_HTML wxDefaultHtmlRenderingStyle : public wxHtmlRenderingStyle
{
public:
    explicit wxDefaultHtmlRenderingStyle(const wxWindowBase* wnd = NULL)
        : m_wnd(wnd)
    {}

    virtual wxColour GetSelectedTextColour(const wxColour& clr) wxOVERRIDE;
    virtual wxColour GetSelectedTextBgColour(const wxColour& clr) wxOVERRIDE;

private:
    const wxWindowBase* const m_wnd;

    wxDECLARE_NO_COPY_CLASS(wxDefaultHtmlRenderingStyle);
};


// Information given to cells when drawing them. Contains rendering state,
// selection information and rendering style object that can be used to
// customize the output.
class WXDLLIMPEXP_HTML wxHtmlRenderingInfo
{
public:
    wxHtmlRenderingInfo()
        : m_selection(NULL),
          m_style(NULL),
          m_prevUnderlined(false)
    {
    }

    void SetSelection(wxHtmlSelection *s) { m_selection = s; }
    wxHtmlSelection *GetSelection() const { return m_selection; }

    void SetStyle(wxHtmlRenderingStyle *style) { m_style = style; }
    wxHtmlRenderingStyle& GetStyle() { return *m_style; }

    void SetCurrentUnderlined(bool u) { m_prevUnderlined = u; }
    bool WasPreviousUnderlined() const { return m_prevUnderlined; }

    wxHtmlRenderingState& GetState() { return m_state; }

protected:
    wxHtmlSelection      *m_selection;
    wxHtmlRenderingStyle *m_style;
    wxHtmlRenderingState m_state;
    bool m_prevUnderlined;
};


// Flags for wxHtmlCell::FindCellByPos
enum
{
    wxHTML_FIND_EXACT             = 1,
    wxHTML_FIND_NEAREST_BEFORE    = 2,
    wxHTML_FIND_NEAREST_AFTER     = 4
};


// Superscript/subscript/normal script mode of a cell
enum wxHtmlScriptMode
{
    wxHTML_SCRIPT_NORMAL,
    wxHTML_SCRIPT_SUB,
    wxHTML_SCRIPT_SUP
};


// ---------------------------------------------------------------------------
// wxHtmlCell
//                  Internal data structure. It represents fragments of parsed
//                  HTML page - a word, picture, table, horizontal line and so
//                  on.  It is used by wxHtmlWindow to represent HTML page in
//                  memory.
// ---------------------------------------------------------------------------


class WXDLLIMPEXP_HTML wxHtmlCell : public wxObject
{
public:
    wxHtmlCell();
    virtual ~wxHtmlCell();

    void SetParent(wxHtmlContainerCell *p) {m_Parent = p;}
    wxHtmlContainerCell *GetParent() const {return m_Parent;}

    int GetPosX() const {return m_PosX;}
    int GetPosY() const {return m_PosY;}
    int GetWidth() const {return m_Width;}

    // Returns the maximum possible length of the cell.
    // Call Layout at least once before using GetMaxTotalWidth()
    virtual int GetMaxTotalWidth() const { return m_Width; }

    int GetHeight() const {return m_Height;}
    int GetDescent() const {return m_Descent;}

    void SetScriptMode(wxHtmlScriptMode mode, long previousBase);
    wxHtmlScriptMode GetScriptMode() const { return m_ScriptMode; }
    long GetScriptBaseline() const { return m_ScriptBaseline; }

    // Formatting cells are not visible on the screen, they only alter
    // renderer's state.
    bool IsFormattingCell() const { return m_Width == 0 && m_Height == 0; }

    const wxString& GetId() const { return m_id; }
    void SetId(const wxString& id) { m_id = id; }

    // returns the link associated with this cell. The position is position
    // within the cell so it varies from 0 to m_Width, from 0 to m_Height
    virtual wxHtmlLinkInfo* GetLink(int WXUNUSED(x) = 0,
                                    int WXUNUSED(y) = 0) const
        { return m_Link; }

    // Returns cursor to be used when mouse is over the cell, can be
    // overridden by the derived classes to use a different cursor whenever the
    // mouse is over this cell.
    virtual wxCursor GetMouseCursor(wxHtmlWindowInterface *window) const;

    // Returns cursor to be used when mouse is over the given point, can be
    // overridden if the cursor should change depending on where exactly inside
    // the cell the mouse is.
    virtual wxCursor GetMouseCursorAt(wxHtmlWindowInterface *window,
                                      const wxPoint& relPos) const;

    // return next cell among parent's cells
    wxHtmlCell *GetNext() const {return m_Next;}
    // returns first child cell (if there are any, i.e. if this is container):
    virtual wxHtmlCell* GetFirstChild() const { return NULL; }

    // members writing methods
    virtual void SetPos(int x, int y) {m_PosX = x; m_PosY = y;}
    void SetLink(const wxHtmlLinkInfo& link);
    void SetNext(wxHtmlCell *cell) {m_Next = cell;}

    // 1. adjust cell's width according to the fact that maximal possible width
    //    is w.  (this has sense when working with horizontal lines, tables
    //    etc.)
    // 2. prepare layout (=fill-in m_PosX, m_PosY (and sometime m_Height)
    //    members) = place items to fit window, according to the width w
    virtual void Layout(int w);

    // renders the cell
    virtual void Draw(wxDC& WXUNUSED(dc),
                      int WXUNUSED(x), int WXUNUSED(y),
                      int WXUNUSED(view_y1), int WXUNUSED(view_y2),
                      wxHtmlRenderingInfo& WXUNUSED(info)) {}

    // proceed drawing actions in case the cell is not visible (scrolled out of
    // screen).  This is needed to change fonts, colors and so on.
    virtual void DrawInvisible(wxDC& WXUNUSED(dc),
                               int WXUNUSED(x), int WXUNUSED(y),
                               wxHtmlRenderingInfo& WXUNUSED(info)) {}

    // This method returns pointer to the FIRST cell for that
    // the condition
    // is true. It first checks if the condition is true for this
    // cell and then calls m_Next->Find(). (Note: it checks
    // all subcells if the cell is container)
    // Condition is unique condition identifier (see htmldefs.h)
    // (user-defined condition IDs should start from 10000)
    // and param is optional parameter
    // Example : m_Cell->Find(wxHTML_COND_ISANCHOR, "news");
    //   returns pointer to anchor news
    virtual const wxHtmlCell* Find(int condition, const void* param) const;


    // This function is called when mouse button is clicked over the cell.
    // Returns true if a link is clicked, false otherwise.
    //
    // 'window' is pointer to wxHtmlWindowInterface of the window which
    // generated the event.
    // HINT: if this handling is not enough for you you should use
    //       wxHtmlWidgetCell
    virtual bool ProcessMouseClick(wxHtmlWindowInterface *window,
                                   const wxPoint& pos,
                                   const wxMouseEvent& event);

    // This method is called when paginating HTML, e.g. when printing.
    //
    // On input, pagebreak contains y-coordinate of page break (i.e. the
    // horizontal line that should not be crossed by words, images etc.)
    // relative to the parent cell on entry and may be modified to request a
    // page break at a position before it if this cell cannot be divided into
    // two pieces (each one on its own page).
    //
    // Note that page break must still happen on the current page, i.e. the
    // returned value must be strictly greater than "*pagebreak - pageHeight"
    // and less or equal to "*pagebreak" for the value of pagebreak on input.
    //
    // Returned value : true if pagebreak was modified, false otherwise
    virtual bool AdjustPagebreak(int *pagebreak, int pageHeight) const;

    // Sets cell's behaviour on pagebreaks (see AdjustPagebreak). Default
    // is true - the cell can be split on two pages
    // If there is no way to fit a cell in the current page size, the cell
    // is always split, ignoring this setting.
    void SetCanLiveOnPagebreak(bool can) { m_CanLiveOnPagebreak = can; }

    // Can the line be broken before this cell?
    virtual bool IsLinebreakAllowed() const
        { return !IsFormattingCell(); }

    // Returns true for simple == terminal cells, i.e. not composite ones.
    // This if for internal usage only and may disappear in future versions!
    virtual bool IsTerminalCell() const { return true; }

    // Find a cell inside this cell positioned at the given coordinates
    // (relative to this's positions). Returns NULL if no such cell exists.
    // The flag can be used to specify whether to look for terminal or
    // nonterminal cells or both. In either case, returned cell is deepest
    // cell in cells tree that contains [x,y].
    virtual wxHtmlCell *FindCellByPos(wxCoord x, wxCoord y,
                                  unsigned flags = wxHTML_FIND_EXACT) const;

    // Returns absolute position of the cell on HTML canvas.
    // If rootCell is provided, then it's considered to be the root of the
    // hierarchy and the returned value is relative to it.
    wxPoint GetAbsPos(const wxHtmlCell *rootCell = NULL) const;

    // Returns minimum bounding rectangle of this cell in coordinates, relative
    // to the rootCell, if it is provided, or relative to the result of
    // GetRootCell() if the rootCell is NULL.
    wxRect GetRect(const wxHtmlCell *rootCell = NULL) const;

    // Returns root cell of the hierarchy (i.e. grand-grand-...-parent that
    // doesn't have a parent itself)
    wxHtmlCell *GetRootCell() const;

    // Returns first (last) terminal cell inside this cell. It may return NULL,
    // but it is rare -- only if there are no terminals in the tree.
    virtual wxHtmlCell *GetFirstTerminal() const
        { return wxConstCast(this, wxHtmlCell); }
    virtual wxHtmlCell *GetLastTerminal() const
        { return wxConstCast(this, wxHtmlCell); }

    // Returns cell's depth, i.e. how far under the root cell it is
    // (if it is the root, depth is 0)
    unsigned GetDepth() const;

    // Returns true if the cell appears before 'cell' in natural order of
    // cells (= as they are read). If cell A is (grand)parent of cell B,
    // then both A.IsBefore(B) and B.IsBefore(A) always return true.
    bool IsBefore(wxHtmlCell *cell) const;

    // Converts the cell into text representation. If sel != NULL then
    // only part of the cell inside the selection is converted.
    virtual wxString ConvertToText(wxHtmlSelection *WXUNUSED(sel)) const
        { return wxEmptyString; }

    // This method is useful for debugging, to customize it for particular cell
    // type, override GetDescription() and not this function itself.
    virtual wxString Dump(int indent = 0) const;

protected:
    // Return the description used by Dump().
    virtual wxString GetDescription() const;


    // pointer to the next cell
    wxHtmlCell *m_Next;
    // pointer to parent cell
    wxHtmlContainerCell *m_Parent;

    // dimensions of fragment (m_Descent is used to position text & images)
    int m_Width, m_Height, m_Descent;
    // position where the fragment is drawn:
    int m_PosX, m_PosY;

    // superscript/subscript/normal:
    wxHtmlScriptMode m_ScriptMode;
    long m_ScriptBaseline;

    // destination address if this fragment is hypertext link, NULL otherwise
    wxHtmlLinkInfo *m_Link;
    // true if this cell can be placed on pagebreak, false otherwise
    bool m_CanLiveOnPagebreak;
    // unique identifier of the cell, generated from "id" property of tags
    wxString m_id;

    wxDECLARE_ABSTRACT_CLASS(wxHtmlCell);
    wxDECLARE_NO_COPY_CLASS(wxHtmlCell);
};




// ----------------------------------------------------------------------------
// Inherited cells:
// ----------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// wxHtmlWordCell
//                  Single word in input stream.
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlWordCell : public wxHtmlCell
{
public:
    wxHtmlWordCell(const wxString& word, const wxDC& dc);
    void Draw(wxDC& dc, int x, int y, int view_y1, int view_y2,
              wxHtmlRenderingInfo& info) wxOVERRIDE;
    virtual wxCursor GetMouseCursor(wxHtmlWindowInterface *window) const wxOVERRIDE;
    virtual wxString ConvertToText(wxHtmlSelection *sel) const wxOVERRIDE;
    bool IsLinebreakAllowed() const wxOVERRIDE { return m_allowLinebreak; }

    void SetPreviousWord(wxHtmlWordCell *cell);

protected:
    virtual wxString GetDescription() const wxOVERRIDE;

    virtual wxString GetAllAsText() const
        { return m_Word; }
    virtual wxString GetPartAsText(int begin, int end) const
        { return m_Word.Mid(begin, end - begin); }

    void SetSelectionPrivPos(const wxDC& dc, wxHtmlSelection *s) const;
    void Split(const wxDC& dc,
               const wxPoint& selFrom, const wxPoint& selTo,
               unsigned& pos1, unsigned& pos2,
               unsigned& ext1, unsigned& ext2) const;

    wxString m_Word;
    bool     m_allowLinebreak;

    wxDECLARE_ABSTRACT_CLASS(wxHtmlWordCell);
    wxDECLARE_NO_COPY_CLASS(wxHtmlWordCell);
};


// wxHtmlWordCell specialization for storing text fragments with embedded
// '\t's; these differ from normal words in that the displayed text is
// different from the text copied to clipboard
class WXDLLIMPEXP_HTML wxHtmlWordWithTabsCell : public wxHtmlWordCell
{
public:
    wxHtmlWordWithTabsCell(const wxString& word,
                           const wxString& wordOrig,
                           size_t linepos,
                           const wxDC& dc)
        : wxHtmlWordCell(word, dc),
          m_wordOrig(wordOrig),
          m_linepos(linepos)
    {}

protected:
    virtual wxString GetAllAsText() const wxOVERRIDE;
    virtual wxString GetPartAsText(int begin, int end) const wxOVERRIDE;

    wxString m_wordOrig;
    size_t   m_linepos;
};


// Container contains other cells, thus forming tree structure of rendering
// elements. Basic code of layout algorithm is contained in this class.
class WXDLLIMPEXP_HTML wxHtmlContainerCell : public wxHtmlCell
{
public:
    explicit wxHtmlContainerCell(wxHtmlContainerCell *parent);
    virtual ~wxHtmlContainerCell();

    virtual void Layout(int w) wxOVERRIDE;
    virtual void Draw(wxDC& dc, int x, int y, int view_y1, int view_y2,
                      wxHtmlRenderingInfo& info) wxOVERRIDE;
    virtual void DrawInvisible(wxDC& dc, int x, int y,
                               wxHtmlRenderingInfo& info) wxOVERRIDE;

    virtual bool AdjustPagebreak(int *pagebreak, int pageHeight) const wxOVERRIDE;

    // insert cell at the end of m_Cells list
    void InsertCell(wxHtmlCell *cell);

    // Detach a child cell. After calling this method, it's the caller
    // responsibility to destroy this cell (possibly by calling InsertCell()
    // with it to attach it elsewhere).
    void Detach(wxHtmlCell *cell);

    // sets horizontal/vertical alignment
    void SetAlignHor(int al) {m_AlignHor = al; m_LastLayout = -1;}
    int GetAlignHor() const {return m_AlignHor;}
    void SetAlignVer(int al) {m_AlignVer = al; m_LastLayout = -1;}
    int GetAlignVer() const {return m_AlignVer;}

    // sets left-border indentation. units is one of wxHTML_UNITS_* constants
    // what is combination of wxHTML_INDENT_*
    void SetIndent(int i, int what, int units = wxHTML_UNITS_PIXELS);
    // returns the indentation. ind is one of wxHTML_INDENT_* constants
    int GetIndent(int ind) const;
    // returns type of value returned by GetIndent(ind)
    int GetIndentUnits(int ind) const;

    // sets alignment info based on given tag's params
    void SetAlign(const wxHtmlTag& tag);
    // sets floating width adjustment
    // (examples : 32 percent of parent container,
    // -15 pixels percent (this means 100 % - 15 pixels)
    void SetWidthFloat(int w, int units) {m_WidthFloat = w; m_WidthFloatUnits = units; m_LastLayout = -1;}
    void SetWidthFloat(const wxHtmlTag& tag, double pixel_scale = 1.0);
    // sets minimal height of this container.
    void SetMinHeight(int h, int align = wxHTML_ALIGN_TOP) {m_MinHeight = h; m_MinHeightAlign = align; m_LastLayout = -1;}

    void SetBackgroundColour(const wxColour& clr) {m_BkColour = clr;}
    // returns background colour (of wxNullColour if none set), so that widgets can
    // adapt to it:
    wxColour GetBackgroundColour();
    void SetBorder(const wxColour& clr1, const wxColour& clr2, int border = 1) {m_Border = border; m_BorderColour1 = clr1; m_BorderColour2 = clr2;}
    virtual wxHtmlLinkInfo* GetLink(int x = 0, int y = 0) const wxOVERRIDE;
    virtual const wxHtmlCell* Find(int condition, const void* param) const wxOVERRIDE;

    virtual bool ProcessMouseClick(wxHtmlWindowInterface *window,
                                   const wxPoint& pos,
                                   const wxMouseEvent& event) wxOVERRIDE;

    virtual wxHtmlCell* GetFirstChild() const wxOVERRIDE { return m_Cells; }

    // returns last child cell:
    wxHtmlCell* GetLastChild() const { return m_LastCell; }

    // see comment in wxHtmlCell about this method
    virtual bool IsTerminalCell() const wxOVERRIDE { return false; }

    virtual wxHtmlCell *FindCellByPos(wxCoord x, wxCoord y,
                                  unsigned flags = wxHTML_FIND_EXACT) const wxOVERRIDE;

    virtual wxHtmlCell *GetFirstTerminal() const wxOVERRIDE;
    virtual wxHtmlCell *GetLastTerminal() const wxOVERRIDE;


    // Removes indentation on top or bottom of the container (i.e. above or
    // below first/last terminal cell). For internal use only.
    virtual void RemoveExtraSpacing(bool top, bool bottom);

    // Returns the maximum possible length of the container.
    // Call Layout at least once before using GetMaxTotalWidth()
    virtual int GetMaxTotalWidth() const wxOVERRIDE { return m_MaxTotalWidth; }

    virtual wxString Dump(int indent = 0) const wxOVERRIDE;

protected:
    void UpdateRenderingStatePre(wxHtmlRenderingInfo& info,
                                 wxHtmlCell *cell) const;
    void UpdateRenderingStatePost(wxHtmlRenderingInfo& info,
                                  wxHtmlCell *cell) const;

protected:
    int m_IndentLeft, m_IndentRight, m_IndentTop, m_IndentBottom;
            // indentation of subcells. There is always m_Indent pixels
            // big space between given border of the container and the subcells
            // it m_Indent < 0 it is in PERCENTS, otherwise it is in pixels
    int m_MinHeight, m_MinHeightAlign;
        // minimal height.
    wxHtmlCell *m_Cells, *m_LastCell;
            // internal cells, m_Cells points to the first of them, m_LastCell to the last one.
            // (LastCell is needed only to speed-up InsertCell)
    int m_AlignHor, m_AlignVer;
            // alignment horizontal and vertical (left, center, right)
    int m_WidthFloat, m_WidthFloatUnits;
            // width float is used in adjustWidth
    wxColour m_BkColour;
            // background color of this container
    int m_Border;
            // border size. Draw only if m_Border > 0
    wxColour m_BorderColour1, m_BorderColour2;
            // borders color of this container
    int m_LastLayout;
            // if != -1 then call to Layout may be no-op
            // if previous call to Layout has same argument
    int m_MaxTotalWidth;
            // Maximum possible length if ignoring line wrap


    wxDECLARE_ABSTRACT_CLASS(wxHtmlContainerCell);
    wxDECLARE_NO_COPY_CLASS(wxHtmlContainerCell);
};



// ---------------------------------------------------------------------------
// wxHtmlColourCell
//                  Color changer.
// ---------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlColourCell : public wxHtmlCell
{
public:
    wxHtmlColourCell(const wxColour& clr, int flags = wxHTML_CLR_FOREGROUND) : wxHtmlCell(), m_Colour(clr) { m_Flags = flags;}
    virtual void Draw(wxDC& dc, int x, int y, int view_y1, int view_y2,
                      wxHtmlRenderingInfo& info) wxOVERRIDE;
    virtual void DrawInvisible(wxDC& dc, int x, int y,
                               wxHtmlRenderingInfo& info) wxOVERRIDE;

    virtual wxString GetDescription() const wxOVERRIDE;

protected:
    wxColour m_Colour;
    unsigned m_Flags;

    wxDECLARE_ABSTRACT_CLASS(wxHtmlColourCell);
    wxDECLARE_NO_COPY_CLASS(wxHtmlColourCell);
};




//--------------------------------------------------------------------------------
// wxHtmlFontCell
//                  Sets actual font used for text rendering
//--------------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlFontCell : public wxHtmlCell
{
public:
    wxHtmlFontCell(wxFont *font) : wxHtmlCell(), m_Font(*font) { }
    virtual void Draw(wxDC& dc, int x, int y, int view_y1, int view_y2,
                      wxHtmlRenderingInfo& info) wxOVERRIDE;
    virtual void DrawInvisible(wxDC& dc, int x, int y,
                               wxHtmlRenderingInfo& info) wxOVERRIDE;

    virtual wxString GetDescription() const wxOVERRIDE;

protected:
    wxFont m_Font;

    wxDECLARE_ABSTRACT_CLASS(wxHtmlFontCell);
    wxDECLARE_NO_COPY_CLASS(wxHtmlFontCell);
};






//--------------------------------------------------------------------------------
// wxHtmlwidgetCell
//                  This cell is connected with wxWindow object
//                  You can use it to insert windows into HTML page
//                  (buttons, input boxes etc.)
//--------------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlWidgetCell : public wxHtmlCell
{
public:
    // !!! wnd must have correct parent!
    // if w != 0 then the m_Wnd has 'floating' width - it adjust
    // it's width according to parent container's width
    // (w is percent of parent's width)
    wxHtmlWidgetCell(wxWindow *wnd, int w = 0);
    virtual ~wxHtmlWidgetCell() { m_Wnd->Destroy(); }
    virtual void Draw(wxDC& dc, int x, int y, int view_y1, int view_y2,
                      wxHtmlRenderingInfo& info) wxOVERRIDE;
    virtual void DrawInvisible(wxDC& dc, int x, int y,
                               wxHtmlRenderingInfo& info) wxOVERRIDE;
    virtual void Layout(int w) wxOVERRIDE;

protected:
    wxWindow* m_Wnd;
    int m_WidthFloat;
            // width float is used in adjustWidth (it is in percents)

    wxDECLARE_ABSTRACT_CLASS(wxHtmlWidgetCell);
    wxDECLARE_NO_COPY_CLASS(wxHtmlWidgetCell);
};



//--------------------------------------------------------------------------------
// wxHtmlLinkInfo
//                  Internal data structure. It represents hypertext link
//--------------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlLinkInfo : public wxObject
{
public:
    wxHtmlLinkInfo()
        { m_Event = NULL; m_Cell = NULL; }
    wxHtmlLinkInfo(const wxString& href, const wxString& target = wxString())
        : m_Href(href)
        , m_Target(target)
        { m_Event = NULL; m_Cell = NULL; }

    void SetEvent(const wxMouseEvent *e) { m_Event = e; }
    void SetHtmlCell(const wxHtmlCell *e) { m_Cell = e; }

    wxString GetHref() const { return m_Href; }
    wxString GetTarget() const { return m_Target; }
    const wxMouseEvent* GetEvent() const { return m_Event; }
    const wxHtmlCell* GetHtmlCell() const { return m_Cell; }

private:
    wxString m_Href, m_Target;
    const wxMouseEvent *m_Event;
    const wxHtmlCell *m_Cell;
};



// ----------------------------------------------------------------------------
// wxHtmlTerminalCellsInterator
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlTerminalCellsInterator
{
public:
    wxHtmlTerminalCellsInterator(const wxHtmlCell *from, const wxHtmlCell *to)
        : m_to(to), m_pos(from) {}

    operator bool() const { return m_pos != NULL; }
    const wxHtmlCell* operator++();
    const wxHtmlCell* operator->() const { return m_pos; }
    const wxHtmlCell* operator*() const { return m_pos; }

private:
    const wxHtmlCell *m_to, *m_pos;
};



#endif // wxUSE_HTML

#endif // _WX_HTMLCELL_H_

