/////////////////////////////////////////////////////////////////////////////
// Name:        wx/html/htmprint.h
// Purpose:     html printing classes
// Author:      Vaclav Slavik
// Created:     25/09/99
// Copyright:   (c) Vaclav Slavik
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_HTMPRINT_H_
#define _WX_HTMPRINT_H_

#include "wx/defs.h"

#if wxUSE_HTML & wxUSE_PRINTING_ARCHITECTURE

#include "wx/html/htmlcell.h"
#include "wx/html/winpars.h"
#include "wx/html/htmlfilt.h"

#include "wx/print.h"
#include "wx/printdlg.h"
#include "wx/vector.h"

#include <limits.h> // INT_MAX

//--------------------------------------------------------------------------------
// wxHtmlDCRenderer
//                  This class is capable of rendering HTML into specified
//                  portion of DC
//--------------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlDCRenderer : public wxObject
{
public:
    wxHtmlDCRenderer();
    virtual ~wxHtmlDCRenderer();

    // Following 3 methods *must* be called before any call to Render:

    // Assign DC to this render
    void SetDC(wxDC *dc, double pixel_scale = 1.0)
        { SetDC(dc, pixel_scale, pixel_scale); }
    void SetDC(wxDC *dc, double pixel_scale, double font_scale);

    // Sets size of output rectangle, in pixels. Note that you *can't* change
    // width of the rectangle between calls to Render! (You can freely change height.)
    void SetSize(int width, int height);

    // Sets the text to be displayed.
    // Basepath is base directory (html string would be stored there if it was in
    // file). It is used to determine path for loading images, for example.
    // isdir is false if basepath is filename, true if it is directory name
    // (see wxFileSystem for detailed explanation)
    void SetHtmlText(const wxString& html, const wxString& basepath = wxEmptyString, bool isdir = true);

    // Sets the HTML cell that will be rendered: this is more efficient than
    // using text as it allows to parse it only once. Note that the cell will
    // be modified by this call.
    void SetHtmlCell(wxHtmlContainerCell& cell);

    // Sets fonts to be used when displaying HTML page. (if size null then default sizes used).
    void SetFonts(const wxString& normal_face, const wxString& fixed_face, const int *sizes = NULL);

    // Sets font sizes to be relative to the given size or the system
    // default size; use either specified or default font
    void SetStandardFonts(int size = -1,
                          const wxString& normal_face = wxEmptyString,
                          const wxString& fixed_face = wxEmptyString);

    // Finds the next page break after the specified (vertical) position.
    // Returns wxNOT_FOUND if passed in position is the last page break.
    int FindNextPageBreak(int pos) const;

    // [x,y] is position of upper-left corner of printing rectangle (see SetSize)
    // from is y-coordinate of the very first visible cell
    // to is y-coordinate of the next following page break, if any
    void Render(int x, int y, int from = 0, int to = INT_MAX);

    // returns total width of the html document
    int GetTotalWidth() const;

    // returns total height of the html document
    int GetTotalHeight() const;

private:
    void DoSetHtmlCell(wxHtmlContainerCell* cell);

    wxDC *m_DC;
    wxFileSystem m_FS;
    wxHtmlWinParser m_Parser;
    wxHtmlContainerCell *m_Cells;
    int m_Width, m_Height;
    bool m_ownsCells;

    wxDECLARE_NO_COPY_CLASS(wxHtmlDCRenderer);
};





enum {
    wxPAGE_ODD,
    wxPAGE_EVEN,
    wxPAGE_ALL
};



//--------------------------------------------------------------------------------
// wxHtmlPrintout
//                  This class is derived from standard wxWidgets printout class
//                  and is used to print HTML documents.
//--------------------------------------------------------------------------------


class WXDLLIMPEXP_HTML wxHtmlPrintout : public wxPrintout
{
public:
    wxHtmlPrintout(const wxString& title = wxT("Printout"));

    void SetHtmlText(const wxString& html, const wxString &basepath = wxEmptyString, bool isdir = true);
            // prepares the class for printing this html document.
            // Must be called before using the class, in fact just after constructor
            //
            // basepath is base directory (html string would be stored there if it was in
            // file). It is used to determine path for loading images, for example.
            // isdir is false if basepath is filename, true if it is directory name
            // (see wxFileSystem for detailed explanation)

    void SetHtmlFile(const wxString &htmlfile);
            // same as SetHtmlText except that it takes regular file as the parameter

    void SetHeader(const wxString& header, int pg = wxPAGE_ALL);
    void SetFooter(const wxString& footer, int pg = wxPAGE_ALL);
            // sets header/footer for the document. The argument is interpreted as HTML document.
            // You can use macros in it:
            //   @PAGENUM@ is replaced by page number
            //   @PAGESCNT@ is replaced by total number of pages
            //
            // pg is one of wxPAGE_ODD, wxPAGE_EVEN and wx_PAGE_ALL constants.
            // You can set different header/footer for odd and even pages

    // Sets fonts to be used when displaying HTML page. (if size null then default sizes used).
    void SetFonts(const wxString& normal_face, const wxString& fixed_face, const int *sizes = NULL);

    // Sets font sizes to be relative to the given size or the system
    // default size; use either specified or default font
    void SetStandardFonts(int size = -1,
                          const wxString& normal_face = wxEmptyString,
                          const wxString& fixed_face = wxEmptyString);

    void SetMargins(float top = 25.2f, float bottom = 25.2f, float left = 25.2f, float right = 25.2f,
                    float spaces = 5);
            // sets margins in millimeters. Defaults to 1 inch for margins and 0.5cm for space
            // between text and header and/or footer

    void SetMargins(const wxPageSetupDialogData& pageSetupData);

    // wxPrintout stuff:
    bool OnPrintPage(int page) wxOVERRIDE;
    bool HasPage(int page) wxOVERRIDE;
    void GetPageInfo(int *minPage, int *maxPage, int *selPageFrom, int *selPageTo) wxOVERRIDE;
    bool OnBeginDocument(int startPage, int endPage) wxOVERRIDE;
    void OnPreparePrinting() wxOVERRIDE;

    // Adds input filter
    static void AddFilter(wxHtmlFilter *filter);

    // Cleanup
    static void CleanUpStatics();

private:
    // this function is called by the base class OnPreparePrinting()
    // implementation and by default checks whether the document fits into
    // pageArea horizontally and warns the user if it does not and, if we're
    // going to print and not just to preview the document, giving him the
    // possibility to cancel printing
    //
    // you may override it to either suppress this check if truncation of the
    // HTML being printed is acceptable or, on the contrary, add more checks to
    // it, e.g. for the fit in the vertical direction if the document should
    // always appear on a single page
    //
    // return true if printing should go ahead or false to cancel it (the
    // return value is ignored when previewing)
    virtual bool CheckFit(const wxSize& pageArea, const wxSize& docArea) const;

    void RenderPage(wxDC *dc, int page);
            // renders one page into dc
    wxString TranslateHeader(const wxString& instr, int page);
            // substitute @PAGENUM@ and @PAGESCNT@ by real values
    void CountPages();
            // fills m_PageBreaks, which indirectly gives the number of pages


private:
    wxVector<int> m_PageBreaks;

    wxString m_Document, m_BasePath;
    bool m_BasePathIsDir;
    wxString m_Headers[2], m_Footers[2];

    int m_HeaderHeight, m_FooterHeight;
    wxHtmlDCRenderer m_Renderer, m_RendererHdr;
    float m_MarginTop, m_MarginBottom, m_MarginLeft, m_MarginRight, m_MarginSpace;

    // list of HTML filters
    static wxVector<wxHtmlFilter*> m_Filters;

    wxDECLARE_NO_COPY_CLASS(wxHtmlPrintout);
};





//--------------------------------------------------------------------------------
// wxHtmlEasyPrinting
//                  This class provides very simple interface to printing
//                  architecture. It allows you to print HTML documents only
//                  with very few commands.
//
//                  Note : do not create this class on stack only.
//                         You should create an instance on app startup and
//                         use this instance for all printing. Why? The class
//                         stores page&printer settings in it.
//--------------------------------------------------------------------------------

class WXDLLIMPEXP_HTML wxHtmlEasyPrinting : public wxObject
{
public:
    wxHtmlEasyPrinting(const wxString& name = wxT("Printing"), wxWindow *parentWindow = NULL);
    virtual ~wxHtmlEasyPrinting();

    bool PreviewFile(const wxString &htmlfile);
    bool PreviewText(const wxString &htmltext, const wxString& basepath = wxEmptyString);
            // Preview file / html-text for printing
            // (and offers printing)
            // basepath is base directory for opening subsequent files (e.g. from <img> tag)

    bool PrintFile(const wxString &htmlfile);
    bool PrintText(const wxString &htmltext, const wxString& basepath = wxEmptyString);
            // Print file / html-text w/o preview

    void PageSetup();
            // pop up printer or page setup dialog

    void SetHeader(const wxString& header, int pg = wxPAGE_ALL);
    void SetFooter(const wxString& footer, int pg = wxPAGE_ALL);
            // sets header/footer for the document. The argument is interpreted as HTML document.
            // You can use macros in it:
            //   @PAGENUM@ is replaced by page number
            //   @PAGESCNT@ is replaced by total number of pages
            //
            // pg is one of wxPAGE_ODD, wxPAGE_EVEN and wx_PAGE_ALL constants.
            // You can set different header/footer for odd and even pages

    void SetFonts(const wxString& normal_face, const wxString& fixed_face, const int* sizes = NULL);
    // Sets fonts to be used when displaying HTML page. (if size null then default sizes used)

    // Sets font sizes to be relative to the given size or the system
    // default size; use either specified or default font
    void SetStandardFonts(int size = -1,
                          const wxString& normal_face = wxEmptyString,
                          const wxString& fixed_face = wxEmptyString);

    wxPrintData *GetPrintData();
    wxPageSetupDialogData *GetPageSetupData() {return m_PageSetupData;}
            // return page setting data objects.
            // (You can set their parameters.)

    wxWindow* GetParentWindow() const { return m_ParentWindow; }
            // get the parent window
    void SetParentWindow(wxWindow* window) { m_ParentWindow = window; }
            // set the parent window

    const wxString& GetName() const { return m_Name; }
            // get the printout name
    void SetName(const wxString& name) { m_Name = name; }
            // set the printout name

    // Controls showing the dialog when printing: by default, always shown.
    enum PromptMode
    {
        Prompt_Never,
        Prompt_Once,
        Prompt_Always
    };

    void SetPromptMode(PromptMode promptMode) { m_promptMode = promptMode; }

protected:
    virtual wxHtmlPrintout *CreatePrintout();
    virtual bool DoPreview(wxHtmlPrintout *printout1, wxHtmlPrintout *printout2);
    virtual bool DoPrint(wxHtmlPrintout *printout);

private:
    wxPrintData *m_PrintData;
    wxPageSetupDialogData *m_PageSetupData;
    wxString m_Name;
    int m_FontsSizesArr[7];
    int *m_FontsSizes;
    wxString m_FontFaceFixed, m_FontFaceNormal;

    enum FontMode
    {
        FontMode_Explicit,
        FontMode_Standard
    };
    FontMode m_fontMode;

    wxString m_Headers[2], m_Footers[2];
    wxWindow *m_ParentWindow;

    PromptMode m_promptMode;

    wxDECLARE_NO_COPY_CLASS(wxHtmlEasyPrinting);
};




#endif  // wxUSE_HTML & wxUSE_PRINTING_ARCHITECTURE

#endif // _WX_HTMPRINT_H_

