///////////////////////////////////////////////////////////////////////////////
// Name:        wx/dvrenderers.h
// Purpose:     Declare all wxDataViewCtrl classes
// Author:      Robert Roebling, Vadim Zeitlin
// Created:     2009-11-08 (extracted from wx/dataview.h)
// Copyright:   (c) 2006 Robert Roebling
//              (c) 2009 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_DVRENDERERS_H_
#define _WX_DVRENDERERS_H_

/*
    Note about the structure of various headers: they're organized in a more
    complicated way than usual because of the various dependencies which are
    different for different ports. In any case the only public header, i.e. the
    one which can be included directly is wx/dataview.h. It, in turn, includes
    this one to define all the renderer classes.

    We define the base wxDataViewRendererBase class first and then include a
    port-dependent wx/xxx/dvrenderer.h which defines wxDataViewRenderer itself.
    After this we can define wxDataViewRendererCustomBase (and maybe in the
    future base classes for other renderers if the need arises, i.e. if there
    is any non-trivial code or API which it makes sense to keep in common code)
    and include wx/xxx/dvrenderers.h (notice the plural) which defines all the
    rest of the renderer classes.
 */

class WXDLLIMPEXP_FWD_CORE wxDataViewCustomRenderer;

// ----------------------------------------------------------------------------
// wxDataViewIconText: helper class used by wxDataViewIconTextRenderer
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewIconText : public wxObject
{
public:
    wxDataViewIconText( const wxString &text = wxEmptyString,
                        const wxIcon& icon = wxNullIcon )
        : m_text(text),
          m_icon(icon)
    { }

    void SetText( const wxString &text ) { m_text = text; }
    wxString GetText() const             { return m_text; }
    void SetIcon( const wxIcon &icon )   { m_icon = icon; }
    const wxIcon &GetIcon() const        { return m_icon; }

    bool IsSameAs(const wxDataViewIconText& other) const
    {
        return m_text == other.m_text && m_icon.IsSameAs(other.m_icon);
    }

    bool operator==(const wxDataViewIconText& other) const
    {
        return IsSameAs(other);
    }

    bool operator!=(const wxDataViewIconText& other) const
    {
        return !IsSameAs(other);
    }

private:
    wxString    m_text;
    wxIcon      m_icon;

    wxDECLARE_DYNAMIC_CLASS(wxDataViewIconText);
};

DECLARE_VARIANT_OBJECT_EXPORTED(wxDataViewIconText, WXDLLIMPEXP_CORE)

// ----------------------------------------------------------------------------
// wxDataViewCheckIconText: value class used by wxDataViewCheckIconTextRenderer
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewCheckIconText : public wxDataViewIconText
{
public:
    wxDataViewCheckIconText(const wxString& text = wxString(),
                            const wxIcon& icon = wxNullIcon,
                            wxCheckBoxState checkedState = wxCHK_UNDETERMINED)
        : wxDataViewIconText(text, icon),
          m_checkedState(checkedState)
    {
    }

    wxCheckBoxState GetCheckedState() const { return m_checkedState; }
    void SetCheckedState(wxCheckBoxState state) { m_checkedState = state; }

private:
    wxCheckBoxState m_checkedState;

    wxDECLARE_DYNAMIC_CLASS(wxDataViewCheckIconText);
};

DECLARE_VARIANT_OBJECT_EXPORTED(wxDataViewCheckIconText, WXDLLIMPEXP_CORE)

// ----------------------------------------------------------------------------
// wxDataViewRendererBase
// ----------------------------------------------------------------------------

enum wxDataViewCellMode
{
    wxDATAVIEW_CELL_INERT,
    wxDATAVIEW_CELL_ACTIVATABLE,
    wxDATAVIEW_CELL_EDITABLE
};

enum wxDataViewCellRenderState
{
    wxDATAVIEW_CELL_SELECTED    = 1,
    wxDATAVIEW_CELL_PRELIT      = 2,
    wxDATAVIEW_CELL_INSENSITIVE = 4,
    wxDATAVIEW_CELL_FOCUSED     = 8
};

// helper for fine-tuning rendering of values depending on row's state
class WXDLLIMPEXP_CORE wxDataViewValueAdjuster
{
public:
    virtual ~wxDataViewValueAdjuster() {}

    // changes the value to have appearance suitable for highlighted rows
    virtual wxVariant MakeHighlighted(const wxVariant& value) const { return value; }
};

class WXDLLIMPEXP_CORE wxDataViewRendererBase: public wxObject
{
public:
    wxDataViewRendererBase( const wxString &varianttype,
                            wxDataViewCellMode mode = wxDATAVIEW_CELL_INERT,
                            int alignment = wxDVR_DEFAULT_ALIGNMENT );
    virtual ~wxDataViewRendererBase();

    virtual bool Validate( wxVariant& WXUNUSED(value) )
        { return true; }

    void SetOwner( wxDataViewColumn *owner )    { m_owner = owner; }
    wxDataViewColumn* GetOwner() const          { return m_owner; }

    // renderer value and attributes: SetValue() and SetAttr() are called
    // before a cell is rendered using this renderer
    virtual bool SetValue(const wxVariant& value) = 0;
    virtual bool GetValue(wxVariant& value) const = 0;
#if wxUSE_ACCESSIBILITY
    virtual wxString GetAccessibleDescription() const = 0;
#endif // wxUSE_ACCESSIBILITY

    wxString GetVariantType() const             { return m_variantType; }

    // Prepare for rendering the value of the corresponding item in the given
    // column taken from the provided non-null model.
    //
    // Notice that the column must be the same as GetOwner()->GetModelColumn(),
    // it is only passed to this method because the existing code already has
    // it and should probably be removed in the future.
    //
    // Return true if this cell is non-empty or false otherwise (and also if
    // the model returned a value of the wrong, i.e. different from our
    // GetVariantType(), type, in which case a debug error is also logged).
    bool PrepareForItem(const wxDataViewModel *model,
                        const wxDataViewItem& item,
                        unsigned column);

    // renderer properties:
    virtual void SetMode( wxDataViewCellMode mode ) = 0;
    virtual wxDataViewCellMode GetMode() const = 0;

    // NOTE: Set/GetAlignment do not take/return a wxAlignment enum but
    //       rather an "int"; that's because for rendering cells it's allowed
    //       to combine alignment flags (e.g. wxALIGN_LEFT|wxALIGN_BOTTOM)
    virtual void SetAlignment( int align ) = 0;
    virtual int GetAlignment() const = 0;

    // enable or disable (if called with wxELLIPSIZE_NONE) replacing parts of
    // the item text (hence this only makes sense for renderers showing
    // text...) with ellipsis in order to make it fit the column width
    virtual void EnableEllipsize(wxEllipsizeMode mode = wxELLIPSIZE_MIDDLE) = 0;
    void DisableEllipsize() { EnableEllipsize(wxELLIPSIZE_NONE); }

    virtual wxEllipsizeMode GetEllipsizeMode() const = 0;

    // in-place editing
    virtual bool HasEditorCtrl() const
        { return false; }
    virtual wxWindow* CreateEditorCtrl(wxWindow * WXUNUSED(parent),
                                       wxRect WXUNUSED(labelRect),
                                       const wxVariant& WXUNUSED(value))
        { return NULL; }
    virtual bool GetValueFromEditorCtrl(wxWindow * WXUNUSED(editor),
                                        wxVariant& WXUNUSED(value))
        { return false; }

    virtual bool StartEditing( const wxDataViewItem &item, wxRect labelRect );
    virtual void CancelEditing();
    virtual bool FinishEditing();

    wxWindow *GetEditorCtrl() const { return m_editorCtrl; }

    virtual bool IsCustomRenderer() const { return false; }


    // Implementation only from now on.

    // Return the alignment of this renderer if it's specified (i.e. has value
    // different from the default wxDVR_DEFAULT_ALIGNMENT) or the alignment of
    // the column it is used for otherwise.
    //
    // Unlike GetAlignment(), this always returns a valid combination of
    // wxALIGN_XXX flags (although possibly wxALIGN_NOT) and never returns
    // wxDVR_DEFAULT_ALIGNMENT.
    int GetEffectiveAlignment() const;

    // Like GetEffectiveAlignment(), but returns wxDVR_DEFAULT_ALIGNMENT if
    // the owner isn't set and GetAlignment() is default.
    int GetEffectiveAlignmentIfKnown() const;

    // Send wxEVT_DATAVIEW_ITEM_EDITING_STARTED event.
    void NotifyEditingStarted(const wxDataViewItem& item);

    // Sets the transformer for fine-tuning rendering of values depending on row's state
    void SetValueAdjuster(wxDataViewValueAdjuster *transformer)
        { delete m_valueAdjuster; m_valueAdjuster = transformer; }

protected:
    // These methods are called from PrepareForItem() and should do whatever is
    // needed for the current platform to ensure that the item is rendered
    // using the given attributes and enabled/disabled state.
    virtual void SetAttr(const wxDataViewItemAttr& attr) = 0;
    virtual void SetEnabled(bool enabled) = 0;

    // Return whether the currently rendered item is on a highlighted row
    // (typically selection with dark background). For internal use only.
    virtual bool IsHighlighted() const = 0;

    // Helper of PrepareForItem() also used in StartEditing(): returns the
    // value checking that its type matches our GetVariantType().
    wxVariant CheckedGetValue(const wxDataViewModel* model,
                              const wxDataViewItem& item,
                              unsigned column) const;

    // Validates the given value (if it is non-null) and sends (in any case)
    // ITEM_EDITING_DONE event and, finally, updates the model with the value
    // (f it is valid, of course) if the event wasn't vetoed.
    bool DoHandleEditingDone(wxVariant* value);


    wxString                m_variantType;
    wxDataViewColumn       *m_owner;
    wxWeakRef<wxWindow>     m_editorCtrl;
    wxDataViewItem          m_item; // Item being currently edited, if valid.

    wxDataViewValueAdjuster *m_valueAdjuster;

    // internal utility, may be used anywhere the window associated with the
    // renderer is required
    wxDataViewCtrl* GetView() const;

private:
    // Called from {Called,Finish}Editing() and dtor to cleanup m_editorCtrl
    void DestroyEditControl();

    wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxDataViewRendererBase);
};

// include the real wxDataViewRenderer declaration for the native ports
#ifdef wxHAS_GENERIC_DATAVIEWCTRL
    // in the generic implementation there is no real wxDataViewRenderer, all
    // renderers are custom so it's the same as wxDataViewCustomRenderer and
    // wxDataViewCustomRendererBase derives from wxDataViewRendererBase directly
    //
    // this is a rather ugly hack but unfortunately it just doesn't seem to be
    // possible to have the same class hierarchy in all ports and avoid
    // duplicating the entire wxDataViewCustomRendererBase in the generic
    // wxDataViewRenderer class (well, we could use a mix-in but this would
    // make classes hierarchy non linear and arguably even more complex)
    #define wxDataViewCustomRendererRealBase wxDataViewRendererBase
#else
    #if defined(__WXGTK20__)
        #include "wx/gtk/dvrenderer.h"
    #elif defined(__WXMAC__)
        #include "wx/osx/dvrenderer.h"
    #elif defined(__WXQT__)
        #include "wx/qt/dvrenderer.h"
    #else
        #error "unknown native wxDataViewCtrl implementation"
    #endif
    #define wxDataViewCustomRendererRealBase wxDataViewRenderer
#endif

// ----------------------------------------------------------------------------
// wxDataViewCustomRendererBase
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewCustomRendererBase
    : public wxDataViewCustomRendererRealBase
{
public:
    // Constructor must specify the usual renderer parameters which we simply
    // pass to the base class
    wxDataViewCustomRendererBase(const wxString& varianttype = wxASCII_STR("string"),
                                 wxDataViewCellMode mode = wxDATAVIEW_CELL_INERT,
                                 int align = wxDVR_DEFAULT_ALIGNMENT)
        : wxDataViewCustomRendererRealBase(varianttype, mode, align)
    {
    }


    // Render the item using the current value (returned by GetValue()).
    virtual bool Render(wxRect cell, wxDC *dc, int state) = 0;

    // Return the size of the item appropriate to its current value.
    virtual wxSize GetSize() const = 0;

    // Define virtual function which are called when a key is pressed on the
    // item, clicked or the user starts to drag it: by default they all simply
    // return false indicating that the events are not handled

    virtual bool ActivateCell(const wxRect& cell,
                              wxDataViewModel *model,
                              const wxDataViewItem & item,
                              unsigned int col,
                              const wxMouseEvent* mouseEvent);

    // Deprecated, use (and override) ActivateCell() instead
    wxDEPRECATED_BUT_USED_INTERNALLY_INLINE(
        virtual bool Activate(wxRect WXUNUSED(cell),
                              wxDataViewModel *WXUNUSED(model),
                              const wxDataViewItem & WXUNUSED(item),
                              unsigned int WXUNUSED(col)),
                          return false; )

    // Deprecated, use (and override) ActivateCell() instead
    wxDEPRECATED_BUT_USED_INTERNALLY_INLINE(
        virtual bool LeftClick(wxPoint WXUNUSED(cursor),
                               wxRect WXUNUSED(cell),
                               wxDataViewModel *WXUNUSED(model),
                               const wxDataViewItem & WXUNUSED(item),
                               unsigned int WXUNUSED(col)),
                          return false; )

    virtual bool StartDrag(const wxPoint& WXUNUSED(cursor),
                           const wxRect& WXUNUSED(cell),
                           wxDataViewModel *WXUNUSED(model),
                           const wxDataViewItem & WXUNUSED(item),
                           unsigned int WXUNUSED(col) )
        { return false; }


    // Helper which can be used by Render() implementation in the derived
    // classes: it will draw the text in the same manner as the standard
    // renderers do.
    virtual void RenderText(const wxString& text,
                            int xoffset,
                            wxRect cell,
                            wxDC *dc,
                            int state);


    // Override the base class virtual method to simply store the attribute so
    // that it can be accessed using GetAttr() from Render() if needed.
    virtual void SetAttr(const wxDataViewItemAttr& attr) wxOVERRIDE { m_attr = attr; }
    const wxDataViewItemAttr& GetAttr() const { return m_attr; }

    // Store the enabled state of the item so that it can be accessed from
    // Render() via GetEnabled() if needed.
    virtual void SetEnabled(bool enabled) wxOVERRIDE;
    bool GetEnabled() const { return m_enabled; }


    // Implementation only from now on

    // Retrieve the DC to use for drawing. This is implemented in derived
    // platform-specific classes.
    virtual wxDC *GetDC() = 0;

    // To draw background use the background colour in wxDataViewItemAttr
    virtual void RenderBackground(wxDC* dc, const wxRect& rect);

    // Prepare DC to use attributes and call Render().
    void WXCallRender(wxRect rect, wxDC *dc, int state);

    virtual bool IsCustomRenderer() const wxOVERRIDE { return true; }

protected:
    // helper for GetSize() implementations, respects attributes
    wxSize GetTextExtent(const wxString& str) const;

private:
    wxDataViewItemAttr m_attr;
    bool m_enabled;

    wxDECLARE_NO_COPY_CLASS(wxDataViewCustomRendererBase);
};

// include the declaration of all the other renderers to get the real
// wxDataViewCustomRenderer from which we need to inherit below
#ifdef wxHAS_GENERIC_DATAVIEWCTRL
    // because of the different renderer classes hierarchy in the generic
    // version, as explained above, we can include the header defining
    // wxDataViewRenderer only here and not before wxDataViewCustomRendererBase
    // declaration as for the native ports
    #include "wx/generic/dvrenderer.h"
    #include "wx/generic/dvrenderers.h"
#elif defined(__WXGTK20__)
    #include "wx/gtk/dvrenderers.h"
#elif defined(__WXMAC__)
    #include "wx/osx/dvrenderers.h"
#elif defined(__WXQT__)
    #include "wx/qt/dvrenderers.h"
#else
    #error "unknown native wxDataViewCtrl implementation"
#endif

#if wxUSE_SPINCTRL

// ----------------------------------------------------------------------------
// wxDataViewSpinRenderer
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewSpinRenderer: public wxDataViewCustomRenderer
{
public:
    wxDataViewSpinRenderer( int min, int max,
                            wxDataViewCellMode mode = wxDATAVIEW_CELL_EDITABLE,
                            int alignment = wxDVR_DEFAULT_ALIGNMENT );
    virtual bool HasEditorCtrl() const wxOVERRIDE { return true; }
    virtual wxWindow* CreateEditorCtrl( wxWindow *parent, wxRect labelRect, const wxVariant &value ) wxOVERRIDE;
    virtual bool GetValueFromEditorCtrl( wxWindow* editor, wxVariant &value ) wxOVERRIDE;
    virtual bool Render( wxRect rect, wxDC *dc, int state ) wxOVERRIDE;
    virtual wxSize GetSize() const wxOVERRIDE;
    virtual bool SetValue( const wxVariant &value ) wxOVERRIDE;
    virtual bool GetValue( wxVariant &value ) const wxOVERRIDE;
#if wxUSE_ACCESSIBILITY
    virtual wxString GetAccessibleDescription() const wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY

private:
    long    m_data;
    long    m_min,m_max;
};

#endif // wxUSE_SPINCTRL

#if defined(wxHAS_GENERIC_DATAVIEWCTRL)

// ----------------------------------------------------------------------------
// wxDataViewChoiceRenderer
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewChoiceRenderer: public wxDataViewCustomRenderer
{
public:
    wxDataViewChoiceRenderer( const wxArrayString &choices,
                            wxDataViewCellMode mode = wxDATAVIEW_CELL_EDITABLE,
                            int alignment = wxDVR_DEFAULT_ALIGNMENT );
    virtual bool HasEditorCtrl() const wxOVERRIDE { return true; }
    virtual wxWindow* CreateEditorCtrl( wxWindow *parent, wxRect labelRect, const wxVariant &value ) wxOVERRIDE;
    virtual bool GetValueFromEditorCtrl( wxWindow* editor, wxVariant &value ) wxOVERRIDE;
    virtual bool Render( wxRect rect, wxDC *dc, int state ) wxOVERRIDE;
    virtual wxSize GetSize() const wxOVERRIDE;
    virtual bool SetValue( const wxVariant &value ) wxOVERRIDE;
    virtual bool GetValue( wxVariant &value ) const wxOVERRIDE;
#if wxUSE_ACCESSIBILITY
    virtual wxString GetAccessibleDescription() const wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY

    wxString GetChoice(size_t index) const { return m_choices[index]; }
    const wxArrayString& GetChoices() const { return m_choices; }

private:
    wxArrayString m_choices;
    wxString      m_data;
};

// ----------------------------------------------------------------------------
// wxDataViewChoiceByIndexRenderer
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxDataViewChoiceByIndexRenderer: public wxDataViewChoiceRenderer
{
public:
    wxDataViewChoiceByIndexRenderer( const wxArrayString &choices,
                              wxDataViewCellMode mode = wxDATAVIEW_CELL_EDITABLE,
                              int alignment = wxDVR_DEFAULT_ALIGNMENT );

    virtual wxWindow* CreateEditorCtrl( wxWindow *parent, wxRect labelRect, const wxVariant &value ) wxOVERRIDE;
    virtual bool GetValueFromEditorCtrl( wxWindow* editor, wxVariant &value ) wxOVERRIDE;

    virtual bool SetValue( const wxVariant &value ) wxOVERRIDE;
    virtual bool GetValue( wxVariant &value ) const wxOVERRIDE;
#if wxUSE_ACCESSIBILITY
    virtual wxString GetAccessibleDescription() const wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY
};


#endif // generic version

#if defined(wxHAS_GENERIC_DATAVIEWCTRL) || defined(__WXGTK__)

// ----------------------------------------------------------------------------
// wxDataViewDateRenderer
// ----------------------------------------------------------------------------

#if wxUSE_DATEPICKCTRL
class WXDLLIMPEXP_CORE wxDataViewDateRenderer: public wxDataViewCustomRenderer
{
public:
    static wxString GetDefaultType() { return wxS("datetime"); }

    wxDataViewDateRenderer(const wxString &varianttype = GetDefaultType(),
                           wxDataViewCellMode mode = wxDATAVIEW_CELL_EDITABLE,
                           int align = wxDVR_DEFAULT_ALIGNMENT);

    virtual bool HasEditorCtrl() const wxOVERRIDE { return true; }
    virtual wxWindow *CreateEditorCtrl(wxWindow *parent, wxRect labelRect, const wxVariant &value) wxOVERRIDE;
    virtual bool GetValueFromEditorCtrl(wxWindow* editor, wxVariant &value) wxOVERRIDE;
    virtual bool SetValue(const wxVariant &value) wxOVERRIDE;
    virtual bool GetValue(wxVariant& value) const wxOVERRIDE;
#if wxUSE_ACCESSIBILITY
    virtual wxString GetAccessibleDescription() const wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY
    virtual bool Render( wxRect cell, wxDC *dc, int state ) wxOVERRIDE;
    virtual wxSize GetSize() const wxOVERRIDE;

private:
    wxDateTime    m_date;
};
#else // !wxUSE_DATEPICKCTRL
typedef wxDataViewTextRenderer wxDataViewDateRenderer;
#endif

#endif // generic or GTK+ versions

// ----------------------------------------------------------------------------
// wxDataViewCheckIconTextRenderer: 3-state checkbox + text + optional icon
// ----------------------------------------------------------------------------

#if defined(wxHAS_GENERIC_DATAVIEWCTRL) || !defined(__WXOSX__)

class WXDLLIMPEXP_CORE wxDataViewCheckIconTextRenderer
    : public wxDataViewCustomRenderer
{
public:
    static wxString GetDefaultType() { return wxS("wxDataViewCheckIconText"); }

    explicit wxDataViewCheckIconTextRenderer
             (
                  wxDataViewCellMode mode = wxDATAVIEW_CELL_ACTIVATABLE,
                  int align = wxDVR_DEFAULT_ALIGNMENT
             );

    // This renderer can always display the 3rd ("indeterminate") checkbox
    // state if the model contains cells with wxCHK_UNDETERMINED value, but it
    // doesn't allow the user to set it by default. Call this method to allow
    // this to happen.
    void Allow3rdStateForUser(bool allow = true);

    virtual bool SetValue(const wxVariant& value) wxOVERRIDE;
    virtual bool GetValue(wxVariant& value) const wxOVERRIDE;

#if wxUSE_ACCESSIBILITY
    virtual wxString GetAccessibleDescription() const wxOVERRIDE;
#endif // wxUSE_ACCESSIBILITY

    virtual wxSize GetSize() const wxOVERRIDE;
    virtual bool Render(wxRect cell, wxDC* dc, int state) wxOVERRIDE;
    virtual bool ActivateCell(const wxRect& cell,
                              wxDataViewModel *model,
                              const wxDataViewItem & item,
                              unsigned int col,
                              const wxMouseEvent *mouseEvent) wxOVERRIDE;

private:
    wxSize GetCheckSize() const;

    // Just some arbitrary constants defining margins, in pixels.
    enum
    {
        MARGIN_CHECK_ICON = 3,
        MARGIN_ICON_TEXT = 4
    };

    wxDataViewCheckIconText m_value;

    bool m_allow3rdStateForUser;

    wxDECLARE_DYNAMIC_CLASS_NO_COPY(wxDataViewCheckIconTextRenderer);
};

#endif // ! native __WXOSX__

// this class is obsolete, its functionality was merged in
// wxDataViewTextRenderer itself now, don't use it any more
#define wxDataViewTextRendererAttr wxDataViewTextRenderer

#endif // _WX_DVRENDERERS_H_

