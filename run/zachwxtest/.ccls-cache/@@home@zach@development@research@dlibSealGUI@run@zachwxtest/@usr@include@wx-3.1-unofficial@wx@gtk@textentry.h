///////////////////////////////////////////////////////////////////////////////
// Name:        wx/gtk/textentry.h
// Purpose:     wxGTK-specific wxTextEntry implementation
// Author:      Vadim Zeitlin
// Created:     2007-09-24
// Copyright:   (c) 2007 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_GTK_TEXTENTRY_H_
#define _WX_GTK_TEXTENTRY_H_

typedef struct _GdkEventKey GdkEventKey;
typedef struct _GtkEditable GtkEditable;
typedef struct _GtkEntry GtkEntry;

class wxTextAutoCompleteData; // private class used only by wxTextEntry itself
class wxTextCoalesceData;     // another private class

// ----------------------------------------------------------------------------
// wxTextEntry: roughly corresponds to GtkEditable
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_CORE wxTextEntry : public wxTextEntryBase
{
public:
    wxTextEntry();
    virtual ~wxTextEntry();

    // implement wxTextEntryBase pure virtual methods
    virtual void WriteText(const wxString& text) wxOVERRIDE;
    virtual void Remove(long from, long to) wxOVERRIDE;

    virtual void Copy() wxOVERRIDE;
    virtual void Cut() wxOVERRIDE;
    virtual void Paste() wxOVERRIDE;

    virtual void Undo() wxOVERRIDE;
    virtual void Redo() wxOVERRIDE;
    virtual bool CanUndo() const wxOVERRIDE;
    virtual bool CanRedo() const wxOVERRIDE;

    virtual void SetInsertionPoint(long pos) wxOVERRIDE;
    virtual long GetInsertionPoint() const wxOVERRIDE;
    virtual long GetLastPosition() const wxOVERRIDE;

    virtual void SetSelection(long from, long to) wxOVERRIDE;
    virtual void GetSelection(long *from, long *to) const wxOVERRIDE;

    virtual bool IsEditable() const wxOVERRIDE;
    virtual void SetEditable(bool editable) wxOVERRIDE;

    virtual void SetMaxLength(unsigned long len) wxOVERRIDE;
    virtual void ForceUpper() wxOVERRIDE;

#ifdef __WXGTK3__
    virtual bool SetHint(const wxString& hint) wxOVERRIDE;
    virtual wxString GetHint() const wxOVERRIDE;
#endif

    // implementation only from now on
    void SendMaxLenEvent();
    bool GTKEntryOnInsertText(const char* text);
    bool GTKIsUpperCase() const { return m_isUpperCase; }

    // Called from "changed" signal handler (or, possibly, slightly later, when
    // coalescing several "changed" signals into a single event) for GtkEntry.
    //
    // By default just generates a wxEVT_TEXT, but overridden to do more things
    // in wxTextCtrl.
    virtual void GTKOnTextChanged() { SendTextUpdatedEvent(); }

    // Helper functions only used internally.
    wxTextCoalesceData* GTKGetCoalesceData() const { return m_coalesceData; }

protected:
    // This method must be called from the derived class Create() to connect
    // the handlers for the clipboard (cut/copy/paste) events.
    void GTKConnectClipboardSignals(GtkWidget* entry);

    // And this one to connect "insert-text" signal.
    void GTKConnectInsertTextSignal(GtkEntry* entry);

    // Finally this one connects to the "changed" signal on the object returned
    // by GetTextObject().
    void GTKConnectChangedSignal();


    virtual void DoSetValue(const wxString& value, int flags) wxOVERRIDE;
    virtual wxString DoGetValue() const wxOVERRIDE;

    // margins functions
    virtual bool DoSetMargins(const wxPoint& pt) wxOVERRIDE;
    virtual wxPoint DoGetMargins() const wxOVERRIDE;

    virtual bool DoAutoCompleteStrings(const wxArrayString& choices) wxOVERRIDE;
    virtual bool DoAutoCompleteCustom(wxTextCompleter *completer) wxOVERRIDE;

    // Call this from the overridden wxWindow::GTKIMFilterKeypress() to use
    // GtkEntry IM context.
    int GTKEntryIMFilterKeypress(GdkEventKey* event) const;

    // If GTKEntryIMFilterKeypress() is not called (as multiline wxTextCtrl
    // uses its own IM), call this method instead to still notify wxTextEntry
    // about the key press events in the given widget.
    void GTKEntryOnKeypress(GtkWidget* widget) const;


    static int GTKGetEntryTextLength(GtkEntry* entry);

    // Block/unblock the corresponding GTK signal.
    //
    // Note that we make it protected in wxGTK as it is called from wxComboBox
    // currently.
    virtual void EnableTextChangedEvents(bool enable) wxOVERRIDE;

    // Helper for wxTE_PROCESS_ENTER handling: activates the default button in
    // the dialog containing this control if any.
    bool ClickDefaultButtonIfPossible();

private:
    // implement this to return the associated GtkEntry or another widget
    // implementing GtkEditable
    virtual GtkEditable *GetEditable() const = 0;

    // implement this to return the associated GtkEntry
    virtual GtkEntry *GetEntry() const = 0;

    // This one exists in order to be overridden by wxTextCtrl which uses
    // either GtkEditable or GtkTextBuffer depending on whether it is single-
    // or multi-line.
    virtual void *GetTextObject() const { return GetEntry(); }


    // Various auto-completion-related stuff, only used if any of AutoComplete()
    // methods are called.
    wxTextAutoCompleteData *m_autoCompleteData;

    // It needs to call our GetEntry() method.
    friend class wxTextAutoCompleteData;

    // Data used for coalescing "changed" events resulting from a single user
    // action.
    mutable wxTextCoalesceData* m_coalesceData;

    bool m_isUpperCase;
};

// We don't need the generic version.
#define wxHAS_NATIVE_TEXT_FORCEUPPER

#endif // _WX_GTK_TEXTENTRY_H_

