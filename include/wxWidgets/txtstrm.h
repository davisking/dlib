/////////////////////////////////////////////////////////////////////////////
// Name:        wx/txtstrm.h
// Purpose:     Text stream classes
// Author:      Guilhem Lavaux
// Modified by:
// Created:     28/06/1998
// Copyright:   (c) Guilhem Lavaux
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_TXTSTREAM_H_
#define _WX_TXTSTREAM_H_

#include "wx/stream.h"
#include "wx/convauto.h"

#if wxUSE_STREAMS

class WXDLLIMPEXP_FWD_BASE wxTextInputStream;
class WXDLLIMPEXP_FWD_BASE wxTextOutputStream;

typedef wxTextInputStream& (*__wxTextInputManip)(wxTextInputStream&);
typedef wxTextOutputStream& (*__wxTextOutputManip)(wxTextOutputStream&);

WXDLLIMPEXP_BASE wxTextOutputStream &endl( wxTextOutputStream &stream );


// Obsolete constant defined only for compatibility, not used.
#define wxEOT wxT('\4')

// If you're scanning through a file using wxTextInputStream, you should check for EOF _before_
// reading the next item (word / number), because otherwise the last item may get lost.
// You should however be prepared to receive an empty item (empty string / zero number) at the
// end of file, especially on Windows systems. This is unavoidable because most (but not all) files end
// with whitespace (i.e. usually a newline).
class WXDLLIMPEXP_BASE wxTextInputStream
{
public:
#if wxUSE_UNICODE
    wxTextInputStream(wxInputStream& s,
                      const wxString &sep=wxT(" \t"),
                      const wxMBConv& conv = wxConvAuto());
#else
    wxTextInputStream(wxInputStream& s, const wxString &sep=wxT(" \t"));
#endif
    ~wxTextInputStream();

    const wxInputStream& GetInputStream() const { return m_input; }

    // base may be between 2 and 36, inclusive, or the special 0 (= C format)
    wxUint64 Read64(int base = 10);
    wxUint32 Read32(int base = 10);
    wxUint16 Read16(int base = 10);
    wxUint8  Read8(int base = 10);
    wxInt64  Read64S(int base = 10);
    wxInt32  Read32S(int base = 10);
    wxInt16  Read16S(int base = 10);
    wxInt8   Read8S(int base = 10);
    double   ReadDouble();
    wxString ReadLine();
    wxString ReadWord();
    wxChar   GetChar();

    wxString GetStringSeparators() const { return m_separators; }
    void SetStringSeparators(const wxString &c) { m_separators = c; }

    // Operators
    wxTextInputStream& operator>>(wxString& word);
    wxTextInputStream& operator>>(char& c);
#if wxUSE_UNICODE && wxWCHAR_T_IS_REAL_TYPE
    wxTextInputStream& operator>>(wchar_t& wc);
#endif // wxUSE_UNICODE
    wxTextInputStream& operator>>(wxInt16& i);
    wxTextInputStream& operator>>(wxInt32& i);
    wxTextInputStream& operator>>(wxInt64& i);
    wxTextInputStream& operator>>(wxUint16& i);
    wxTextInputStream& operator>>(wxUint32& i);
    wxTextInputStream& operator>>(wxUint64& i);
    wxTextInputStream& operator>>(double& i);
    wxTextInputStream& operator>>(float& f);

    wxTextInputStream& operator>>( __wxTextInputManip func) { return func(*this); }

protected:
    wxInputStream &m_input;
    wxString m_separators;

    // Data possibly (see m_validXXX) read from the stream but not decoded yet.
    // This is necessary because GetChar() may only return a single character
    // but we may get more than one character when decoding raw input bytes.
    char m_lastBytes[10];

    // The bytes [0, m_validEnd) of m_lastBytes contain the bytes read by the
    // last GetChar() call (this interval may be empty if GetChar() hasn't been
    // called yet). The bytes [0, m_validBegin) have been already decoded and
    // returned to caller or stored in m_lastWChar in the particularly
    // egregious case of decoding a non-BMP character when using UTF-16 for
    // wchar_t. Finally, the bytes [m_validBegin, m_validEnd) remain to be
    // decoded and returned during the next call (again, this interval can, and
    // usually will, be empty too if m_validBegin == m_validEnd).
    size_t m_validBegin,
           m_validEnd;

#if wxUSE_UNICODE
    wxMBConv *m_conv;

    // The second half of a surrogate character when using UTF-16 for wchar_t:
    // we can't return it immediately from GetChar() when we read a Unicode
    // code point outside of the BMP, but we can't keep it in m_lastBytes
    // neither because it can't separately decoded, so we have a separate 1
    // wchar_t buffer just for this case.
#if SIZEOF_WCHAR_T == 2
    wchar_t m_lastWChar;
#endif // SIZEOF_WCHAR_T == 2
#endif // wxUSE_UNICODE

    bool   EatEOL(const wxChar &c);
    void   UngetLast(); // should be used instead of wxInputStream::Ungetch() because of Unicode issues
    wxChar NextNonSeparators();

    wxDECLARE_NO_COPY_CLASS(wxTextInputStream);
};

enum wxEOL
{
  wxEOL_NATIVE,
  wxEOL_UNIX,
  wxEOL_MAC,
  wxEOL_DOS
};

class WXDLLIMPEXP_BASE wxTextOutputStream
{
public:
#if wxUSE_UNICODE
    wxTextOutputStream(wxOutputStream& s,
                       wxEOL mode = wxEOL_NATIVE,
                       const wxMBConv& conv = wxConvAuto());
#else
    wxTextOutputStream(wxOutputStream& s, wxEOL mode = wxEOL_NATIVE);
#endif
    virtual ~wxTextOutputStream();

    const wxOutputStream& GetOutputStream() const { return m_output; }

    void SetMode( wxEOL mode = wxEOL_NATIVE );
    wxEOL GetMode() { return m_mode; }

    template<typename T>
    void Write(const T& i)
    {
        wxString str;
        str << i;

        WriteString(str);
    }

    void Write64(wxUint64 i);
    void Write32(wxUint32 i);
    void Write16(wxUint16 i);
    void Write8(wxUint8 i);
    virtual void WriteDouble(double d);
    virtual void WriteString(const wxString& string);

    wxTextOutputStream& PutChar(wxChar c);

    void Flush();

    wxTextOutputStream& operator<<(const wxString& string);
    wxTextOutputStream& operator<<(char c);
#if wxUSE_UNICODE && wxWCHAR_T_IS_REAL_TYPE
    wxTextOutputStream& operator<<(wchar_t wc);
#endif // wxUSE_UNICODE
    wxTextOutputStream& operator<<(wxInt16 c);
    wxTextOutputStream& operator<<(wxInt32 c);
    wxTextOutputStream& operator<<(wxInt64 c);
    wxTextOutputStream& operator<<(wxUint16 c);
    wxTextOutputStream& operator<<(wxUint32 c);
    wxTextOutputStream& operator<<(wxUint64 c);
    wxTextOutputStream& operator<<(double f);
    wxTextOutputStream& operator<<(float f);

    wxTextOutputStream& operator<<( __wxTextOutputManip func) { return func(*this); }

protected:
    wxOutputStream &m_output;
    wxEOL           m_mode;

#if wxUSE_UNICODE
    wxMBConv *m_conv;

#if SIZEOF_WCHAR_T == 2
    // The first half of a surrogate character if one was passed to PutChar()
    // and couldn't be output when it was called the last time.
    wchar_t m_lastWChar;
#endif // SIZEOF_WCHAR_T == 2
#endif // wxUSE_UNICODE

    wxDECLARE_NO_COPY_CLASS(wxTextOutputStream);
};

#endif
  // wxUSE_STREAMS

#endif
    // _WX_DATSTREAM_H_
