///////////////////////////////////////////////////////////////////////////////
// Name:        wx/lzmastream.h
// Purpose:     Filters streams using LZMA(2) compression
// Author:      Vadim Zeitlin
// Created:     2018-03-29
// Copyright:   (c) 2018 Vadim Zeitlin <vadim@wxwidgets.org>
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_LZMASTREAM_H_
#define _WX_LZMASTREAM_H_

#include "wx/defs.h"

#if wxUSE_LIBLZMA && wxUSE_STREAMS

#include "wx/stream.h"
#include "wx/versioninfo.h"

namespace wxPrivate
{

// Private wrapper for lzma_stream struct.
struct wxLZMAStream;

// Common part of input and output LZMA streams: this is just an implementation
// detail and is not part of the public API.
class WXDLLIMPEXP_BASE wxLZMAData
{
protected:
    wxLZMAData();
    ~wxLZMAData();

    wxLZMAStream* m_stream;
    wxUint8* m_streamBuf;
    wxFileOffset m_pos;

    wxDECLARE_NO_COPY_CLASS(wxLZMAData);
};

} // namespace wxPrivate

// ----------------------------------------------------------------------------
// Filter for decompressing data compressed using LZMA
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxLZMAInputStream : public wxFilterInputStream,
                                           private wxPrivate::wxLZMAData
{
public:
    explicit wxLZMAInputStream(wxInputStream& stream)
        : wxFilterInputStream(stream)
    {
        Init();
    }

    explicit wxLZMAInputStream(wxInputStream* stream)
        : wxFilterInputStream(stream)
    {
        Init();
    }

    char Peek() wxOVERRIDE { return wxInputStream::Peek(); }
    wxFileOffset GetLength() const wxOVERRIDE { return wxInputStream::GetLength(); }

protected:
    size_t OnSysRead(void *buffer, size_t size) wxOVERRIDE;
    wxFileOffset OnSysTell() const wxOVERRIDE { return m_pos; }

private:
    void Init();
};

// ----------------------------------------------------------------------------
// Filter for compressing data using LZMA(2) algorithm
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxLZMAOutputStream : public wxFilterOutputStream,
                                            private wxPrivate::wxLZMAData
{
public:
    explicit wxLZMAOutputStream(wxOutputStream& stream, int level = -1)
        : wxFilterOutputStream(stream)
    {
        Init(level);
    }

    explicit wxLZMAOutputStream(wxOutputStream* stream, int level = -1)
        : wxFilterOutputStream(stream)
    {
        Init(level);
    }

    virtual ~wxLZMAOutputStream() { Close(); }

    void Sync() wxOVERRIDE { DoFlush(false); }
    bool Close() wxOVERRIDE;
    wxFileOffset GetLength() const wxOVERRIDE { return m_pos; }

protected:
    size_t OnSysWrite(const void *buffer, size_t size) wxOVERRIDE;
    wxFileOffset OnSysTell() const wxOVERRIDE { return m_pos; }

private:
    void Init(int level);

    // Write the contents of the internal buffer to the output stream.
    bool UpdateOutput();

    // Write out the current buffer if necessary, i.e. if no space remains in
    // it, and reinitialize m_stream to point to it. Returns false on success
    // or false on error, in which case m_lasterror is updated.
    bool UpdateOutputIfNecessary();

    // Run LZMA_FINISH (if argument is true) or LZMA_FULL_FLUSH, return true on
    // success or false on error.
    bool DoFlush(bool finish);
};

// ----------------------------------------------------------------------------
// Support for creating LZMA streams from extension/MIME type
// ----------------------------------------------------------------------------

class WXDLLIMPEXP_BASE wxLZMAClassFactory: public wxFilterClassFactory
{
public:
    wxLZMAClassFactory();

    wxFilterInputStream *NewStream(wxInputStream& stream) const wxOVERRIDE
        { return new wxLZMAInputStream(stream); }
    wxFilterOutputStream *NewStream(wxOutputStream& stream) const wxOVERRIDE
        { return new wxLZMAOutputStream(stream, -1); }
    wxFilterInputStream *NewStream(wxInputStream *stream) const wxOVERRIDE
        { return new wxLZMAInputStream(stream); }
    wxFilterOutputStream *NewStream(wxOutputStream *stream) const wxOVERRIDE
        { return new wxLZMAOutputStream(stream, -1); }

    const wxChar * const *GetProtocols(wxStreamProtocolType type
                                       = wxSTREAM_PROTOCOL) const wxOVERRIDE;

private:
    wxDECLARE_DYNAMIC_CLASS(wxLZMAClassFactory);
};

WXDLLIMPEXP_BASE wxVersionInfo wxGetLibLZMAVersionInfo();

#endif // wxUSE_LIBLZMA && wxUSE_STREAMS

#endif // _WX_LZMASTREAM_H_
