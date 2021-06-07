/////////////////////////////////////////////////////////////////////////////
// Name:        wx/zipstrm.h
// Purpose:     Streams for Zip files
// Author:      Mike Wetherell
// Copyright:   (c) Mike Wetherell
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_WXZIPSTREAM_H__
#define _WX_WXZIPSTREAM_H__

#include "wx/defs.h"

#if wxUSE_ZIPSTREAM

#include "wx/archive.h"
#include "wx/filename.h"

// some methods from wxZipInputStream and wxZipOutputStream stream do not get
// exported/imported when compiled with Mingw versions before 3.4.2. So they
// are imported/exported individually as a workaround
#if (defined(__GNUWIN32__) || defined(__MINGW32__)) \
    && (!defined __GNUC__ \
       || !defined __GNUC_MINOR__ \
       || !defined __GNUC_PATCHLEVEL__ \
       || __GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__ < 30402)
#define WXZIPFIX WXDLLIMPEXP_BASE
#else
#define WXZIPFIX
#endif

/////////////////////////////////////////////////////////////////////////////
// constants

// Compression Method, only 0 (store) and 8 (deflate) are supported here
//
enum wxZipMethod
{
    wxZIP_METHOD_STORE,
    wxZIP_METHOD_SHRINK,
    wxZIP_METHOD_REDUCE1,
    wxZIP_METHOD_REDUCE2,
    wxZIP_METHOD_REDUCE3,
    wxZIP_METHOD_REDUCE4,
    wxZIP_METHOD_IMPLODE,
    wxZIP_METHOD_TOKENIZE,
    wxZIP_METHOD_DEFLATE,
    wxZIP_METHOD_DEFLATE64,
    wxZIP_METHOD_BZIP2 = 12,
    wxZIP_METHOD_DEFAULT = 0xffff
};

// Originating File-System.
//
// These are Pkware's values. Note that Info-zip disagree on some of them,
// most notably NTFS.
//
enum wxZipSystem
{
    wxZIP_SYSTEM_MSDOS,
    wxZIP_SYSTEM_AMIGA,
    wxZIP_SYSTEM_OPENVMS,
    wxZIP_SYSTEM_UNIX,
    wxZIP_SYSTEM_VM_CMS,
    wxZIP_SYSTEM_ATARI_ST,
    wxZIP_SYSTEM_OS2_HPFS,
    wxZIP_SYSTEM_MACINTOSH,
    wxZIP_SYSTEM_Z_SYSTEM,
    wxZIP_SYSTEM_CPM,
    wxZIP_SYSTEM_WINDOWS_NTFS,
    wxZIP_SYSTEM_MVS,
    wxZIP_SYSTEM_VSE,
    wxZIP_SYSTEM_ACORN_RISC,
    wxZIP_SYSTEM_VFAT,
    wxZIP_SYSTEM_ALTERNATE_MVS,
    wxZIP_SYSTEM_BEOS,
    wxZIP_SYSTEM_TANDEM,
    wxZIP_SYSTEM_OS_400
};

// Dos/Win file attributes
//
enum wxZipAttributes
{
    wxZIP_A_RDONLY = 0x01,
    wxZIP_A_HIDDEN = 0x02,
    wxZIP_A_SYSTEM = 0x04,
    wxZIP_A_SUBDIR = 0x10,
    wxZIP_A_ARCH   = 0x20,

    wxZIP_A_MASK   = 0x37
};

// Values for the flags field in the zip headers
//
enum wxZipFlags
{
    wxZIP_ENCRYPTED         = 0x0001,
    wxZIP_DEFLATE_NORMAL    = 0x0000,   // normal compression
    wxZIP_DEFLATE_EXTRA     = 0x0002,   // extra compression
    wxZIP_DEFLATE_FAST      = 0x0004,   // fast compression
    wxZIP_DEFLATE_SUPERFAST = 0x0006,   // superfast compression
    wxZIP_DEFLATE_MASK      = 0x0006,
    wxZIP_SUMS_FOLLOW       = 0x0008,   // crc and sizes come after the data
    wxZIP_ENHANCED          = 0x0010,
    wxZIP_PATCH             = 0x0020,
    wxZIP_STRONG_ENC        = 0x0040,
    wxZIP_LANG_ENC_UTF8     = 0x0800,   // filename and comment are UTF8
    wxZIP_UNUSED            = 0x0F80,
    wxZIP_RESERVED          = 0xF000
};

enum wxZipArchiveFormat
{
    /// Default zip format
    wxZIP_FORMAT_DEFAULT,
    /// ZIP64 format
    wxZIP_FORMAT_ZIP64
};

// Forward decls
//
class WXDLLIMPEXP_FWD_BASE wxZipEntry;
class WXDLLIMPEXP_FWD_BASE wxZipInputStream;


/////////////////////////////////////////////////////////////////////////////
// wxZipNotifier

class WXDLLIMPEXP_BASE wxZipNotifier
{
public:
    virtual ~wxZipNotifier() { }

    virtual void OnEntryUpdated(wxZipEntry& entry) = 0;
};


/////////////////////////////////////////////////////////////////////////////
// Zip Entry - holds the meta data for a file in the zip

class wxDataOutputStream;

class WXDLLIMPEXP_BASE wxZipEntry : public wxArchiveEntry
{
public:
    wxZipEntry(const wxString& name = wxEmptyString,
               const wxDateTime& dt = wxDateTime::Now(),
               wxFileOffset size = wxInvalidOffset);
    virtual ~wxZipEntry();

    wxZipEntry(const wxZipEntry& entry);
    wxZipEntry& operator=(const wxZipEntry& entry);

    // Get accessors
    wxDateTime   GetDateTime() const wxOVERRIDE            { return m_DateTime; }
    wxFileOffset GetSize() const wxOVERRIDE                { return m_Size; }
    wxFileOffset GetOffset() const wxOVERRIDE              { return m_Offset; }
    wxString     GetInternalName() const wxOVERRIDE        { return m_Name; }
    int          GetMethod() const              { return m_Method; }
    int          GetFlags() const               { return m_Flags; }
    wxUint32     GetCrc() const                 { return m_Crc; }
    wxFileOffset GetCompressedSize() const      { return m_CompressedSize; }
    int          GetSystemMadeBy() const        { return m_SystemMadeBy; }
    wxString     GetComment() const             { return m_Comment; }
    wxUint32     GetExternalAttributes() const  { return m_ExternalAttributes; }
    wxPathFormat GetInternalFormat() const wxOVERRIDE      { return wxPATH_UNIX; }
    int          GetMode() const;
    const char  *GetLocalExtra() const;
    size_t       GetLocalExtraLen() const;
    const char  *GetExtra() const;
    size_t       GetExtraLen() const;
    wxString     GetName(wxPathFormat format = wxPATH_NATIVE) const wxOVERRIDE;

    // is accessors
    inline bool IsDir() const wxOVERRIDE;
    inline bool IsText() const;
    inline bool IsReadOnly() const wxOVERRIDE;
    inline bool IsMadeByUnix() const;

    // set accessors
    void SetDateTime(const wxDateTime& dt) wxOVERRIDE      { m_DateTime = dt; }
    void SetSize(wxFileOffset size) wxOVERRIDE             { m_Size = size; }
    void SetMethod(int method)                  { m_Method = (wxUint16)method; }
    void SetComment(const wxString& comment)    { m_Comment = comment; }
    void SetExternalAttributes(wxUint32 attr )  { m_ExternalAttributes = attr; }
    void SetSystemMadeBy(int system);
    void SetMode(int mode);
    void SetExtra(const char *extra, size_t len);
    void SetLocalExtra(const char *extra, size_t len);

    inline void SetName(const wxString& name,
                        wxPathFormat format = wxPATH_NATIVE) wxOVERRIDE;

    static wxString GetInternalName(const wxString& name,
                                    wxPathFormat format = wxPATH_NATIVE,
                                    bool *pIsDir = NULL);

    // set is accessors
    void SetIsDir(bool isDir = true) wxOVERRIDE;
    inline void SetIsReadOnly(bool isReadOnly = true) wxOVERRIDE;
    inline void SetIsText(bool isText = true);

    wxZipEntry *Clone() const                   { return ZipClone(); }

    void SetNotifier(wxZipNotifier& notifier);
    void UnsetNotifier() wxOVERRIDE;

protected:
    // Internal attributes
    enum { TEXT_ATTR = 1 };

    // protected Get accessors
    int GetVersionNeeded() const                { return m_VersionNeeded; }
    wxFileOffset GetKey() const                 { return m_Key; }
    int GetVersionMadeBy() const                { return m_VersionMadeBy; }
    int GetDiskStart() const                    { return m_DiskStart; }
    int GetInternalAttributes() const           { return m_InternalAttributes; }

    void SetVersionNeeded(int version)          { m_VersionNeeded = (wxUint16)version; }
    void SetOffset(wxFileOffset offset) wxOVERRIDE         { m_Offset = offset; }
    void SetFlags(int flags)                    { m_Flags = (wxUint16)flags; }
    void SetVersionMadeBy(int version)          { m_VersionMadeBy = (wxUint8)version; }
    void SetCrc(wxUint32 crc)                   { m_Crc = crc; }
    void SetCompressedSize(wxFileOffset size)   { m_CompressedSize = size; }
    void SetKey(wxFileOffset offset)            { m_Key = offset; }
    void SetDiskStart(int start)                { m_DiskStart = (wxUint16)start; }
    void SetInternalAttributes(int attr)        { m_InternalAttributes = (wxUint16)attr; }

    virtual wxZipEntry *ZipClone() const        { return new wxZipEntry(*this); }

    void Notify();

private:
    wxArchiveEntry* DoClone() const wxOVERRIDE             { return ZipClone(); }

    size_t ReadLocal(wxInputStream& stream, wxMBConv& conv);
    size_t WriteLocal(wxOutputStream& stream, wxMBConv& conv, wxZipArchiveFormat zipFormat);

    size_t ReadCentral(wxInputStream& stream, wxMBConv& conv);
    size_t WriteCentral(wxOutputStream& stream, wxMBConv& conv) const;

    size_t ReadDescriptor(wxInputStream& stream);
    size_t WriteDescriptor(wxOutputStream& stream, wxUint32 crc,
                           wxFileOffset compressedSize, wxFileOffset size);

    void WriteLocalFileSizes(wxDataOutputStream& ds) const;
    void WriteLocalZip64ExtraInfo(wxOutputStream& stream) const;

    bool LoadExtraInfo(const char* extraData, wxUint16 extraLen, bool localInfo);

    wxUint16 GetInternalFlags(bool checkForUTF8) const;

    wxUint8      m_SystemMadeBy;       // one of enum wxZipSystem
    wxUint8      m_VersionMadeBy;      // major * 10 + minor

    wxUint16     m_VersionNeeded;      // ver needed to extract (20 i.e. v2.0)
    wxUint16     m_Flags;
    wxUint16     m_Method;             // compression method (one of wxZipMethod)
    wxDateTime   m_DateTime;
    wxUint32     m_Crc;
    wxFileOffset m_CompressedSize;
    wxFileOffset m_Size;
    wxString     m_Name;               // in internal format
    wxFileOffset m_Key;                // the original offset for copied entries
    wxFileOffset m_Offset;             // file offset of the entry

    wxString     m_Comment;
    wxUint16     m_DiskStart;          // for multidisk archives, not unsupported
    wxUint16     m_InternalAttributes; // bit 0 set for text files
    wxUint32     m_ExternalAttributes; // system specific depends on SystemMadeBy
    wxUint16     m_z64infoOffset;      // Offset of ZIP64 local extra data for file sizes

    class wxZipMemory *m_Extra;
    class wxZipMemory *m_LocalExtra;

    wxZipNotifier *m_zipnotifier;
    class wxZipWeakLinks *m_backlink;

    friend class wxZipInputStream;
    friend class wxZipOutputStream;

    wxDECLARE_DYNAMIC_CLASS(wxZipEntry);
};


/////////////////////////////////////////////////////////////////////////////
// wxZipOutputStream

WX_DECLARE_LIST_WITH_DECL(wxZipEntry, wxZipEntryList_, class WXDLLIMPEXP_BASE);

class WXDLLIMPEXP_BASE wxZipOutputStream : public wxArchiveOutputStream
{
public:
    wxZipOutputStream(wxOutputStream& stream,
                      int level = -1,
                      wxMBConv& conv = wxConvUTF8);
    wxZipOutputStream(wxOutputStream *stream,
                      int level = -1,
                      wxMBConv& conv = wxConvUTF8);
    virtual WXZIPFIX ~wxZipOutputStream();

    bool PutNextEntry(wxZipEntry *entry)        { return DoCreate(entry); }

    bool WXZIPFIX PutNextEntry(const wxString& name,
                               const wxDateTime& dt = wxDateTime::Now(),
                               wxFileOffset size = wxInvalidOffset) wxOVERRIDE;

    bool WXZIPFIX PutNextDirEntry(const wxString& name,
                                  const wxDateTime& dt = wxDateTime::Now()) wxOVERRIDE;

    bool WXZIPFIX CopyEntry(wxZipEntry *entry, wxZipInputStream& inputStream);
    bool WXZIPFIX CopyArchiveMetaData(wxZipInputStream& inputStream);

    void WXZIPFIX Sync() wxOVERRIDE;
    bool WXZIPFIX CloseEntry() wxOVERRIDE;
    bool WXZIPFIX Close() wxOVERRIDE;

    void SetComment(const wxString& comment)    { m_Comment = comment; }

    int  GetLevel() const                       { return m_level; }
    void WXZIPFIX SetLevel(int level);

    void SetFormat(wxZipArchiveFormat format)   { m_format = format; }
    wxZipArchiveFormat GetFormat() const        { return m_format; }

protected:
    virtual size_t WXZIPFIX OnSysWrite(const void *buffer, size_t size) wxOVERRIDE;
    virtual wxFileOffset OnSysTell() const wxOVERRIDE      { return m_entrySize; }

    // this protected interface isn't yet finalised
    struct Buffer { const char *m_data; size_t m_size; };
    virtual wxOutputStream* WXZIPFIX OpenCompressor(wxOutputStream& stream,
                                                    wxZipEntry& entry,
                                                    const Buffer bufs[]);
    virtual bool WXZIPFIX CloseCompressor(wxOutputStream *comp);

    bool IsParentSeekable() const
        { return m_offsetAdjustment != wxInvalidOffset; }

private:
    void Init(int level);

    bool WXZIPFIX PutNextEntry(wxArchiveEntry *entry) wxOVERRIDE;
    bool WXZIPFIX CopyEntry(wxArchiveEntry *entry, wxArchiveInputStream& stream) wxOVERRIDE;
    bool WXZIPFIX CopyArchiveMetaData(wxArchiveInputStream& stream) wxOVERRIDE;

    bool IsOpened() const { return m_comp || m_pending; }

    bool DoCreate(wxZipEntry *entry, bool raw = false);
    void CreatePendingEntry(const void *buffer, size_t size);
    void CreatePendingEntry();

    class wxStoredOutputStream *m_store;
    class wxZlibOutputStream2 *m_deflate;
    class wxZipStreamLink *m_backlink;
    wxZipEntryList_ m_entries;
    char *m_initialData;
    size_t m_initialSize;
    wxZipEntry *m_pending;
    bool m_raw;
    wxFileOffset m_headerOffset;
    size_t m_headerSize;
    wxFileOffset m_entrySize;
    wxUint32 m_crcAccumulator;
    wxOutputStream *m_comp;
    int m_level;
    wxFileOffset m_offsetAdjustment;
    wxString m_Comment;
    bool m_endrecWritten;
    wxZipArchiveFormat m_format;

    wxDECLARE_NO_COPY_CLASS(wxZipOutputStream);
};


/////////////////////////////////////////////////////////////////////////////
// wxZipInputStream

class WXDLLIMPEXP_BASE wxZipInputStream : public wxArchiveInputStream
{
public:
    typedef wxZipEntry entry_type;

    wxZipInputStream(wxInputStream& stream, wxMBConv& conv = wxConvLocal);
    wxZipInputStream(wxInputStream *stream, wxMBConv& conv = wxConvLocal);

    virtual WXZIPFIX ~wxZipInputStream();

    bool OpenEntry(wxZipEntry& entry)   { return DoOpen(&entry); }
    bool WXZIPFIX CloseEntry() wxOVERRIDE;

    wxZipEntry *GetNextEntry();

    wxString WXZIPFIX GetComment();
    int WXZIPFIX GetTotalEntries();

    virtual wxFileOffset GetLength() const wxOVERRIDE { return m_entry.GetSize(); }

protected:
    size_t WXZIPFIX OnSysRead(void *buffer, size_t size) wxOVERRIDE;
    wxFileOffset OnSysTell() const wxOVERRIDE { return m_decomp ? m_decomp->TellI() : 0; }

    // this protected interface isn't yet finalised
    virtual wxInputStream* WXZIPFIX OpenDecompressor(wxInputStream& stream);
    virtual bool WXZIPFIX CloseDecompressor(wxInputStream *decomp);

private:
    void Init();
    void Init(const wxString& file);

    wxArchiveEntry *DoGetNextEntry() wxOVERRIDE    { return GetNextEntry(); }

    bool WXZIPFIX OpenEntry(wxArchiveEntry& entry) wxOVERRIDE;

    wxStreamError ReadLocal(bool readEndRec = false);
    wxStreamError ReadCentral();

    wxUint32 ReadSignature();
    bool FindEndRecord();
    bool LoadEndRecord();

    bool AtHeader() const       { return m_headerSize == 0; }
    bool AfterHeader() const    { return m_headerSize > 0 && !m_decomp; }
    bool IsOpened() const       { return m_decomp != NULL; }

    wxZipStreamLink *MakeLink(wxZipOutputStream *out);

    bool DoOpen(wxZipEntry *entry = NULL, bool raw = false);
    bool OpenDecompressor(bool raw = false);

    class wxStoredInputStream *m_store;
    class wxZlibInputStream2 *m_inflate;
    class wxRawInputStream *m_rawin;
    wxZipEntry m_entry;
    bool m_raw;
    size_t m_headerSize;
    wxUint32 m_crcAccumulator;
    wxInputStream *m_decomp;
    bool m_parentSeekable;
    class wxZipWeakLinks *m_weaklinks;
    class wxZipStreamLink *m_streamlink;
    wxFileOffset m_offsetAdjustment;
    wxFileOffset m_position;
    wxUint32 m_signature;
    size_t m_TotalEntries;
    wxString m_Comment;

    friend bool wxZipOutputStream::CopyEntry(
                    wxZipEntry *entry, wxZipInputStream& inputStream);
    friend bool wxZipOutputStream::CopyArchiveMetaData(
                    wxZipInputStream& inputStream);

    wxDECLARE_NO_COPY_CLASS(wxZipInputStream);
};


/////////////////////////////////////////////////////////////////////////////
// Iterators

#if wxUSE_STL || defined WX_TEST_ARCHIVE_ITERATOR
typedef wxArchiveIterator<wxZipInputStream> wxZipIter;
typedef wxArchiveIterator<wxZipInputStream,
         std::pair<wxString, wxZipEntry*> > wxZipPairIter;
#endif


/////////////////////////////////////////////////////////////////////////////
// wxZipClassFactory

class WXDLLIMPEXP_BASE wxZipClassFactory : public wxArchiveClassFactory
{
public:
    typedef wxZipEntry        entry_type;
    typedef wxZipInputStream  instream_type;
    typedef wxZipOutputStream outstream_type;
    typedef wxZipNotifier     notifier_type;
#if wxUSE_STL || defined WX_TEST_ARCHIVE_ITERATOR
    typedef wxZipIter         iter_type;
    typedef wxZipPairIter     pairiter_type;
#endif

    wxZipClassFactory();

    wxZipEntry *NewEntry() const
        { return new wxZipEntry; }
    wxZipInputStream *NewStream(wxInputStream& stream) const
        { return new wxZipInputStream(stream, GetConv()); }
    wxZipOutputStream *NewStream(wxOutputStream& stream) const
        { return new wxZipOutputStream(stream, -1, GetConv()); }
    wxZipInputStream *NewStream(wxInputStream *stream) const
        { return new wxZipInputStream(stream, GetConv()); }
    wxZipOutputStream *NewStream(wxOutputStream *stream) const
        { return new wxZipOutputStream(stream, -1, GetConv()); }

    wxString GetInternalName(const wxString& name,
                             wxPathFormat format = wxPATH_NATIVE) const wxOVERRIDE
        { return wxZipEntry::GetInternalName(name, format); }

    const wxChar * const *GetProtocols(wxStreamProtocolType type
                                       = wxSTREAM_PROTOCOL) const wxOVERRIDE;

protected:
    wxArchiveEntry *DoNewEntry() const wxOVERRIDE
        { return NewEntry(); }
    wxArchiveInputStream *DoNewStream(wxInputStream& stream) const wxOVERRIDE
        { return NewStream(stream); }
    wxArchiveOutputStream *DoNewStream(wxOutputStream& stream) const wxOVERRIDE
        { return NewStream(stream); }
    wxArchiveInputStream *DoNewStream(wxInputStream *stream) const wxOVERRIDE
        { return NewStream(stream); }
    wxArchiveOutputStream *DoNewStream(wxOutputStream *stream) const wxOVERRIDE
        { return NewStream(stream); }

private:
    wxDECLARE_DYNAMIC_CLASS(wxZipClassFactory);
};


/////////////////////////////////////////////////////////////////////////////
// wxZipEntry inlines

inline bool wxZipEntry::IsText() const
{
    return (m_InternalAttributes & TEXT_ATTR) != 0;
}

inline bool wxZipEntry::IsDir() const
{
    return (m_ExternalAttributes & wxZIP_A_SUBDIR) != 0;
}

inline bool wxZipEntry::IsReadOnly() const
{
    return (m_ExternalAttributes & wxZIP_A_RDONLY) != 0;
}

inline bool wxZipEntry::IsMadeByUnix() const
{
    switch ( m_SystemMadeBy )
    {
        case wxZIP_SYSTEM_MSDOS:
            // note: some unix zippers put madeby = dos
            return (m_ExternalAttributes & ~0xFFFF) != 0;

        case wxZIP_SYSTEM_OPENVMS:
        case wxZIP_SYSTEM_UNIX:
        case wxZIP_SYSTEM_ATARI_ST:
        case wxZIP_SYSTEM_ACORN_RISC:
        case wxZIP_SYSTEM_BEOS:
        case wxZIP_SYSTEM_TANDEM:
            return true;

        case wxZIP_SYSTEM_AMIGA:
        case wxZIP_SYSTEM_VM_CMS:
        case wxZIP_SYSTEM_OS2_HPFS:
        case wxZIP_SYSTEM_MACINTOSH:
        case wxZIP_SYSTEM_Z_SYSTEM:
        case wxZIP_SYSTEM_CPM:
        case wxZIP_SYSTEM_WINDOWS_NTFS:
        case wxZIP_SYSTEM_MVS:
        case wxZIP_SYSTEM_VSE:
        case wxZIP_SYSTEM_VFAT:
        case wxZIP_SYSTEM_ALTERNATE_MVS:
        case wxZIP_SYSTEM_OS_400:
            return false;
    }

    // Unknown system, assume not Unix.
    return false;
}

inline void wxZipEntry::SetIsText(bool isText)
{
    if (isText)
        m_InternalAttributes |= TEXT_ATTR;
    else
        m_InternalAttributes &= ~TEXT_ATTR;
}

inline void wxZipEntry::SetIsReadOnly(bool isReadOnly)
{
    if (isReadOnly)
        SetMode(GetMode() & ~0222);
    else
        SetMode(GetMode() | 0200);
}

inline void wxZipEntry::SetName(const wxString& name,
                                wxPathFormat format /*=wxPATH_NATIVE*/)
{
    bool isDir;
    m_Name = GetInternalName(name, format, &isDir);
    SetIsDir(isDir);
}


#endif // wxUSE_ZIPSTREAM

#endif // _WX_WXZIPSTREAM_H__
