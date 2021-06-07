///////////////////////////////////////////////////////////////////////////////
// Name:        wx/stringops.h
// Purpose:     implementation of wxString primitive operations
// Author:      Vaclav Slavik
// Modified by:
// Created:     2007-04-16
// Copyright:   (c) 2007 REA Elektronik GmbH
// Licence:     wxWindows licence
///////////////////////////////////////////////////////////////////////////////

#ifndef _WX_WXSTRINGOPS_H__
#define _WX_WXSTRINGOPS_H__

#include "wx/chartype.h"
#include "wx/stringimpl.h"
#include "wx/unichar.h"
#include "wx/buffer.h"

// This header contains wxStringOperations "namespace" class that implements
// elementary operations on string data as static methods; wxString methods and
// iterators are implemented in terms of it. Two implementations are available,
// one for UTF-8 encoded char* string and one for "raw" wchar_t* strings (or
// char* in ANSI build).

// FIXME-UTF8: only wchar after we remove ANSI build
#if wxUSE_UNICODE_WCHAR || !wxUSE_UNICODE
struct WXDLLIMPEXP_BASE wxStringOperationsWchar
{
    // moves the iterator to the next Unicode character
    template <typename Iterator>
    static void IncIter(Iterator& i) { ++i; }

    // moves the iterator to the previous Unicode character
    template <typename Iterator>
    static void DecIter(Iterator& i) { --i; }

    // moves the iterator by n Unicode characters
    template <typename Iterator>
    static Iterator AddToIter(const Iterator& i, ptrdiff_t n)
        { return i + n; }

    // returns distance of the two iterators in Unicode characters
    template <typename Iterator>
    static ptrdiff_t DiffIters(const Iterator& i1, const Iterator& i2)
        { return i1 - i2; }

#if wxUSE_UNICODE_UTF16
    // encodes the characters as UTF-16:
    struct Utf16CharBuffer
    {
        // Notice that data is left uninitialized, it is filled by EncodeChar()
        // which is the only function creating objects of this class.

        wchar_t data[3];
        operator const wchar_t*() const { return data; }
    };
    static Utf16CharBuffer EncodeChar(const wxUniChar& ch);
    static wxWCharBuffer EncodeNChars(size_t n, const wxUniChar& ch);
    static bool IsSingleCodeUnitCharacter(const wxUniChar& ch)
        { return !ch.IsSupplementary(); }
#else
    // encodes the character to a form used to represent it in internal
    // representation
    struct SingleCharBuffer
    {
        wxChar data[2];
        operator const wxChar*() const { return data; }
    };
    static SingleCharBuffer EncodeChar(const wxUniChar& ch)
    {
        SingleCharBuffer buf;
        buf.data[0] = (wxChar)ch;
        buf.data[1] = 0;
        return buf;
    }
    static wxWxCharBuffer EncodeNChars(size_t n, const wxUniChar& ch);
    static bool IsSingleCodeUnitCharacter(const wxUniChar&) { return true; }
#endif

    static wxUniChar DecodeChar(const wxStringImpl::const_iterator& i)
        { return *i; }
};
#endif // wxUSE_UNICODE_WCHAR || !wxUSE_UNICODE


#if wxUSE_UNICODE_UTF8
struct WXDLLIMPEXP_BASE wxStringOperationsUtf8
{
    // checks correctness of UTF-8 sequence
    static bool IsValidUtf8String(const char *c,
                                  size_t len = wxStringImpl::npos);
    static bool IsValidUtf8LeadByte(unsigned char c)
    {
        return (c <= 0x7F) || (c >= 0xC2 && c <= 0xF4);
    }

    // returns offset to skip forward when iterating over UTF-8 sequence
    static unsigned char GetUTF8IterOffset(unsigned char c);


    template<typename Iterator>
    static void IncIter(Iterator& i)
    {
        wxASSERT( IsValidUtf8LeadByte(*i) );
        i += GetUTF8IterOffset(*i);
    }

    template<typename Iterator>
    static void DecIter(Iterator& i)
    {
        // Non-lead bytes are all in the 0x80..0xBF range (i.e. 10xxxxxx in
        // binary), so we just have to go back until we hit a byte that is
        // either < 0x80 (i.e. 0xxxxxxx in binary) or 0xC0..0xFF (11xxxxxx in
        // binary; this includes some invalid values, but we can ignore it
        // here, because we assume valid UTF-8 input for the purpose of
        // efficient implementation).
        --i;
        while ( ((*i) & 0xC0) == 0x80 /* 2 highest bits are '10' */ )
            --i;
    }

    template<typename Iterator>
    static Iterator AddToIter(const Iterator& i, ptrdiff_t n)
    {
        Iterator out(i);

        if ( n > 0 )
        {
            for ( ptrdiff_t j = 0; j < n; ++j )
                IncIter(out);
        }
        else if ( n < 0 )
        {
            for ( ptrdiff_t j = 0; j > n; --j )
                DecIter(out);
        }

        return out;
    }

    template<typename Iterator>
    static ptrdiff_t DiffIters(Iterator i1, Iterator i2)
    {
        ptrdiff_t dist = 0;

        if ( i1 < i2 )
        {
            while ( i1 != i2 )
            {
                IncIter(i1);
                dist--;
            }
        }
        else if ( i2 < i1 )
        {
            while ( i2 != i1 )
            {
                IncIter(i2);
                dist++;
            }
        }

        return dist;
    }

    static bool IsSingleCodeUnitCharacter(const wxUniChar& ch)
        { return ch.IsAscii(); }

    // encodes the character as UTF-8:
    typedef wxUniChar::Utf8CharBuffer Utf8CharBuffer;
    static Utf8CharBuffer EncodeChar(const wxUniChar& ch)
        { return ch.AsUTF8(); }

    // returns n copies of ch encoded in UTF-8 string
    static wxCharBuffer EncodeNChars(size_t n, const wxUniChar& ch);

    // returns the length of UTF-8 encoding of the character with lead byte 'c'
    static size_t GetUtf8CharLength(char c)
    {
        wxASSERT( IsValidUtf8LeadByte(c) );
        return GetUTF8IterOffset(c);
    }

    // decodes single UTF-8 character from UTF-8 string
    static wxUniChar DecodeChar(wxStringImpl::const_iterator i)
    {
        if ( (unsigned char)*i < 0x80 )
            return (int)*i;
        return DecodeNonAsciiChar(i);
    }

private:
    static wxUniChar DecodeNonAsciiChar(wxStringImpl::const_iterator i);
};
#endif // wxUSE_UNICODE_UTF8


#if wxUSE_UNICODE_UTF8
typedef wxStringOperationsUtf8 wxStringOperations;
#else
typedef wxStringOperationsWchar wxStringOperations;
#endif

#endif  // _WX_WXSTRINGOPS_H_
