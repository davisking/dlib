/////////////////////////////////////////////////////////////////////////////
// Name:        wx/sckaddr.h
// Purpose:     Network address classes
// Author:      Guilhem Lavaux
// Modified by: Vadim Zeitlin to switch to wxSockAddressImpl implementation
// Created:     26/04/1997
// Copyright:   (c) 1997, 1998 Guilhem Lavaux
//              (c) 2008, 2009 Vadim Zeitlin
// Licence:     wxWindows licence
/////////////////////////////////////////////////////////////////////////////

#ifndef _WX_SCKADDR_H_
#define _WX_SCKADDR_H_

#include "wx/defs.h"

#if wxUSE_SOCKETS

#include "wx/string.h"

class wxSockAddressImpl;

// forward declare it instead of including the system headers defining it which
// can bring in <windows.h> under Windows which we don't want to include from
// public wx headers
struct sockaddr;

// Any socket address kind
class WXDLLIMPEXP_NET wxSockAddress : public wxObject
{
public:
    enum Family
    {
        NONE,
        IPV4,
        IPV6,
        UNIX
    };

    wxSockAddress();
    wxSockAddress(const wxSockAddress& other);
    virtual ~wxSockAddress();

    wxSockAddress& operator=(const wxSockAddress& other);

    virtual void Clear();
    virtual Family Type() = 0;

    // accessors for the low level address represented by this object
    const sockaddr *GetAddressData() const;
    int GetAddressDataLen() const;

    // we need to be able to create copies of the addresses polymorphically
    // (i.e. without knowing the exact address class)
    virtual wxSockAddress *Clone() const = 0;


    // implementation only, don't use
    const wxSockAddressImpl& GetAddress() const { return *m_impl; }
    void SetAddress(const wxSockAddressImpl& address);

protected:
    wxSockAddressImpl *m_impl;

private:
    void Init();
    wxDECLARE_ABSTRACT_CLASS(wxSockAddress);
};

// An IP address (either IPv4 or IPv6)
class WXDLLIMPEXP_NET wxIPaddress : public wxSockAddress
{
public:
    bool operator==(const wxIPaddress& addr) const;

    bool Hostname(const wxString& name);
    bool Service(const wxString& name);
    bool Service(unsigned short port);

    bool LocalHost();
    virtual bool IsLocalHost() const = 0;

    bool AnyAddress();

    virtual wxString IPAddress() const = 0;

    wxString Hostname() const;
    unsigned short Service() const;

    wxString OrigHostname() const { return m_origHostname; }

protected:
    // get m_impl initialized to the right family if it hadn't been done yet
    wxSockAddressImpl& GetImpl();
    const wxSockAddressImpl& GetImpl() const
    {
        return const_cast<wxIPaddress *>(this)->GetImpl();
    }

    // host name originally passed to Hostname()
    wxString m_origHostname;

private:
    // create the wxSockAddressImpl object of the correct family if it's
    // currently uninitialized
    virtual void DoInitImpl() = 0;


    wxDECLARE_ABSTRACT_CLASS(wxIPaddress);
};

// An IPv4 address
class WXDLLIMPEXP_NET wxIPV4address : public wxIPaddress
{
public:
    // implement wxSockAddress pure virtuals:
    virtual Family Type() wxOVERRIDE { return IPV4; }
    virtual wxSockAddress *Clone() const wxOVERRIDE { return new wxIPV4address(*this); }


    // implement wxIPaddress pure virtuals:
    virtual bool IsLocalHost() const wxOVERRIDE;

    virtual wxString IPAddress() const wxOVERRIDE;


    // IPv4-specific methods:
    bool Hostname(unsigned long addr);

    // make base class methods hidden by our overload visible
    using wxIPaddress::Hostname;

    bool BroadcastAddress();

private:
    virtual void DoInitImpl() wxOVERRIDE;

    wxDECLARE_DYNAMIC_CLASS(wxIPV4address);
};


#if wxUSE_IPV6

// An IPv6 address
class WXDLLIMPEXP_NET wxIPV6address : public wxIPaddress
{
public:
    // implement wxSockAddress pure virtuals:
    virtual Family Type() wxOVERRIDE { return IPV6; }
    virtual wxSockAddress *Clone() const wxOVERRIDE { return new wxIPV6address(*this); }


    // implement wxIPaddress pure virtuals:
    virtual bool IsLocalHost() const wxOVERRIDE;

    virtual wxString IPAddress() const wxOVERRIDE;

    // IPv6-specific methods:
    bool Hostname(unsigned char addr[16]);

    using wxIPaddress::Hostname;

private:
    virtual void DoInitImpl() wxOVERRIDE;

    wxDECLARE_DYNAMIC_CLASS(wxIPV6address);
};

#endif // wxUSE_IPV6

// Unix domain sockets are only available under, well, Unix
#if defined(__UNIX__) && !defined(__WINDOWS__) && !defined(__WINE__)
    #define wxHAS_UNIX_DOMAIN_SOCKETS
#endif

#ifdef wxHAS_UNIX_DOMAIN_SOCKETS

// A Unix domain socket address
class WXDLLIMPEXP_NET wxUNIXaddress : public wxSockAddress
{
public:
    void Filename(const wxString& name);
    wxString Filename() const;

    virtual Family Type() wxOVERRIDE { return UNIX; }
    virtual wxSockAddress *Clone() const wxOVERRIDE { return new wxUNIXaddress(*this); }

private:
    wxSockAddressImpl& GetUNIX();
    const wxSockAddressImpl& GetUNIX() const
    {
        return const_cast<wxUNIXaddress *>(this)->GetUNIX();
    }

    wxDECLARE_DYNAMIC_CLASS(wxUNIXaddress);
};

#endif // wxHAS_UNIX_DOMAIN_SOCKETS

#endif // wxUSE_SOCKETS

#endif // _WX_SCKADDR_H_
