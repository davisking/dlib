// std::messages implementation details, GNU version -*- C++ -*-

// Copyright (C) 2001-2019 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/** @file bits/messages_members.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{locale}
 */

//
// ISO C++ 14882: 22.2.7.1.2  messages functions
//

// Written by Benjamin Kosnik <bkoz@redhat.com>

#include <libintl.h>

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  // Non-virtual member functions.
  template<typename _CharT>
    messages<_CharT>::messages(size_t __refs)
    : facet(__refs), _M_c_locale_messages(_S_get_c_locale()),
      _M_name_messages(_S_get_c_name())
    { }

  template<typename _CharT>
    messages<_CharT>::messages(__c_locale __cloc, const char* __s,
			       size_t __refs)
    : facet(__refs), _M_c_locale_messages(0), _M_name_messages(0)
    {
      if (__builtin_strcmp(__s, _S_get_c_name()) != 0)
	{
	  const size_t __len = __builtin_strlen(__s) + 1;
	  char* __tmp = new char[__len];
	  __builtin_memcpy(__tmp, __s, __len);
	  _M_name_messages = __tmp;
	}
      else
	_M_name_messages = _S_get_c_name();

      // Last to avoid leaking memory if new throws.
      _M_c_locale_messages = _S_clone_c_locale(__cloc);
    }

  template<typename _CharT>
    typename messages<_CharT>::catalog
    messages<_CharT>::open(const basic_string<char>& __s, const locale& __loc,
			   const char* __dir) const
    {
      bindtextdomain(__s.c_str(), __dir);
      return this->do_open(__s, __loc);
    }

  // Virtual member functions.
  template<typename _CharT>
    messages<_CharT>::~messages()
    {
      if (_M_name_messages != _S_get_c_name())
	delete [] _M_name_messages;
      _S_destroy_c_locale(_M_c_locale_messages);
    }

  template<typename _CharT>
    typename messages<_CharT>::catalog
    messages<_CharT>::do_open(const basic_string<char>& __s,
			      const locale&) const
    {
      // No error checking is done, assume the catalog exists and can
      // be used.
      textdomain(__s.c_str());
      return 0;
    }

  template<typename _CharT>
    void
    messages<_CharT>::do_close(catalog) const
    { }

  // messages_byname
  template<typename _CharT>
    messages_byname<_CharT>::messages_byname(const char* __s, size_t __refs)
    : messages<_CharT>(__refs)
    {
      if (this->_M_name_messages != locale::facet::_S_get_c_name())
	{
	  delete [] this->_M_name_messages;
	  if (__builtin_strcmp(__s, locale::facet::_S_get_c_name()) != 0)
	    {
	      const size_t __len = __builtin_strlen(__s) + 1;
	      char* __tmp = new char[__len];
	      __builtin_memcpy(__tmp, __s, __len);
	      this->_M_name_messages = __tmp;
	    }
	  else
	    this->_M_name_messages = locale::facet::_S_get_c_name();
	}

      if (__builtin_strcmp(__s, "C") != 0
	  && __builtin_strcmp(__s, "POSIX") != 0)
	{
	  this->_S_destroy_c_locale(this->_M_c_locale_messages);
	  this->_S_create_c_locale(this->_M_c_locale_messages, __s);
	}
    }

   //Specializations.
  template<>
    typename messages<char>::catalog
    messages<char>::do_open(const basic_string<char>&,
			    const locale&) const;

  template<>
    void
    messages<char>::do_close(catalog) const;

#ifdef _GLIBCXX_USE_WCHAR_T
  template<>
    typename messages<wchar_t>::catalog
    messages<wchar_t>::do_open(const basic_string<char>&,
			       const locale&) const;

  template<>
    void
    messages<wchar_t>::do_close(catalog) const;
#endif

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace
