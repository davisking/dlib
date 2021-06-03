// Locale support -*- C++ -*-

// Copyright (C) 2007-2019 Free Software Foundation, Inc.
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

/** @file bits/locale_facets_nonio.tcc
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{locale}
 */

#ifndef _LOCALE_FACETS_NONIO_TCC
#define _LOCALE_FACETS_NONIO_TCC 1

#pragma GCC system_header

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  template<typename _CharT, bool _Intl>
    struct __use_cache<__moneypunct_cache<_CharT, _Intl> >
    {
      const __moneypunct_cache<_CharT, _Intl>*
      operator() (const locale& __loc) const
      {
	const size_t __i = moneypunct<_CharT, _Intl>::id._M_id();
	const locale::facet** __caches = __loc._M_impl->_M_caches;
	if (!__caches[__i])
	  {
	    __moneypunct_cache<_CharT, _Intl>* __tmp = 0;
	    __try
	      {
		__tmp = new __moneypunct_cache<_CharT, _Intl>;
		__tmp->_M_cache(__loc);
	      }
	    __catch(...)
	      {
		delete __tmp;
		__throw_exception_again;
	      }
	    __loc._M_impl->_M_install_cache(__tmp, __i);
	  }
	return static_cast<
	  const __moneypunct_cache<_CharT, _Intl>*>(__caches[__i]);
      }
    };

  template<typename _CharT, bool _Intl>
    void
    __moneypunct_cache<_CharT, _Intl>::_M_cache(const locale& __loc)
    {
      const moneypunct<_CharT, _Intl>& __mp =
	use_facet<moneypunct<_CharT, _Intl> >(__loc);

      _M_decimal_point = __mp.decimal_point();
      _M_thousands_sep = __mp.thousands_sep();
      _M_frac_digits = __mp.frac_digits();

      char* __grouping = 0;
      _CharT* __curr_symbol = 0;
      _CharT* __positive_sign = 0;
      _CharT* __negative_sign = 0;     
      __try
	{
	  const string& __g = __mp.grouping();
	  _M_grouping_size = __g.size();
	  __grouping = new char[_M_grouping_size];
	  __g.copy(__grouping, _M_grouping_size);
	  _M_use_grouping = (_M_grouping_size
			     && static_cast<signed char>(__grouping[0]) > 0
			     && (__grouping[0]
				 != __gnu_cxx::__numeric_traits<char>::__max));

	  const basic_string<_CharT>& __cs = __mp.curr_symbol();
	  _M_curr_symbol_size = __cs.size();
	  __curr_symbol = new _CharT[_M_curr_symbol_size];
	  __cs.copy(__curr_symbol, _M_curr_symbol_size);

	  const basic_string<_CharT>& __ps = __mp.positive_sign();
	  _M_positive_sign_size = __ps.size();
	  __positive_sign = new _CharT[_M_positive_sign_size];
	  __ps.copy(__positive_sign, _M_positive_sign_size);

	  const basic_string<_CharT>& __ns = __mp.negative_sign();
	  _M_negative_sign_size = __ns.size();
	  __negative_sign = new _CharT[_M_negative_sign_size];
	  __ns.copy(__negative_sign, _M_negative_sign_size);

	  _M_pos_format = __mp.pos_format();
	  _M_neg_format = __mp.neg_format();

	  const ctype<_CharT>& __ct = use_facet<ctype<_CharT> >(__loc);
	  __ct.widen(money_base::_S_atoms,
		     money_base::_S_atoms + money_base::_S_end, _M_atoms);

	  _M_grouping = __grouping;
	  _M_curr_symbol = __curr_symbol;
	  _M_positive_sign = __positive_sign;
	  _M_negative_sign = __negative_sign;
	  _M_allocated = true;
	}
      __catch(...)
	{
	  delete [] __grouping;
	  delete [] __curr_symbol;
	  delete [] __positive_sign;
	  delete [] __negative_sign;
	  __throw_exception_again;
	}
    }

_GLIBCXX_BEGIN_NAMESPACE_LDBL_OR_CXX11

  template<typename _CharT, typename _InIter>
    template<bool _Intl>
      _InIter
      money_get<_CharT, _InIter>::
      _M_extract(iter_type __beg, iter_type __end, ios_base& __io,
		 ios_base::iostate& __err, string& __units) const
      {
	typedef char_traits<_CharT>			  __traits_type;
	typedef typename string_type::size_type	          size_type;	
	typedef money_base::part			  part;
	typedef __moneypunct_cache<_CharT, _Intl>         __cache_type;
	
	const locale& __loc = __io._M_getloc();
	const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);

	__use_cache<__cache_type> __uc;
	const __cache_type* __lc = __uc(__loc);
	const char_type* __lit = __lc->_M_atoms;

	// Deduced sign.
	bool __negative = false;
	// Sign size.
	size_type __sign_size = 0;
	// True if sign is mandatory.
	const bool __mandatory_sign = (__lc->_M_positive_sign_size
				       && __lc->_M_negative_sign_size);
	// String of grouping info from thousands_sep plucked from __units.
	string __grouping_tmp;
	if (__lc->_M_use_grouping)
	  __grouping_tmp.reserve(32);
	// Last position before the decimal point.
	int __last_pos = 0;
	// Separator positions, then, possibly, fractional digits.
	int __n = 0;
	// If input iterator is in a valid state.
	bool __testvalid = true;
	// Flag marking when a decimal point is found.
	bool __testdecfound = false;

	// The tentative returned string is stored here.
	string __res;
	__res.reserve(32);

	const char_type* __lit_zero = __lit + money_base::_S_zero;
	const money_base::pattern __p = __lc->_M_neg_format;
	for (int __i = 0; __i < 4 && __testvalid; ++__i)
	  {
	    const part __which = static_cast<part>(__p.field[__i]);
	    switch (__which)
	      {
	      case money_base::symbol:
		// According to 22.2.6.1.2, p2, symbol is required
		// if (__io.flags() & ios_base::showbase), otherwise
		// is optional and consumed only if other characters
		// are needed to complete the format.
		if (__io.flags() & ios_base::showbase || __sign_size > 1
		    || __i == 0
		    || (__i == 1 && (__mandatory_sign
				     || (static_cast<part>(__p.field[0])
					 == money_base::sign)
				     || (static_cast<part>(__p.field[2])
					 == money_base::space)))
		    || (__i == 2 && ((static_cast<part>(__p.field[3])
				      == money_base::value)
				     || (__mandatory_sign
					 && (static_cast<part>(__p.field[3])
					     == money_base::sign)))))
		  {
		    const size_type __len = __lc->_M_curr_symbol_size;
		    size_type __j = 0;
		    for (; __beg != __end && __j < __len
			   && *__beg == __lc->_M_curr_symbol[__j];
			 ++__beg, (void)++__j);
		    if (__j != __len
			&& (__j || __io.flags() & ios_base::showbase))
		      __testvalid = false;
		  }
		break;
	      case money_base::sign:
		// Sign might not exist, or be more than one character long.
		if (__lc->_M_positive_sign_size && __beg != __end
		    && *__beg == __lc->_M_positive_sign[0])
		  {
		    __sign_size = __lc->_M_positive_sign_size;
		    ++__beg;
		  }
		else if (__lc->_M_negative_sign_size && __beg != __end
			 && *__beg == __lc->_M_negative_sign[0])
		  {
		    __negative = true;
		    __sign_size = __lc->_M_negative_sign_size;
		    ++__beg;
		  }
		else if (__lc->_M_positive_sign_size
			 && !__lc->_M_negative_sign_size)
		  // "... if no sign is detected, the result is given the sign
		  // that corresponds to the source of the empty string"
		  __negative = true;
		else if (__mandatory_sign)
		  __testvalid = false;
		break;
	      case money_base::value:
		// Extract digits, remove and stash away the
		// grouping of found thousands separators.
		for (; __beg != __end; ++__beg)
		  {
		    const char_type __c = *__beg;
		    const char_type* __q = __traits_type::find(__lit_zero, 
							       10, __c);
		    if (__q != 0)
		      {
			__res += money_base::_S_atoms[__q - __lit];
			++__n;
		      }
		    else if (__c == __lc->_M_decimal_point 
			     && !__testdecfound)
		      {
			if (__lc->_M_frac_digits <= 0)
			  break;

			__last_pos = __n;
			__n = 0;
			__testdecfound = true;
		      }
		    else if (__lc->_M_use_grouping
			     && __c == __lc->_M_thousands_sep
			     && !__testdecfound)
		      {
			if (__n)
			  {
			    // Mark position for later analysis.
			    __grouping_tmp += static_cast<char>(__n);
			    __n = 0;
			  }
			else
			  {
			    __testvalid = false;
			    break;
			  }
		      }
		    else
		      break;
		  }
		if (__res.empty())
		  __testvalid = false;
		break;
	      case money_base::space:
		// At least one space is required.
		if (__beg != __end && __ctype.is(ctype_base::space, *__beg))
		  ++__beg;
		else
		  __testvalid = false;
		// fallthrough
	      case money_base::none:
		// Only if not at the end of the pattern.
		if (__i != 3)
		  for (; __beg != __end
			 && __ctype.is(ctype_base::space, *__beg); ++__beg);
		break;
	      }
	  }

	// Need to get the rest of the sign characters, if they exist.
	if (__sign_size > 1 && __testvalid)
	  {
	    const char_type* __sign = __negative ? __lc->_M_negative_sign
	                                         : __lc->_M_positive_sign;
	    size_type __i = 1;
	    for (; __beg != __end && __i < __sign_size
		   && *__beg == __sign[__i]; ++__beg, (void)++__i);
	    
	    if (__i != __sign_size)
	      __testvalid = false;
	  }

	if (__testvalid)
	  {
	    // Strip leading zeros.
	    if (__res.size() > 1)
	      {
		const size_type __first = __res.find_first_not_of('0');
		const bool __only_zeros = __first == string::npos;
		if (__first)
		  __res.erase(0, __only_zeros ? __res.size() - 1 : __first);
	      }

	    // 22.2.6.1.2, p4
	    if (__negative && __res[0] != '0')
	      __res.insert(__res.begin(), '-');
	    
	    // Test for grouping fidelity.
	    if (__grouping_tmp.size())
	      {
		// Add the ending grouping.
		__grouping_tmp += static_cast<char>(__testdecfound ? __last_pos
						                   : __n);
		if (!std::__verify_grouping(__lc->_M_grouping,
					    __lc->_M_grouping_size,
					    __grouping_tmp))
		  __err |= ios_base::failbit;
	      }
	    
	    // Iff not enough digits were supplied after the decimal-point.
	    if (__testdecfound && __n != __lc->_M_frac_digits)
	      __testvalid = false;
	  }

	// Iff valid sequence is not recognized.
	if (!__testvalid)
	  __err |= ios_base::failbit;
	else
	  __units.swap(__res);
	
	// Iff no more characters are available.
	if (__beg == __end)
	  __err |= ios_base::eofbit;
	return __beg;
      }

#if defined _GLIBCXX_LONG_DOUBLE_COMPAT && defined __LONG_DOUBLE_128__ \
      && _GLIBCXX_USE_CXX11_ABI == 0
  template<typename _CharT, typename _InIter>
    _InIter
    money_get<_CharT, _InIter>::
    __do_get(iter_type __beg, iter_type __end, bool __intl, ios_base& __io,
	     ios_base::iostate& __err, double& __units) const
    {
      string __str;
      __beg = __intl ? _M_extract<true>(__beg, __end, __io, __err, __str)
                     : _M_extract<false>(__beg, __end, __io, __err, __str);
      std::__convert_to_v(__str.c_str(), __units, __err, _S_get_c_locale());
      return __beg;
    }
#endif

  template<typename _CharT, typename _InIter>
    _InIter
    money_get<_CharT, _InIter>::
    do_get(iter_type __beg, iter_type __end, bool __intl, ios_base& __io,
	   ios_base::iostate& __err, long double& __units) const
    {
      string __str;
      __beg = __intl ? _M_extract<true>(__beg, __end, __io, __err, __str)
	             : _M_extract<false>(__beg, __end, __io, __err, __str);
      std::__convert_to_v(__str.c_str(), __units, __err, _S_get_c_locale());
      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    money_get<_CharT, _InIter>::
    do_get(iter_type __beg, iter_type __end, bool __intl, ios_base& __io,
	   ios_base::iostate& __err, string_type& __digits) const
    {
      typedef typename string::size_type                  size_type;

      const locale& __loc = __io._M_getloc();
      const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);

      string __str;
      __beg = __intl ? _M_extract<true>(__beg, __end, __io, __err, __str)
	             : _M_extract<false>(__beg, __end, __io, __err, __str);
      const size_type __len = __str.size();
      if (__len)
	{
	  __digits.resize(__len);
	  __ctype.widen(__str.data(), __str.data() + __len, &__digits[0]);
	}
      return __beg;
    }

  template<typename _CharT, typename _OutIter>
    template<bool _Intl>
      _OutIter
      money_put<_CharT, _OutIter>::
      _M_insert(iter_type __s, ios_base& __io, char_type __fill,
		const string_type& __digits) const
      {
	typedef typename string_type::size_type	          size_type;
	typedef money_base::part                          part;
	typedef __moneypunct_cache<_CharT, _Intl>         __cache_type;
      
	const locale& __loc = __io._M_getloc();
	const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);

	__use_cache<__cache_type> __uc;
	const __cache_type* __lc = __uc(__loc);
	const char_type* __lit = __lc->_M_atoms;

	// Determine if negative or positive formats are to be used, and
	// discard leading negative_sign if it is present.
	const char_type* __beg = __digits.data();

	money_base::pattern __p;
	const char_type* __sign;
	size_type __sign_size;
	if (!(*__beg == __lit[money_base::_S_minus]))
	  {
	    __p = __lc->_M_pos_format;
	    __sign = __lc->_M_positive_sign;
	    __sign_size = __lc->_M_positive_sign_size;
	  }
	else
	  {
	    __p = __lc->_M_neg_format;
	    __sign = __lc->_M_negative_sign;
	    __sign_size = __lc->_M_negative_sign_size;
	    if (__digits.size())
	      ++__beg;
	  }
       
	// Look for valid numbers in the ctype facet within input digits.
	size_type __len = __ctype.scan_not(ctype_base::digit, __beg,
					   __beg + __digits.size()) - __beg;
	if (__len)
	  {
	    // Assume valid input, and attempt to format.
	    // Break down input numbers into base components, as follows:
	    //   final_value = grouped units + (decimal point) + (digits)
	    string_type __value;
	    __value.reserve(2 * __len);

	    // Add thousands separators to non-decimal digits, per
	    // grouping rules.
	    long __paddec = __len - __lc->_M_frac_digits;
	    if (__paddec > 0)
  	      {
		if (__lc->_M_frac_digits < 0)
		  __paddec = __len;
  		if (__lc->_M_grouping_size)
  		  {
		    __value.assign(2 * __paddec, char_type());
 		    _CharT* __vend = 
		      std::__add_grouping(&__value[0], __lc->_M_thousands_sep,
					  __lc->_M_grouping,
					  __lc->_M_grouping_size,
					  __beg, __beg + __paddec);
		    __value.erase(__vend - &__value[0]);
  		  }
  		else
		  __value.assign(__beg, __paddec);
	      }

	    // Deal with decimal point, decimal digits.
	    if (__lc->_M_frac_digits > 0)
	      {
		__value += __lc->_M_decimal_point;
		if (__paddec >= 0)
		  __value.append(__beg + __paddec, __lc->_M_frac_digits);
		else
		  {
		    // Have to pad zeros in the decimal position.
		    __value.append(-__paddec, __lit[money_base::_S_zero]);
		    __value.append(__beg, __len);
		  }
  	      }
  
	    // Calculate length of resulting string.
	    const ios_base::fmtflags __f = __io.flags() 
	                                   & ios_base::adjustfield;
	    __len = __value.size() + __sign_size;
	    __len += ((__io.flags() & ios_base::showbase)
		      ? __lc->_M_curr_symbol_size : 0);

	    string_type __res;
	    __res.reserve(2 * __len);
	    
	    const size_type __width = static_cast<size_type>(__io.width());  
	    const bool __testipad = (__f == ios_base::internal
				     && __len < __width);
	    // Fit formatted digits into the required pattern.
	    for (int __i = 0; __i < 4; ++__i)
	      {
		const part __which = static_cast<part>(__p.field[__i]);
		switch (__which)
		  {
		  case money_base::symbol:
		    if (__io.flags() & ios_base::showbase)
		      __res.append(__lc->_M_curr_symbol,
				   __lc->_M_curr_symbol_size);
		    break;
		  case money_base::sign:
		    // Sign might not exist, or be more than one
		    // character long. In that case, add in the rest
		    // below.
		    if (__sign_size)
		      __res += __sign[0];
		    break;
		  case money_base::value:
		    __res += __value;
		    break;
		  case money_base::space:
		    // At least one space is required, but if internal
		    // formatting is required, an arbitrary number of
		    // fill spaces will be necessary.
		    if (__testipad)
		      __res.append(__width - __len, __fill);
		    else
		      __res += __fill;
		    break;
		  case money_base::none:
		    if (__testipad)
		      __res.append(__width - __len, __fill);
		    break;
		  }
	      }
	    
	    // Special case of multi-part sign parts.
	    if (__sign_size > 1)
	      __res.append(__sign + 1, __sign_size - 1);
	    
	    // Pad, if still necessary.
	    __len = __res.size();
	    if (__width > __len)
	      {
		if (__f == ios_base::left)
		  // After.
		  __res.append(__width - __len, __fill);
		else
		  // Before.
		  __res.insert(0, __width - __len, __fill);
		__len = __width;
	      }
	    
	    // Write resulting, fully-formatted string to output iterator.
	    __s = std::__write(__s, __res.data(), __len);
	  }
	__io.width(0);
	return __s;    
      }

#if defined _GLIBCXX_LONG_DOUBLE_COMPAT && defined __LONG_DOUBLE_128__ \
      && _GLIBCXX_USE_CXX11_ABI == 0
  template<typename _CharT, typename _OutIter>
    _OutIter
    money_put<_CharT, _OutIter>::
    __do_put(iter_type __s, bool __intl, ios_base& __io, char_type __fill,
	     double __units) const
    { return this->do_put(__s, __intl, __io, __fill, (long double) __units); }
#endif

  template<typename _CharT, typename _OutIter>
    _OutIter
    money_put<_CharT, _OutIter>::
    do_put(iter_type __s, bool __intl, ios_base& __io, char_type __fill,
	   long double __units) const
    {
      const locale __loc = __io.getloc();
      const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);
#if _GLIBCXX_USE_C99_STDIO
      // First try a buffer perhaps big enough.
      int __cs_size = 64;
      char* __cs = static_cast<char*>(__builtin_alloca(__cs_size));
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 328. Bad sprintf format modifier in money_put<>::do_put()
      int __len = std::__convert_from_v(_S_get_c_locale(), __cs, __cs_size,
					"%.*Lf", 0, __units);
      // If the buffer was not large enough, try again with the correct size.
      if (__len >= __cs_size)
	{
	  __cs_size = __len + 1;
	  __cs = static_cast<char*>(__builtin_alloca(__cs_size));
	  __len = std::__convert_from_v(_S_get_c_locale(), __cs, __cs_size,
					"%.*Lf", 0, __units);
	}
#else
      // max_exponent10 + 1 for the integer part, + 2 for sign and '\0'.
      const int __cs_size =
	__gnu_cxx::__numeric_traits<long double>::__max_exponent10 + 3;
      char* __cs = static_cast<char*>(__builtin_alloca(__cs_size));
      int __len = std::__convert_from_v(_S_get_c_locale(), __cs, 0, "%.*Lf", 
					0, __units);
#endif
      string_type __digits(__len, char_type());
      __ctype.widen(__cs, __cs + __len, &__digits[0]);
      return __intl ? _M_insert<true>(__s, __io, __fill, __digits)
	            : _M_insert<false>(__s, __io, __fill, __digits);
    }

  template<typename _CharT, typename _OutIter>
    _OutIter
    money_put<_CharT, _OutIter>::
    do_put(iter_type __s, bool __intl, ios_base& __io, char_type __fill,
	   const string_type& __digits) const
    { return __intl ? _M_insert<true>(__s, __io, __fill, __digits)
	            : _M_insert<false>(__s, __io, __fill, __digits); }

_GLIBCXX_END_NAMESPACE_LDBL_OR_CXX11

  // NB: Not especially useful. Without an ios_base object or some
  // kind of locale reference, we are left clawing at the air where
  // the side of the mountain used to be...
  template<typename _CharT, typename _InIter>
    time_base::dateorder
    time_get<_CharT, _InIter>::do_date_order() const
    { return time_base::no_order; }

  // Expand a strftime format string and parse it.  E.g., do_get_date() may
  // pass %m/%d/%Y => extracted characters.
  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    _M_extract_via_format(iter_type __beg, iter_type __end, ios_base& __io,
			  ios_base::iostate& __err, tm* __tm,
			  const _CharT* __format) const
    {
      const locale& __loc = __io._M_getloc();
      const __timepunct<_CharT>& __tp = use_facet<__timepunct<_CharT> >(__loc);
      const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);
      const size_t __len = char_traits<_CharT>::length(__format);

      ios_base::iostate __tmperr = ios_base::goodbit;
      size_t __i = 0;
      for (; __beg != __end && __i < __len && !__tmperr; ++__i)
	{
	  if (__ctype.narrow(__format[__i], 0) == '%')
	    {
	      // Verify valid formatting code, attempt to extract.
	      char __c = __ctype.narrow(__format[++__i], 0);
	      int __mem = 0;
	      if (__c == 'E' || __c == 'O')
		__c = __ctype.narrow(__format[++__i], 0);
	      switch (__c)
		{
		  const char* __cs;
		  _CharT __wcs[10];
		case 'a':
		  // Abbreviated weekday name [tm_wday]
		  const char_type*  __days1[7];
		  __tp._M_days_abbreviated(__days1);
		  __beg = _M_extract_name(__beg, __end, __mem, __days1,
					  7, __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_wday = __mem;
		  break;
		case 'A':
		  // Weekday name [tm_wday].
		  const char_type*  __days2[7];
		  __tp._M_days(__days2);
		  __beg = _M_extract_name(__beg, __end, __mem, __days2,
					  7, __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_wday = __mem;
		  break;
		case 'h':
		case 'b':
		  // Abbreviated month name [tm_mon]
		  const char_type*  __months1[12];
		  __tp._M_months_abbreviated(__months1);
		  __beg = _M_extract_name(__beg, __end, __mem,
					  __months1, 12, __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_mon = __mem;
		  break;
		case 'B':
		  // Month name [tm_mon].
		  const char_type*  __months2[12];
		  __tp._M_months(__months2);
		  __beg = _M_extract_name(__beg, __end, __mem,
					  __months2, 12, __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_mon = __mem;
		  break;
		case 'c':
		  // Default time and date representation.
		  const char_type*  __dt[2];
		  __tp._M_date_time_formats(__dt);
		  __beg = _M_extract_via_format(__beg, __end, __io, __tmperr, 
						__tm, __dt[0]);
		  break;
		case 'd':
		  // Day [01, 31]. [tm_mday]
		  __beg = _M_extract_num(__beg, __end, __mem, 1, 31, 2,
					 __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_mday = __mem;
		  break;
		case 'e':
		  // Day [1, 31], with single digits preceded by
		  // space. [tm_mday]
		  if (__ctype.is(ctype_base::space, *__beg))
		    __beg = _M_extract_num(++__beg, __end, __mem, 1, 9,
					   1, __io, __tmperr);
		  else
		    __beg = _M_extract_num(__beg, __end, __mem, 10, 31,
					   2, __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_mday = __mem;
		  break;
		case 'D':
		  // Equivalent to %m/%d/%y.[tm_mon, tm_mday, tm_year]
		  __cs = "%m/%d/%y";
		  __ctype.widen(__cs, __cs + 9, __wcs);
		  __beg = _M_extract_via_format(__beg, __end, __io, __tmperr, 
						__tm, __wcs);
		  break;
		case 'H':
		  // Hour [00, 23]. [tm_hour]
		  __beg = _M_extract_num(__beg, __end, __mem, 0, 23, 2,
					 __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_hour = __mem;
		  break;
		case 'I':
		  // Hour [01, 12]. [tm_hour]
		  __beg = _M_extract_num(__beg, __end, __mem, 1, 12, 2,
					 __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_hour = __mem;
		  break;
		case 'm':
		  // Month [01, 12]. [tm_mon]
		  __beg = _M_extract_num(__beg, __end, __mem, 1, 12, 2, 
					 __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_mon = __mem - 1;
		  break;
		case 'M':
		  // Minute [00, 59]. [tm_min]
		  __beg = _M_extract_num(__beg, __end, __mem, 0, 59, 2,
					 __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_min = __mem;
		  break;
		case 'n':
		  if (__ctype.narrow(*__beg, 0) == '\n')
		    ++__beg;
		  else
		    __tmperr |= ios_base::failbit;
		  break;
		case 'R':
		  // Equivalent to (%H:%M).
		  __cs = "%H:%M";
		  __ctype.widen(__cs, __cs + 6, __wcs);
		  __beg = _M_extract_via_format(__beg, __end, __io, __tmperr, 
						__tm, __wcs);
		  break;
		case 'S':
		  // Seconds. [tm_sec]
		  // [00, 60] in C99 (one leap-second), [00, 61] in C89.
#if _GLIBCXX_USE_C99
		  __beg = _M_extract_num(__beg, __end, __mem, 0, 60, 2,
#else
		  __beg = _M_extract_num(__beg, __end, __mem, 0, 61, 2,
#endif
					 __io, __tmperr);
		  if (!__tmperr)
		  __tm->tm_sec = __mem;
		  break;
		case 't':
		  if (__ctype.narrow(*__beg, 0) == '\t')
		    ++__beg;
		  else
		    __tmperr |= ios_base::failbit;
		  break;
		case 'T':
		  // Equivalent to (%H:%M:%S).
		  __cs = "%H:%M:%S";
		  __ctype.widen(__cs, __cs + 9, __wcs);
		  __beg = _M_extract_via_format(__beg, __end, __io, __tmperr, 
						__tm, __wcs);
		  break;
		case 'x':
		  // Locale's date.
		  const char_type*  __dates[2];
		  __tp._M_date_formats(__dates);
		  __beg = _M_extract_via_format(__beg, __end, __io, __tmperr, 
						__tm, __dates[0]);
		  break;
		case 'X':
		  // Locale's time.
		  const char_type*  __times[2];
		  __tp._M_time_formats(__times);
		  __beg = _M_extract_via_format(__beg, __end, __io, __tmperr, 
						__tm, __times[0]);
		  break;
		case 'y':
		case 'C': // C99
		  // Two digit year.
		case 'Y':
		  // Year [1900).
		  // NB: We parse either two digits, implicitly years since
		  // 1900, or 4 digits, full year.  In both cases we can 
		  // reconstruct [tm_year].  See also libstdc++/26701.
		  __beg = _M_extract_num(__beg, __end, __mem, 0, 9999, 4,
					 __io, __tmperr);
		  if (!__tmperr)
		    __tm->tm_year = __mem < 0 ? __mem + 100 : __mem - 1900;
		  break;
		case 'Z':
		  // Timezone info.
		  if (__ctype.is(ctype_base::upper, *__beg))
		    {
		      int __tmp;
		      __beg = _M_extract_name(__beg, __end, __tmp,
				       __timepunct_cache<_CharT>::_S_timezones,
					      14, __io, __tmperr);

		      // GMT requires special effort.
		      if (__beg != __end && !__tmperr && __tmp == 0
			  && (*__beg == __ctype.widen('-')
			      || *__beg == __ctype.widen('+')))
			{
			  __beg = _M_extract_num(__beg, __end, __tmp, 0, 23, 2,
						 __io, __tmperr);
			  __beg = _M_extract_num(__beg, __end, __tmp, 0, 59, 2,
						 __io, __tmperr);
			}
		    }
		  else
		    __tmperr |= ios_base::failbit;
		  break;
		default:
		  // Not recognized.
		  __tmperr |= ios_base::failbit;
		}
	    }
	  else
	    {
	      // Verify format and input match, extract and discard.
	      if (__format[__i] == *__beg)
		++__beg;
	      else
		__tmperr |= ios_base::failbit;
	    }
	}

      if (__tmperr || __i != __len)
	__err |= ios_base::failbit;
  
      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    _M_extract_num(iter_type __beg, iter_type __end, int& __member,
		   int __min, int __max, size_t __len,
		   ios_base& __io, ios_base::iostate& __err) const
    {
      const locale& __loc = __io._M_getloc();
      const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);

      // As-is works for __len = 1, 2, 4, the values actually used.
      int __mult = __len == 2 ? 10 : (__len == 4 ? 1000 : 1);

      ++__min;
      size_t __i = 0;
      int __value = 0;
      for (; __beg != __end && __i < __len; ++__beg, (void)++__i)
	{
	  const char __c = __ctype.narrow(*__beg, '*');
	  if (__c >= '0' && __c <= '9')
	    {
	      __value = __value * 10 + (__c - '0');
	      const int __valuec = __value * __mult;
	      if (__valuec > __max || __valuec + __mult < __min)
		break;
	      __mult /= 10;
	    }
	  else
	    break;
	}
      if (__i == __len)
	__member = __value;
      // Special encoding for do_get_year, 'y', and 'Y' above.
      else if (__len == 4 && __i == 2)
	__member = __value - 100;
      else
	__err |= ios_base::failbit;

      return __beg;
    }

  // Assumptions:
  // All elements in __names are unique.
  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    _M_extract_name(iter_type __beg, iter_type __end, int& __member,
		    const _CharT** __names, size_t __indexlen,
		    ios_base& __io, ios_base::iostate& __err) const
    {
      typedef char_traits<_CharT>		__traits_type;
      const locale& __loc = __io._M_getloc();
      const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);

      int* __matches = static_cast<int*>(__builtin_alloca(sizeof(int)
							  * __indexlen));
      size_t __nmatches = 0;
      size_t __pos = 0;
      bool __testvalid = true;
      const char_type* __name;

      // Look for initial matches.
      // NB: Some of the locale data is in the form of all lowercase
      // names, and some is in the form of initially-capitalized
      // names. Look for both.
      if (__beg != __end)
	{
	  const char_type __c = *__beg;
	  for (size_t __i1 = 0; __i1 < __indexlen; ++__i1)
	    if (__c == __names[__i1][0]
		|| __c == __ctype.toupper(__names[__i1][0]))
	      __matches[__nmatches++] = __i1;
	}

      while (__nmatches > 1)
	{
	  // Find smallest matching string.
	  size_t __minlen = __traits_type::length(__names[__matches[0]]);
	  for (size_t __i2 = 1; __i2 < __nmatches; ++__i2)
	    __minlen = std::min(__minlen,
			      __traits_type::length(__names[__matches[__i2]]));
	  ++__beg;
	  ++__pos;
	  if (__pos < __minlen && __beg != __end)
	    for (size_t __i3 = 0; __i3 < __nmatches;)
	      {
		__name = __names[__matches[__i3]];
		if (!(__name[__pos] == *__beg))
		  __matches[__i3] = __matches[--__nmatches];
		else
		  ++__i3;
	      }
	  else
	    break;
	}

      if (__nmatches == 1)
	{
	  // Make sure found name is completely extracted.
	  ++__beg;
	  ++__pos;
	  __name = __names[__matches[0]];
	  const size_t __len = __traits_type::length(__name);
	  while (__pos < __len && __beg != __end && __name[__pos] == *__beg)
	    ++__beg, (void)++__pos;

	  if (__len == __pos)
	    __member = __matches[0];
	  else
	    __testvalid = false;
	}
      else
	__testvalid = false;
      if (!__testvalid)
	__err |= ios_base::failbit;

      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    _M_extract_wday_or_month(iter_type __beg, iter_type __end, int& __member,
			     const _CharT** __names, size_t __indexlen,
			     ios_base& __io, ios_base::iostate& __err) const
    {
      typedef char_traits<_CharT>		__traits_type;
      const locale& __loc = __io._M_getloc();
      const ctype<_CharT>& __ctype = use_facet<ctype<_CharT> >(__loc);

      int* __matches = static_cast<int*>(__builtin_alloca(2 * sizeof(int)
							  * __indexlen));
      size_t __nmatches = 0;
      size_t* __matches_lengths = 0;
      size_t __pos = 0;

      if (__beg != __end)
	{
	  const char_type __c = *__beg;
	  for (size_t __i = 0; __i < 2 * __indexlen; ++__i)
	    if (__c == __names[__i][0]
		|| __c == __ctype.toupper(__names[__i][0]))
	      __matches[__nmatches++] = __i;
	}

      if (__nmatches)
	{
	  ++__beg;
	  ++__pos;

	  __matches_lengths
	    = static_cast<size_t*>(__builtin_alloca(sizeof(size_t)
						    * __nmatches));
	  for (size_t __i = 0; __i < __nmatches; ++__i)
	    __matches_lengths[__i]
	      = __traits_type::length(__names[__matches[__i]]);
	}

      for (; __beg != __end; ++__beg, (void)++__pos)
	{
	  size_t __nskipped = 0;
	  const char_type __c = *__beg;
	  for (size_t __i = 0; __i < __nmatches;)
	    {
	      const char_type* __name = __names[__matches[__i]];
	      if (__pos >= __matches_lengths[__i])
		++__nskipped, ++__i;
	      else if (!(__name[__pos] == __c))
		{
		  --__nmatches;
		  __matches[__i] = __matches[__nmatches];
		  __matches_lengths[__i] = __matches_lengths[__nmatches];
		}
	      else
		++__i;
	    }
	  if (__nskipped == __nmatches)
	    break;
	}

      if ((__nmatches == 1 && __matches_lengths[0] == __pos)
	  || (__nmatches == 2 && (__matches_lengths[0] == __pos
				  || __matches_lengths[1] == __pos)))
	__member = (__matches[0] >= __indexlen
		    ? __matches[0] - __indexlen : __matches[0]);
      else
	__err |= ios_base::failbit;

      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    do_get_time(iter_type __beg, iter_type __end, ios_base& __io,
		ios_base::iostate& __err, tm* __tm) const
    {
      const locale& __loc = __io._M_getloc();
      const __timepunct<_CharT>& __tp = use_facet<__timepunct<_CharT> >(__loc);
      const char_type*  __times[2];
      __tp._M_time_formats(__times);
      __beg = _M_extract_via_format(__beg, __end, __io, __err, 
				    __tm, __times[0]);
      if (__beg == __end)
	__err |= ios_base::eofbit;
      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    do_get_date(iter_type __beg, iter_type __end, ios_base& __io,
		ios_base::iostate& __err, tm* __tm) const
    {
      const locale& __loc = __io._M_getloc();
      const __timepunct<_CharT>& __tp = use_facet<__timepunct<_CharT> >(__loc);
      const char_type*  __dates[2];
      __tp._M_date_formats(__dates);
      __beg = _M_extract_via_format(__beg, __end, __io, __err, 
				    __tm, __dates[0]);
      if (__beg == __end)
	__err |= ios_base::eofbit;
      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    do_get_weekday(iter_type __beg, iter_type __end, ios_base& __io,
		   ios_base::iostate& __err, tm* __tm) const
    {
      const locale& __loc = __io._M_getloc();
      const __timepunct<_CharT>& __tp = use_facet<__timepunct<_CharT> >(__loc);
      const char_type* __days[14];
      __tp._M_days_abbreviated(__days);
      __tp._M_days(__days + 7);
      int __tmpwday;
      ios_base::iostate __tmperr = ios_base::goodbit;

      __beg = _M_extract_wday_or_month(__beg, __end, __tmpwday, __days, 7,
				       __io, __tmperr);
      if (!__tmperr)
	__tm->tm_wday = __tmpwday;
      else
	__err |= ios_base::failbit;

      if (__beg == __end)
	__err |= ios_base::eofbit;
      return __beg;
     }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    do_get_monthname(iter_type __beg, iter_type __end,
                     ios_base& __io, ios_base::iostate& __err, tm* __tm) const
    {
      const locale& __loc = __io._M_getloc();
      const __timepunct<_CharT>& __tp = use_facet<__timepunct<_CharT> >(__loc);
      const char_type*  __months[24];
      __tp._M_months_abbreviated(__months);
      __tp._M_months(__months + 12);
      int __tmpmon;
      ios_base::iostate __tmperr = ios_base::goodbit;

      __beg = _M_extract_wday_or_month(__beg, __end, __tmpmon, __months, 12,
				       __io, __tmperr);
      if (!__tmperr)
	__tm->tm_mon = __tmpmon;
      else
	__err |= ios_base::failbit;

      if (__beg == __end)
	__err |= ios_base::eofbit;
      return __beg;
    }

  template<typename _CharT, typename _InIter>
    _InIter
    time_get<_CharT, _InIter>::
    do_get_year(iter_type __beg, iter_type __end, ios_base& __io,
		ios_base::iostate& __err, tm* __tm) const
    {
      int __tmpyear;
      ios_base::iostate __tmperr = ios_base::goodbit;

      __beg = _M_extract_num(__beg, __end, __tmpyear, 0, 9999, 4,
			     __io, __tmperr);
      if (!__tmperr)
	__tm->tm_year = __tmpyear < 0 ? __tmpyear + 100 : __tmpyear - 1900;
      else
	__err |= ios_base::failbit;

      if (__beg == __end)
	__err |= ios_base::eofbit;
      return __beg;
    }

#if __cplusplus >= 201103L
  template<typename _CharT, typename _InIter>
    inline
    _InIter
    time_get<_CharT, _InIter>::
    get(iter_type __s, iter_type __end, ios_base& __io,
        ios_base::iostate& __err, tm* __tm, const char_type* __fmt,
        const char_type* __fmtend) const
    {
      const locale& __loc = __io._M_getloc();
      ctype<_CharT> const& __ctype = use_facet<ctype<_CharT> >(__loc);
      __err = ios_base::goodbit;
      while (__fmt != __fmtend &&
             __err == ios_base::goodbit)
        {
          if (__s == __end)
            {
              __err = ios_base::eofbit | ios_base::failbit;
              break;
            }
          else if (__ctype.narrow(*__fmt, 0) == '%')
            {
              char __format;
              char __mod = 0;
              if (++__fmt == __fmtend)
                {
                  __err = ios_base::failbit;
                  break;
                }
              const char __c = __ctype.narrow(*__fmt, 0);
              if (__c != 'E' && __c != 'O')
                __format = __c;
              else if (++__fmt != __fmtend)
                {
                  __mod = __c;
                  __format = __ctype.narrow(*__fmt, 0);
                }
              else
                {
                  __err = ios_base::failbit;
                  break;
                }
              __s = this->do_get(__s, __end, __io, __err, __tm, __format,
				 __mod);
              ++__fmt;
            }
          else if (__ctype.is(ctype_base::space, *__fmt))
            {
              ++__fmt;
              while (__fmt != __fmtend &&
                     __ctype.is(ctype_base::space, *__fmt))
                ++__fmt;

              while (__s != __end &&
                     __ctype.is(ctype_base::space, *__s))
                ++__s;
            }
          // TODO real case-insensitive comparison
          else if (__ctype.tolower(*__s) == __ctype.tolower(*__fmt) ||
                   __ctype.toupper(*__s) == __ctype.toupper(*__fmt))
            {
              ++__s;
              ++__fmt;
            }
          else
            {
              __err = ios_base::failbit;
              break;
            }
        }
      return __s;
    }

  template<typename _CharT, typename _InIter>
    inline
    _InIter
    time_get<_CharT, _InIter>::
    do_get(iter_type __beg, iter_type __end, ios_base& __io,
           ios_base::iostate& __err, tm* __tm,
           char __format, char __mod) const
    {
      const locale& __loc = __io._M_getloc();
      ctype<_CharT> const& __ctype = use_facet<ctype<_CharT> >(__loc);
      __err = ios_base::goodbit;

      char_type __fmt[4];
      __fmt[0] = __ctype.widen('%');
      if (!__mod)
        {
          __fmt[1] = __format;
          __fmt[2] = char_type();
        }
      else
        {
          __fmt[1] = __mod;
          __fmt[2] = __format;
          __fmt[3] = char_type();
        }

      __beg = _M_extract_via_format(__beg, __end, __io, __err, __tm, __fmt);
      if (__beg == __end)
	__err |= ios_base::eofbit;
      return __beg;
    }

#endif // __cplusplus >= 201103L

  template<typename _CharT, typename _OutIter>
    _OutIter
    time_put<_CharT, _OutIter>::
    put(iter_type __s, ios_base& __io, char_type __fill, const tm* __tm,
	const _CharT* __beg, const _CharT* __end) const
    {
      const locale& __loc = __io._M_getloc();
      ctype<_CharT> const& __ctype = use_facet<ctype<_CharT> >(__loc);
      for (; __beg != __end; ++__beg)
	if (__ctype.narrow(*__beg, 0) != '%')
	  {
	    *__s = *__beg;
	    ++__s;
	  }
	else if (++__beg != __end)
	  {
	    char __format;
	    char __mod = 0;
	    const char __c = __ctype.narrow(*__beg, 0);
	    if (__c != 'E' && __c != 'O')
	      __format = __c;
	    else if (++__beg != __end)
	      {
		__mod = __c;
		__format = __ctype.narrow(*__beg, 0);
	      }
	    else
	      break;
	    __s = this->do_put(__s, __io, __fill, __tm, __format, __mod);
	  }
	else
	  break;
      return __s;
    }

  template<typename _CharT, typename _OutIter>
    _OutIter
    time_put<_CharT, _OutIter>::
    do_put(iter_type __s, ios_base& __io, char_type, const tm* __tm,
	   char __format, char __mod) const
    {
      const locale& __loc = __io._M_getloc();
      ctype<_CharT> const& __ctype = use_facet<ctype<_CharT> >(__loc);
      __timepunct<_CharT> const& __tp = use_facet<__timepunct<_CharT> >(__loc);

      // NB: This size is arbitrary. Should this be a data member,
      // initialized at construction?
      const size_t __maxlen = 128;
      char_type __res[__maxlen];

      // NB: In IEE 1003.1-200x, and perhaps other locale models, it
      // is possible that the format character will be longer than one
      // character. Possibilities include 'E' or 'O' followed by a
      // format character: if __mod is not the default argument, assume
      // it's a valid modifier.
      char_type __fmt[4];
      __fmt[0] = __ctype.widen('%');
      if (!__mod)
	{
	  __fmt[1] = __format;
	  __fmt[2] = char_type();
	}
      else
	{
	  __fmt[1] = __mod;
	  __fmt[2] = __format;
	  __fmt[3] = char_type();
	}

      __tp._M_put(__res, __maxlen, __fmt, __tm);

      // Write resulting, fully-formatted string to output iterator.
      return std::__write(__s, __res, char_traits<char_type>::length(__res));
    }


  // Inhibit implicit instantiations for required instantiations,
  // which are defined via explicit instantiations elsewhere.
#if _GLIBCXX_EXTERN_TEMPLATE
  extern template class moneypunct<char, false>;
  extern template class moneypunct<char, true>;
  extern template class moneypunct_byname<char, false>;
  extern template class moneypunct_byname<char, true>;
  extern template class _GLIBCXX_NAMESPACE_LDBL_OR_CXX11 money_get<char>;
  extern template class _GLIBCXX_NAMESPACE_LDBL_OR_CXX11 money_put<char>;
  extern template class __timepunct<char>;
  extern template class time_put<char>;
  extern template class time_put_byname<char>;
  extern template class time_get<char>;
  extern template class time_get_byname<char>;
  extern template class messages<char>;
  extern template class messages_byname<char>;

  extern template
    const moneypunct<char, true>&
    use_facet<moneypunct<char, true> >(const locale&);

  extern template
    const moneypunct<char, false>&
    use_facet<moneypunct<char, false> >(const locale&);

  extern template
    const money_put<char>&
    use_facet<money_put<char> >(const locale&);

  extern template
    const money_get<char>&
    use_facet<money_get<char> >(const locale&);

  extern template
    const __timepunct<char>&
    use_facet<__timepunct<char> >(const locale&);

  extern template
    const time_put<char>&
    use_facet<time_put<char> >(const locale&);

  extern template
    const time_get<char>&
    use_facet<time_get<char> >(const locale&);

  extern template
    const messages<char>&
    use_facet<messages<char> >(const locale&);

  extern template
    bool
    has_facet<moneypunct<char> >(const locale&);

  extern template
    bool
    has_facet<money_put<char> >(const locale&);

  extern template
    bool
    has_facet<money_get<char> >(const locale&);

  extern template
    bool
    has_facet<__timepunct<char> >(const locale&);

  extern template
    bool
    has_facet<time_put<char> >(const locale&);

  extern template
    bool
    has_facet<time_get<char> >(const locale&);

  extern template
    bool
    has_facet<messages<char> >(const locale&);

#ifdef _GLIBCXX_USE_WCHAR_T
  extern template class moneypunct<wchar_t, false>;
  extern template class moneypunct<wchar_t, true>;
  extern template class moneypunct_byname<wchar_t, false>;
  extern template class moneypunct_byname<wchar_t, true>;
  extern template class _GLIBCXX_NAMESPACE_LDBL_OR_CXX11 money_get<wchar_t>;
  extern template class _GLIBCXX_NAMESPACE_LDBL_OR_CXX11 money_put<wchar_t>;
  extern template class __timepunct<wchar_t>;
  extern template class time_put<wchar_t>;
  extern template class time_put_byname<wchar_t>;
  extern template class time_get<wchar_t>;
  extern template class time_get_byname<wchar_t>;
  extern template class messages<wchar_t>;
  extern template class messages_byname<wchar_t>;

  extern template
    const moneypunct<wchar_t, true>&
    use_facet<moneypunct<wchar_t, true> >(const locale&);

  extern template
    const moneypunct<wchar_t, false>&
    use_facet<moneypunct<wchar_t, false> >(const locale&);

  extern template
    const money_put<wchar_t>&
    use_facet<money_put<wchar_t> >(const locale&);

  extern template
    const money_get<wchar_t>&
    use_facet<money_get<wchar_t> >(const locale&);

  extern template
    const __timepunct<wchar_t>&
    use_facet<__timepunct<wchar_t> >(const locale&);

  extern template
    const time_put<wchar_t>&
    use_facet<time_put<wchar_t> >(const locale&);

  extern template
    const time_get<wchar_t>&
    use_facet<time_get<wchar_t> >(const locale&);

  extern template
    const messages<wchar_t>&
    use_facet<messages<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<moneypunct<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<money_put<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<money_get<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<__timepunct<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<time_put<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<time_get<wchar_t> >(const locale&);

  extern template
    bool
    has_facet<messages<wchar_t> >(const locale&);
#endif
#endif

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace std

#endif
