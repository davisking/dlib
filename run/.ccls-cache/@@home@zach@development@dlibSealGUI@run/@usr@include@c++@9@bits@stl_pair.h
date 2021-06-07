// Pair implementation -*- C++ -*-

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

/*
 *
 * Copyright (c) 1994
 * Hewlett-Packard Company
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Hewlett-Packard Company makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 *
 *
 * Copyright (c) 1996,1997
 * Silicon Graphics Computer Systems, Inc.
 *
 * Permission to use, copy, modify, distribute and sell this software
 * and its documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  Silicon Graphics makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 */

/** @file bits/stl_pair.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{utility}
 */

#ifndef _STL_PAIR_H
#define _STL_PAIR_H 1

#include <bits/move.h> // for std::move / std::forward, and std::swap

#if __cplusplus >= 201103L
#include <type_traits> // for std::__decay_and_strip too
#endif

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  /**
   *  @addtogroup utilities
   *  @{
   */

#if __cplusplus >= 201103L
  /// piecewise_construct_t
  struct piecewise_construct_t { explicit piecewise_construct_t() = default; };

  /// piecewise_construct
  _GLIBCXX17_INLINE constexpr piecewise_construct_t piecewise_construct =
    piecewise_construct_t();

  // Forward declarations.
  template<typename...>
    class tuple;

  template<std::size_t...>
    struct _Index_tuple;

  // Concept utility functions, reused in conditionally-explicit
  // constructors.
  // See PR 70437, don't look at is_constructible or
  // is_convertible if the types are the same to
  // avoid querying those properties for incomplete types.
  template <bool, typename _T1, typename _T2>
    struct _PCC
    {
      template <typename _U1, typename _U2>
      static constexpr bool _ConstructiblePair()
      {
	return __and_<is_constructible<_T1, const _U1&>,
		      is_constructible<_T2, const _U2&>>::value;
      }

      template <typename _U1, typename _U2>
      static constexpr bool _ImplicitlyConvertiblePair()
      {
	return __and_<is_convertible<const _U1&, _T1>,
		      is_convertible<const _U2&, _T2>>::value;
      }

      template <typename _U1, typename _U2>
      static constexpr bool _MoveConstructiblePair()
      {
	return __and_<is_constructible<_T1, _U1&&>,
		      is_constructible<_T2, _U2&&>>::value;
      }

      template <typename _U1, typename _U2>
      static constexpr bool _ImplicitlyMoveConvertiblePair()
      {
	return __and_<is_convertible<_U1&&, _T1>,
		      is_convertible<_U2&&, _T2>>::value;
      }

      template <bool __implicit, typename _U1, typename _U2>
      static constexpr bool _CopyMovePair()
      {
	using __do_converts = __and_<is_convertible<const _U1&, _T1>,
				  is_convertible<_U2&&, _T2>>;
	using __converts = typename conditional<__implicit,
				       __do_converts,
				       __not_<__do_converts>>::type;
	return __and_<is_constructible<_T1, const _U1&>,
		      is_constructible<_T2, _U2&&>,
		      __converts
		      >::value;
      }

      template <bool __implicit, typename _U1, typename _U2>
      static constexpr bool _MoveCopyPair()
      {
	using __do_converts = __and_<is_convertible<_U1&&, _T1>,
				  is_convertible<const _U2&, _T2>>;
	using __converts = typename conditional<__implicit,
				       __do_converts,
				       __not_<__do_converts>>::type;
	return __and_<is_constructible<_T1, _U1&&>,
		      is_constructible<_T2, const _U2&&>,
		      __converts
		      >::value;
      }
  };

  template <typename _T1, typename _T2>
    struct _PCC<false, _T1, _T2>
    {
      template <typename _U1, typename _U2>
      static constexpr bool _ConstructiblePair()
      {
	return false;
      }

      template <typename _U1, typename _U2>
      static constexpr bool _ImplicitlyConvertiblePair()
      {
	return false;
      }

      template <typename _U1, typename _U2>
      static constexpr bool _MoveConstructiblePair()
      {
	return false;
      }

      template <typename _U1, typename _U2>
      static constexpr bool _ImplicitlyMoveConvertiblePair()
      {
	return false;
      }
  };

  // PR libstdc++/79141, a utility type for preventing
  // initialization of an argument of a disabled assignment
  // operator from a pair of empty braces.
  struct __nonesuch_no_braces : std::__nonesuch {
    explicit __nonesuch_no_braces(const __nonesuch&) = delete;
  };
#endif // C++11

  template<typename _U1, typename _U2> class __pair_base
  {
#if __cplusplus >= 201103L
    template<typename _T1, typename _T2> friend struct pair;
    __pair_base() = default;
    ~__pair_base() = default;
    __pair_base(const __pair_base&) = default;
    __pair_base& operator=(const __pair_base&) = delete;
#endif // C++11
  };

 /**
   *  @brief Struct holding two objects of arbitrary type.
   *
   *  @tparam _T1  Type of first object.
   *  @tparam _T2  Type of second object.
   */
  template<typename _T1, typename _T2>
    struct pair
    : private __pair_base<_T1, _T2>
    {
      typedef _T1 first_type;    /// @c first_type is the first bound type
      typedef _T2 second_type;   /// @c second_type is the second bound type

      _T1 first;                 /// @c first is a copy of the first object
      _T2 second;                /// @c second is a copy of the second object

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 265.  std::pair::pair() effects overly restrictive
      /** The default constructor creates @c first and @c second using their
       *  respective default constructors.  */
#if __cplusplus >= 201103L
      template <typename _U1 = _T1,
                typename _U2 = _T2,
                typename enable_if<__and_<
                                     __is_implicitly_default_constructible<_U1>,
                                     __is_implicitly_default_constructible<_U2>>
                                   ::value, bool>::type = true>
#endif
      _GLIBCXX_CONSTEXPR pair()
      : first(), second() { }

#if __cplusplus >= 201103L
      template <typename _U1 = _T1,
                typename _U2 = _T2,
                typename enable_if<__and_<
                       is_default_constructible<_U1>,
                       is_default_constructible<_U2>,
                       __not_<
                         __and_<__is_implicitly_default_constructible<_U1>,
                                __is_implicitly_default_constructible<_U2>>>>
                                   ::value, bool>::type = false>
      explicit constexpr pair()
      : first(), second() { }
#endif

      /** Two objects may be passed to a @c pair constructor to be copied.  */
#if __cplusplus < 201103L
      pair(const _T1& __a, const _T2& __b)
      : first(__a), second(__b) { }
#else
      // Shortcut for constraining the templates that don't take pairs.
      using _PCCP = _PCC<true, _T1, _T2>;

      template<typename _U1 = _T1, typename _U2=_T2, typename
	       enable_if<_PCCP::template
			   _ConstructiblePair<_U1, _U2>()
	                 && _PCCP::template
			   _ImplicitlyConvertiblePair<_U1, _U2>(),
                         bool>::type=true>
      constexpr pair(const _T1& __a, const _T2& __b)
      : first(__a), second(__b) { }

       template<typename _U1 = _T1, typename _U2=_T2, typename
		enable_if<_PCCP::template
			    _ConstructiblePair<_U1, _U2>()
	                  && !_PCCP::template
			    _ImplicitlyConvertiblePair<_U1, _U2>(),
                         bool>::type=false>
      explicit constexpr pair(const _T1& __a, const _T2& __b)
      : first(__a), second(__b) { }
#endif

      /** There is also a templated copy ctor for the @c pair class itself.  */
#if __cplusplus < 201103L
      template<typename _U1, typename _U2>
	pair(const pair<_U1, _U2>& __p)
	: first(__p.first), second(__p.second) { }
#else
      // Shortcut for constraining the templates that take pairs.
      template <typename _U1, typename _U2>
        using _PCCFP = _PCC<!is_same<_T1, _U1>::value
			    || !is_same<_T2, _U2>::value,
			    _T1, _T2>;

      template<typename _U1, typename _U2, typename
	       enable_if<_PCCFP<_U1, _U2>::template
			   _ConstructiblePair<_U1, _U2>()
	                 && _PCCFP<_U1, _U2>::template
			   _ImplicitlyConvertiblePair<_U1, _U2>(),
			  bool>::type=true>
        constexpr pair(const pair<_U1, _U2>& __p)
        : first(__p.first), second(__p.second) { }

      template<typename _U1, typename _U2, typename
	       enable_if<_PCCFP<_U1, _U2>::template
			   _ConstructiblePair<_U1, _U2>()
			 && !_PCCFP<_U1, _U2>::template
			   _ImplicitlyConvertiblePair<_U1, _U2>(),
                         bool>::type=false>
	explicit constexpr pair(const pair<_U1, _U2>& __p)
	: first(__p.first), second(__p.second) { }

      constexpr pair(const pair&) = default;
      constexpr pair(pair&&) = default;

      // DR 811.
      template<typename _U1, typename
	       enable_if<_PCCP::template
			   _MoveCopyPair<true, _U1, _T2>(),
                         bool>::type=true>
       constexpr pair(_U1&& __x, const _T2& __y)
       : first(std::forward<_U1>(__x)), second(__y) { }

      template<typename _U1, typename
	       enable_if<_PCCP::template
			   _MoveCopyPair<false, _U1, _T2>(),
                         bool>::type=false>
       explicit constexpr pair(_U1&& __x, const _T2& __y)
       : first(std::forward<_U1>(__x)), second(__y) { }

      template<typename _U2, typename
	       enable_if<_PCCP::template
			   _CopyMovePair<true, _T1, _U2>(),
                         bool>::type=true>
       constexpr pair(const _T1& __x, _U2&& __y)
       : first(__x), second(std::forward<_U2>(__y)) { }

      template<typename _U2, typename
	       enable_if<_PCCP::template
			   _CopyMovePair<false, _T1, _U2>(),
                         bool>::type=false>
       explicit pair(const _T1& __x, _U2&& __y)
       : first(__x), second(std::forward<_U2>(__y)) { }

      template<typename _U1, typename _U2, typename
	       enable_if<_PCCP::template
			   _MoveConstructiblePair<_U1, _U2>()
			  && _PCCP::template
			   _ImplicitlyMoveConvertiblePair<_U1, _U2>(),
                         bool>::type=true>
	constexpr pair(_U1&& __x, _U2&& __y)
	: first(std::forward<_U1>(__x)), second(std::forward<_U2>(__y)) { }

      template<typename _U1, typename _U2, typename
	       enable_if<_PCCP::template
			   _MoveConstructiblePair<_U1, _U2>()
			  && !_PCCP::template
			   _ImplicitlyMoveConvertiblePair<_U1, _U2>(),
                         bool>::type=false>
	explicit constexpr pair(_U1&& __x, _U2&& __y)
	: first(std::forward<_U1>(__x)), second(std::forward<_U2>(__y)) { }


      template<typename _U1, typename _U2, typename
	       enable_if<_PCCFP<_U1, _U2>::template
			   _MoveConstructiblePair<_U1, _U2>()
			  && _PCCFP<_U1, _U2>::template
			   _ImplicitlyMoveConvertiblePair<_U1, _U2>(),
                         bool>::type=true>
	constexpr pair(pair<_U1, _U2>&& __p)
	: first(std::forward<_U1>(__p.first)),
	  second(std::forward<_U2>(__p.second)) { }

      template<typename _U1, typename _U2, typename
	       enable_if<_PCCFP<_U1, _U2>::template
			   _MoveConstructiblePair<_U1, _U2>()
			  && !_PCCFP<_U1, _U2>::template
			   _ImplicitlyMoveConvertiblePair<_U1, _U2>(),
                         bool>::type=false>
	explicit constexpr pair(pair<_U1, _U2>&& __p)
	: first(std::forward<_U1>(__p.first)),
	  second(std::forward<_U2>(__p.second)) { }

      template<typename... _Args1, typename... _Args2>
        pair(piecewise_construct_t, tuple<_Args1...>, tuple<_Args2...>);

      pair&
      operator=(typename conditional<
		__and_<is_copy_assignable<_T1>,
		       is_copy_assignable<_T2>>::value,
		const pair&, const __nonesuch_no_braces&>::type __p)
      {
	first = __p.first;
	second = __p.second;
	return *this;
      }

      pair&
      operator=(typename conditional<
		__and_<is_move_assignable<_T1>,
		       is_move_assignable<_T2>>::value,
		pair&&, __nonesuch_no_braces&&>::type __p)
      noexcept(__and_<is_nothrow_move_assignable<_T1>,
		      is_nothrow_move_assignable<_T2>>::value)
      {
	first = std::forward<first_type>(__p.first);
	second = std::forward<second_type>(__p.second);
	return *this;
      }

      template<typename _U1, typename _U2>
      typename enable_if<__and_<is_assignable<_T1&, const _U1&>,
				is_assignable<_T2&, const _U2&>>::value,
			 pair&>::type
	operator=(const pair<_U1, _U2>& __p)
	{
	  first = __p.first;
	  second = __p.second;
	  return *this;
	}

      template<typename _U1, typename _U2>
      typename enable_if<__and_<is_assignable<_T1&, _U1&&>,
				is_assignable<_T2&, _U2&&>>::value,
			 pair&>::type
	operator=(pair<_U1, _U2>&& __p)
	{
	  first = std::forward<_U1>(__p.first);
	  second = std::forward<_U2>(__p.second);
	  return *this;
	}

      void
      swap(pair& __p)
      noexcept(__and_<__is_nothrow_swappable<_T1>,
                      __is_nothrow_swappable<_T2>>::value)
      {
	using std::swap;
	swap(first, __p.first);
	swap(second, __p.second);
      }

    private:
      template<typename... _Args1, std::size_t... _Indexes1,
               typename... _Args2, std::size_t... _Indexes2>
        pair(tuple<_Args1...>&, tuple<_Args2...>&,
             _Index_tuple<_Indexes1...>, _Index_tuple<_Indexes2...>);
#endif
    };

#if __cpp_deduction_guides >= 201606
  template<typename _T1, typename _T2> pair(_T1, _T2) -> pair<_T1, _T2>;
#endif

  /// Two pairs of the same type are equal iff their members are equal.
  template<typename _T1, typename _T2>
    inline _GLIBCXX_CONSTEXPR bool
    operator==(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    { return __x.first == __y.first && __x.second == __y.second; }

  /// <http://gcc.gnu.org/onlinedocs/libstdc++/manual/utilities.html>
  template<typename _T1, typename _T2>
    inline _GLIBCXX_CONSTEXPR bool
    operator<(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    { return __x.first < __y.first
	     || (!(__y.first < __x.first) && __x.second < __y.second); }

  /// Uses @c operator== to find the result.
  template<typename _T1, typename _T2>
    inline _GLIBCXX_CONSTEXPR bool
    operator!=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    { return !(__x == __y); }

  /// Uses @c operator< to find the result.
  template<typename _T1, typename _T2>
    inline _GLIBCXX_CONSTEXPR bool
    operator>(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    { return __y < __x; }

  /// Uses @c operator< to find the result.
  template<typename _T1, typename _T2>
    inline _GLIBCXX_CONSTEXPR bool
    operator<=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    { return !(__y < __x); }

  /// Uses @c operator< to find the result.
  template<typename _T1, typename _T2>
    inline _GLIBCXX_CONSTEXPR bool
    operator>=(const pair<_T1, _T2>& __x, const pair<_T1, _T2>& __y)
    { return !(__x < __y); }

#if __cplusplus >= 201103L
  /// See std::pair::swap().
  // Note:  no std::swap overloads in C++03 mode, this has performance
  //        implications, see, eg, libstdc++/38466.
  template<typename _T1, typename _T2>
    inline
#if __cplusplus > 201402L || !defined(__STRICT_ANSI__) // c++1z or gnu++11
    // Constrained free swap overload, see p0185r1
    typename enable_if<__and_<__is_swappable<_T1>,
                              __is_swappable<_T2>>::value>::type
#else
    void
#endif
    swap(pair<_T1, _T2>& __x, pair<_T1, _T2>& __y)
    noexcept(noexcept(__x.swap(__y)))
    { __x.swap(__y); }

#if __cplusplus > 201402L || !defined(__STRICT_ANSI__) // c++1z or gnu++11
  template<typename _T1, typename _T2>
    typename enable_if<!__and_<__is_swappable<_T1>,
			       __is_swappable<_T2>>::value>::type
    swap(pair<_T1, _T2>&, pair<_T1, _T2>&) = delete;
#endif
#endif // __cplusplus >= 201103L

  /**
   *  @brief A convenience wrapper for creating a pair from two objects.
   *  @param  __x  The first object.
   *  @param  __y  The second object.
   *  @return   A newly-constructed pair<> object of the appropriate type.
   *
   *  The standard requires that the objects be passed by reference-to-const,
   *  but LWG issue #181 says they should be passed by const value.  We follow
   *  the LWG by default.
   */
  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // 181.  make_pair() unintended behavior
#if __cplusplus >= 201103L
  // NB: DR 706.
  template<typename _T1, typename _T2>
    constexpr pair<typename __decay_and_strip<_T1>::__type,
                   typename __decay_and_strip<_T2>::__type>
    make_pair(_T1&& __x, _T2&& __y)
    {
      typedef typename __decay_and_strip<_T1>::__type __ds_type1;
      typedef typename __decay_and_strip<_T2>::__type __ds_type2;
      typedef pair<__ds_type1, __ds_type2> 	      __pair_type;
      return __pair_type(std::forward<_T1>(__x), std::forward<_T2>(__y));
    }
#else
  template<typename _T1, typename _T2>
    inline pair<_T1, _T2>
    make_pair(_T1 __x, _T2 __y)
    { return pair<_T1, _T2>(__x, __y); }
#endif

  /// @}

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace std

#endif /* _STL_PAIR_H */
