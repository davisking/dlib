// Allocators -*- C++ -*-

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
 * Copyright (c) 1996-1997
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

/** @file bits/allocator.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{memory}
 */

#ifndef _ALLOCATOR_H
#define _ALLOCATOR_H 1

#include <bits/c++allocator.h> // Define the base class to std::allocator.
#include <bits/memoryfwd.h>
#if __cplusplus >= 201103L
#include <type_traits>
#endif

#define __cpp_lib_incomplete_container_elements 201505
#if __cplusplus >= 201103L
# define __cpp_lib_allocator_is_always_equal 201411
#endif

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  /**
   *  @addtogroup allocators
   *  @{
   */

  /// allocator<void> specialization.
  template<>
    class allocator<void>
    {
    public:
      typedef size_t      size_type;
      typedef ptrdiff_t   difference_type;
      typedef void*       pointer;
      typedef const void* const_pointer;
      typedef void        value_type;

      template<typename _Tp1>
	struct rebind
	{ typedef allocator<_Tp1> other; };

#if __cplusplus >= 201103L
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 2103. std::allocator propagate_on_container_move_assignment
      typedef true_type propagate_on_container_move_assignment;

      typedef true_type is_always_equal;

      template<typename _Up, typename... _Args>
	void
	construct(_Up* __p, _Args&&... __args)
	noexcept(noexcept(::new((void *)__p)
			    _Up(std::forward<_Args>(__args)...)))
	{ ::new((void *)__p) _Up(std::forward<_Args>(__args)...); }

      template<typename _Up>
	void
	destroy(_Up* __p)
	noexcept(noexcept(__p->~_Up()))
	{ __p->~_Up(); }
#endif
    };

  /**
   * @brief  The @a standard allocator, as per [20.4].
   *
   *  See https://gcc.gnu.org/onlinedocs/libstdc++/manual/memory.html#std.util.memory.allocator
   *  for further details.
   *
   *  @tparam  _Tp  Type of allocated object.
   */
  template<typename _Tp>
    class allocator : public __allocator_base<_Tp>
    {
   public:
      typedef size_t     size_type;
      typedef ptrdiff_t  difference_type;
      typedef _Tp*       pointer;
      typedef const _Tp* const_pointer;
      typedef _Tp&       reference;
      typedef const _Tp& const_reference;
      typedef _Tp        value_type;

      template<typename _Tp1>
	struct rebind
	{ typedef allocator<_Tp1> other; };

#if __cplusplus >= 201103L
      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 2103. std::allocator propagate_on_container_move_assignment
      typedef true_type propagate_on_container_move_assignment;

      typedef true_type is_always_equal;
#endif

      // _GLIBCXX_RESOLVE_LIB_DEFECTS
      // 3035. std::allocator's constructors should be constexpr
      _GLIBCXX20_CONSTEXPR
      allocator() _GLIBCXX_NOTHROW { }

      _GLIBCXX20_CONSTEXPR
      allocator(const allocator& __a) _GLIBCXX_NOTHROW
      : __allocator_base<_Tp>(__a) { }

#if __cplusplus >= 201103L
      // Avoid implicit deprecation.
      allocator& operator=(const allocator&) = default;
#endif

      template<typename _Tp1>
	_GLIBCXX20_CONSTEXPR
	allocator(const allocator<_Tp1>&) _GLIBCXX_NOTHROW { }

      ~allocator() _GLIBCXX_NOTHROW { }

      friend bool
      operator==(const allocator&, const allocator&) _GLIBCXX_NOTHROW
      { return true; }

      friend bool
      operator!=(const allocator&, const allocator&) _GLIBCXX_NOTHROW
      { return false; }

      // Inherit everything else.
    };

  template<typename _T1, typename _T2>
    inline bool
    operator==(const allocator<_T1>&, const allocator<_T2>&)
    _GLIBCXX_NOTHROW
    { return true; }

  template<typename _T1, typename _T2>
    inline bool
    operator!=(const allocator<_T1>&, const allocator<_T2>&)
    _GLIBCXX_NOTHROW
    { return false; }

  // Invalid allocator<cv T> partial specializations.
  // allocator_traits::rebind_alloc can be used to form a valid allocator type.
  template<typename _Tp>
    class allocator<const _Tp>
    {
    public:
      typedef _Tp value_type;
      template<typename _Up> allocator(const allocator<_Up>&) { }
    };

  template<typename _Tp>
    class allocator<volatile _Tp>
    {
    public:
      typedef _Tp value_type;
      template<typename _Up> allocator(const allocator<_Up>&) { }
    };

  template<typename _Tp>
    class allocator<const volatile _Tp>
    {
    public:
      typedef _Tp value_type;
      template<typename _Up> allocator(const allocator<_Up>&) { }
    };

  /// @} group allocator

  // Inhibit implicit instantiations for required instantiations,
  // which are defined via explicit instantiations elsewhere.
#if _GLIBCXX_EXTERN_TEMPLATE
  extern template class allocator<char>;
  extern template class allocator<wchar_t>;
#endif

  // Undefine.
#undef __allocator_base

  // To implement Option 3 of DR 431.
  template<typename _Alloc, bool = __is_empty(_Alloc)>
    struct __alloc_swap
    { static void _S_do_it(_Alloc&, _Alloc&) _GLIBCXX_NOEXCEPT { } };

  template<typename _Alloc>
    struct __alloc_swap<_Alloc, false>
    {
      static void
      _S_do_it(_Alloc& __one, _Alloc& __two) _GLIBCXX_NOEXCEPT
      {
	// Precondition: swappable allocators.
	if (__one != __two)
	  swap(__one, __two);
      }
    };

  // Optimize for stateless allocators.
  template<typename _Alloc, bool = __is_empty(_Alloc)>
    struct __alloc_neq
    {
      static bool
      _S_do_it(const _Alloc&, const _Alloc&)
      { return false; }
    };

  template<typename _Alloc>
    struct __alloc_neq<_Alloc, false>
    {
      static bool
      _S_do_it(const _Alloc& __one, const _Alloc& __two)
      { return __one != __two; }
    };

#if __cplusplus >= 201103L
  template<typename _Tp, bool
    = __or_<is_copy_constructible<typename _Tp::value_type>,
            is_nothrow_move_constructible<typename _Tp::value_type>>::value>
    struct __shrink_to_fit_aux
    { static bool _S_do_it(_Tp&) noexcept { return false; } };

  template<typename _Tp>
    struct __shrink_to_fit_aux<_Tp, true>
    {
      static bool
      _S_do_it(_Tp& __c) noexcept
      {
#if __cpp_exceptions
	try
	  {
	    _Tp(__make_move_if_noexcept_iterator(__c.begin()),
		__make_move_if_noexcept_iterator(__c.end()),
		__c.get_allocator()).swap(__c);
	    return true;
	  }
	catch(...)
	  { return false; }
#else
	return false;
#endif
      }
    };
#endif

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace std

#endif
