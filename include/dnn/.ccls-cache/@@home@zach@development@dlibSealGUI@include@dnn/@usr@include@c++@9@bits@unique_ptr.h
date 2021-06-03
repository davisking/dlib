// unique_ptr implementation -*- C++ -*-

// Copyright (C) 2008-2019 Free Software Foundation, Inc.
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

/** @file bits/unique_ptr.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{memory}
 */

#ifndef _UNIQUE_PTR_H
#define _UNIQUE_PTR_H 1

#include <bits/c++config.h>
#include <debug/assertions.h>
#include <type_traits>
#include <utility>
#include <tuple>
#include <bits/stl_function.h>
#include <bits/functional_hash.h>

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  /**
   * @addtogroup pointer_abstractions
   * @{
   */

#if _GLIBCXX_USE_DEPRECATED
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  template<typename> class auto_ptr;
#pragma GCC diagnostic pop
#endif

  /// Primary template of default_delete, used by unique_ptr
  template<typename _Tp>
    struct default_delete
    {
      /// Default constructor
      constexpr default_delete() noexcept = default;

      /** @brief Converting constructor.
       *
       * Allows conversion from a deleter for arrays of another type, @p _Up,
       * only if @p _Up* is convertible to @p _Tp*.
       */
      template<typename _Up, typename = typename
	       enable_if<is_convertible<_Up*, _Tp*>::value>::type>
        default_delete(const default_delete<_Up>&) noexcept { }

      /// Calls @c delete @p __ptr
      void
      operator()(_Tp* __ptr) const
      {
	static_assert(!is_void<_Tp>::value,
		      "can't delete pointer to incomplete type");
	static_assert(sizeof(_Tp)>0,
		      "can't delete pointer to incomplete type");
	delete __ptr;
      }
    };

  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // DR 740 - omit specialization for array objects with a compile time length
  /// Specialization for arrays, default_delete.
  template<typename _Tp>
    struct default_delete<_Tp[]>
    {
    public:
      /// Default constructor
      constexpr default_delete() noexcept = default;

      /** @brief Converting constructor.
       *
       * Allows conversion from a deleter for arrays of another type, such as
       * a const-qualified version of @p _Tp.
       *
       * Conversions from types derived from @c _Tp are not allowed because
       * it is unsafe to @c delete[] an array of derived types through a
       * pointer to the base type.
       */
      template<typename _Up, typename = typename
	       enable_if<is_convertible<_Up(*)[], _Tp(*)[]>::value>::type>
        default_delete(const default_delete<_Up[]>&) noexcept { }

      /// Calls @c delete[] @p __ptr
      template<typename _Up>
      typename enable_if<is_convertible<_Up(*)[], _Tp(*)[]>::value>::type
	operator()(_Up* __ptr) const
      {
	static_assert(sizeof(_Tp)>0,
		      "can't delete pointer to incomplete type");
	delete [] __ptr;
      }
    };

  template <typename _Tp, typename _Dp>
    class __uniq_ptr_impl
    {
      template <typename _Up, typename _Ep, typename = void>
	struct _Ptr
	{
	  using type = _Up*;
	};

      template <typename _Up, typename _Ep>
	struct
	_Ptr<_Up, _Ep, __void_t<typename remove_reference<_Ep>::type::pointer>>
	{
	  using type = typename remove_reference<_Ep>::type::pointer;
	};

    public:
      using _DeleterConstraint = enable_if<
        __and_<__not_<is_pointer<_Dp>>,
	       is_default_constructible<_Dp>>::value>;

      using pointer = typename _Ptr<_Tp, _Dp>::type;

      static_assert( !is_rvalue_reference<_Dp>::value,
		     "unique_ptr's deleter type must be a function object type"
		     " or an lvalue reference type" );

      __uniq_ptr_impl() = default;
      __uniq_ptr_impl(pointer __p) : _M_t() { _M_ptr() = __p; }

      template<typename _Del>
      __uniq_ptr_impl(pointer __p, _Del&& __d)
	: _M_t(__p, std::forward<_Del>(__d)) { }

      pointer&   _M_ptr() { return std::get<0>(_M_t); }
      pointer    _M_ptr() const { return std::get<0>(_M_t); }
      _Dp&       _M_deleter() { return std::get<1>(_M_t); }
      const _Dp& _M_deleter() const { return std::get<1>(_M_t); }

      void
      swap(__uniq_ptr_impl& __rhs) noexcept
      {
	using std::swap;
	swap(this->_M_ptr(), __rhs._M_ptr());
	swap(this->_M_deleter(), __rhs._M_deleter());
      }

    private:
      tuple<pointer, _Dp> _M_t;
    };

  /// 20.7.1.2 unique_ptr for single objects.
  template <typename _Tp, typename _Dp = default_delete<_Tp>>
    class unique_ptr
    {
      template <typename _Up>
	using _DeleterConstraint =
	  typename __uniq_ptr_impl<_Tp, _Up>::_DeleterConstraint::type;

      __uniq_ptr_impl<_Tp, _Dp> _M_t;

    public:
      using pointer	  = typename __uniq_ptr_impl<_Tp, _Dp>::pointer;
      using element_type  = _Tp;
      using deleter_type  = _Dp;

    private:
      // helper template for detecting a safe conversion from another
      // unique_ptr
      template<typename _Up, typename _Ep>
	using __safe_conversion_up = __and_<
	  is_convertible<typename unique_ptr<_Up, _Ep>::pointer, pointer>,
	  __not_<is_array<_Up>>
        >;

    public:
      // Constructors.

      /// Default constructor, creates a unique_ptr that owns nothing.
      template<typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
	constexpr unique_ptr() noexcept
	: _M_t()
	{ }

      /** Takes ownership of a pointer.
       *
       * @param __p  A pointer to an object of @c element_type
       *
       * The deleter will be value-initialized.
       */
      template<typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
	explicit
	unique_ptr(pointer __p) noexcept
	: _M_t(__p)
        { }

      /** Takes ownership of a pointer.
       *
       * @param __p  A pointer to an object of @c element_type
       * @param __d  A reference to a deleter.
       *
       * The deleter will be initialized with @p __d
       */
      template<typename _Del = deleter_type,
	       typename = _Require<is_copy_constructible<_Del>>>
	unique_ptr(pointer __p, const deleter_type& __d) noexcept
	: _M_t(__p, __d) { }

      /** Takes ownership of a pointer.
       *
       * @param __p  A pointer to an object of @c element_type
       * @param __d  An rvalue reference to a (non-reference) deleter.
       *
       * The deleter will be initialized with @p std::move(__d)
       */
      template<typename _Del = deleter_type,
	       typename = _Require<is_move_constructible<_Del>>>
	unique_ptr(pointer __p,
		   __enable_if_t<!is_lvalue_reference<_Del>::value,
				 _Del&&> __d) noexcept
	: _M_t(__p, std::move(__d))
	{ }

      template<typename _Del = deleter_type,
	       typename _DelUnref = typename remove_reference<_Del>::type>
	unique_ptr(pointer,
		   __enable_if_t<is_lvalue_reference<_Del>::value,
				 _DelUnref&&>) = delete;

      /// Creates a unique_ptr that owns nothing.
      template<typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
	constexpr unique_ptr(nullptr_t) noexcept
	: _M_t()
	{ }

      // Move constructors.

      /// Move constructor.
      unique_ptr(unique_ptr&& __u) noexcept
      : _M_t(__u.release(), std::forward<deleter_type>(__u.get_deleter())) { }

      /** @brief Converting constructor from another type
       *
       * Requires that the pointer owned by @p __u is convertible to the
       * type of pointer owned by this object, @p __u does not own an array,
       * and @p __u has a compatible deleter type.
       */
      template<typename _Up, typename _Ep, typename = _Require<
               __safe_conversion_up<_Up, _Ep>,
	       typename conditional<is_reference<_Dp>::value,
				    is_same<_Ep, _Dp>,
				    is_convertible<_Ep, _Dp>>::type>>
	unique_ptr(unique_ptr<_Up, _Ep>&& __u) noexcept
	: _M_t(__u.release(), std::forward<_Ep>(__u.get_deleter()))
	{ }

#if _GLIBCXX_USE_DEPRECATED
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      /// Converting constructor from @c auto_ptr
      template<typename _Up, typename = _Require<
	       is_convertible<_Up*, _Tp*>, is_same<_Dp, default_delete<_Tp>>>>
	unique_ptr(auto_ptr<_Up>&& __u) noexcept;
#pragma GCC diagnostic pop
#endif

      /// Destructor, invokes the deleter if the stored pointer is not null.
      ~unique_ptr() noexcept
      {
	static_assert(__is_invocable<deleter_type&, pointer>::value,
		      "unique_ptr's deleter must be invocable with a pointer");
	auto& __ptr = _M_t._M_ptr();
	if (__ptr != nullptr)
	  get_deleter()(std::move(__ptr));
	__ptr = pointer();
      }

      // Assignment.

      /** @brief Move assignment operator.
       *
       * @param __u  The object to transfer ownership from.
       *
       * Invokes the deleter first if this object owns a pointer.
       */
      unique_ptr&
      operator=(unique_ptr&& __u) noexcept
      {
	reset(__u.release());
	get_deleter() = std::forward<deleter_type>(__u.get_deleter());
	return *this;
      }

      /** @brief Assignment from another type.
       *
       * @param __u  The object to transfer ownership from, which owns a
       *             convertible pointer to a non-array object.
       *
       * Invokes the deleter first if this object owns a pointer.
       */
      template<typename _Up, typename _Ep>
        typename enable_if< __and_<
          __safe_conversion_up<_Up, _Ep>,
          is_assignable<deleter_type&, _Ep&&>
          >::value,
          unique_ptr&>::type
	operator=(unique_ptr<_Up, _Ep>&& __u) noexcept
	{
	  reset(__u.release());
	  get_deleter() = std::forward<_Ep>(__u.get_deleter());
	  return *this;
	}

      /// Reset the %unique_ptr to empty, invoking the deleter if necessary.
      unique_ptr&
      operator=(nullptr_t) noexcept
      {
	reset();
	return *this;
      }

      // Observers.

      /// Dereference the stored pointer.
      typename add_lvalue_reference<element_type>::type
      operator*() const
      {
	__glibcxx_assert(get() != pointer());
	return *get();
      }

      /// Return the stored pointer.
      pointer
      operator->() const noexcept
      {
	_GLIBCXX_DEBUG_PEDASSERT(get() != pointer());
	return get();
      }

      /// Return the stored pointer.
      pointer
      get() const noexcept
      { return _M_t._M_ptr(); }

      /// Return a reference to the stored deleter.
      deleter_type&
      get_deleter() noexcept
      { return _M_t._M_deleter(); }

      /// Return a reference to the stored deleter.
      const deleter_type&
      get_deleter() const noexcept
      { return _M_t._M_deleter(); }

      /// Return @c true if the stored pointer is not null.
      explicit operator bool() const noexcept
      { return get() == pointer() ? false : true; }

      // Modifiers.

      /// Release ownership of any stored pointer.
      pointer
      release() noexcept
      {
	pointer __p = get();
	_M_t._M_ptr() = pointer();
	return __p;
      }

      /** @brief Replace the stored pointer.
       *
       * @param __p  The new pointer to store.
       *
       * The deleter will be invoked if a pointer is already owned.
       */
      void
      reset(pointer __p = pointer()) noexcept
      {
	static_assert(__is_invocable<deleter_type&, pointer>::value,
		      "unique_ptr's deleter must be invocable with a pointer");
	using std::swap;
	swap(_M_t._M_ptr(), __p);
	if (__p != pointer())
	  get_deleter()(std::move(__p));
      }

      /// Exchange the pointer and deleter with another object.
      void
      swap(unique_ptr& __u) noexcept
      {
	static_assert(__is_swappable<_Dp>::value, "deleter must be swappable");
	_M_t.swap(__u._M_t);
      }

      // Disable copy from lvalue.
      unique_ptr(const unique_ptr&) = delete;
      unique_ptr& operator=(const unique_ptr&) = delete;
  };

  /// 20.7.1.3 unique_ptr for array objects with a runtime length
  // [unique.ptr.runtime]
  // _GLIBCXX_RESOLVE_LIB_DEFECTS
  // DR 740 - omit specialization for array objects with a compile time length
  template<typename _Tp, typename _Dp>
    class unique_ptr<_Tp[], _Dp>
    {
      template <typename _Up>
      using _DeleterConstraint =
	typename __uniq_ptr_impl<_Tp, _Up>::_DeleterConstraint::type;

      __uniq_ptr_impl<_Tp, _Dp> _M_t;

      template<typename _Up>
	using __remove_cv = typename remove_cv<_Up>::type;

      // like is_base_of<_Tp, _Up> but false if unqualified types are the same
      template<typename _Up>
	using __is_derived_Tp
	  = __and_< is_base_of<_Tp, _Up>,
		    __not_<is_same<__remove_cv<_Tp>, __remove_cv<_Up>>> >;

    public:
      using pointer	  = typename __uniq_ptr_impl<_Tp, _Dp>::pointer;
      using element_type  = _Tp;
      using deleter_type  = _Dp;

      // helper template for detecting a safe conversion from another
      // unique_ptr
      template<typename _Up, typename _Ep,
               typename _UPtr = unique_ptr<_Up, _Ep>,
	       typename _UP_pointer = typename _UPtr::pointer,
	       typename _UP_element_type = typename _UPtr::element_type>
	using __safe_conversion_up = __and_<
          is_array<_Up>,
          is_same<pointer, element_type*>,
          is_same<_UP_pointer, _UP_element_type*>,
          is_convertible<_UP_element_type(*)[], element_type(*)[]>
        >;

      // helper template for detecting a safe conversion from a raw pointer
      template<typename _Up>
        using __safe_conversion_raw = __and_<
          __or_<__or_<is_same<_Up, pointer>,
                      is_same<_Up, nullptr_t>>,
                __and_<is_pointer<_Up>,
                       is_same<pointer, element_type*>,
                       is_convertible<
                         typename remove_pointer<_Up>::type(*)[],
                         element_type(*)[]>
                >
          >
        >;

      // Constructors.

      /// Default constructor, creates a unique_ptr that owns nothing.
      template<typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
	constexpr unique_ptr() noexcept
	: _M_t()
	{ }

      /** Takes ownership of a pointer.
       *
       * @param __p  A pointer to an array of a type safely convertible
       * to an array of @c element_type
       *
       * The deleter will be value-initialized.
       */
      template<typename _Up,
	       typename _Vp = _Dp,
	       typename = _DeleterConstraint<_Vp>,
	       typename = typename enable_if<
                 __safe_conversion_raw<_Up>::value, bool>::type>
	explicit
	unique_ptr(_Up __p) noexcept
	: _M_t(__p)
        { }

      /** Takes ownership of a pointer.
       *
       * @param __p  A pointer to an array of a type safely convertible
       * to an array of @c element_type
       * @param __d  A reference to a deleter.
       *
       * The deleter will be initialized with @p __d
       */
      template<typename _Up, typename _Del = deleter_type,
	       typename = _Require<__safe_conversion_raw<_Up>,
				   is_copy_constructible<_Del>>>
      unique_ptr(_Up __p, const deleter_type& __d) noexcept
      : _M_t(__p, __d) { }

      /** Takes ownership of a pointer.
       *
       * @param __p  A pointer to an array of a type safely convertible
       * to an array of @c element_type
       * @param __d  A reference to a deleter.
       *
       * The deleter will be initialized with @p std::move(__d)
       */
      template<typename _Up, typename _Del = deleter_type,
	       typename = _Require<__safe_conversion_raw<_Up>,
				   is_move_constructible<_Del>>>
	unique_ptr(_Up __p,
		   __enable_if_t<!is_lvalue_reference<_Del>::value,
				 _Del&&> __d) noexcept
	: _M_t(std::move(__p), std::move(__d))
	{ }

      template<typename _Up, typename _Del = deleter_type,
	       typename _DelUnref = typename remove_reference<_Del>::type,
	       typename = _Require<__safe_conversion_raw<_Up>>>
	unique_ptr(_Up,
		   __enable_if_t<is_lvalue_reference<_Del>::value,
				 _DelUnref&&>) = delete;

      /// Move constructor.
      unique_ptr(unique_ptr&& __u) noexcept
      : _M_t(__u.release(), std::forward<deleter_type>(__u.get_deleter())) { }

      /// Creates a unique_ptr that owns nothing.
      template<typename _Del = _Dp, typename = _DeleterConstraint<_Del>>
	constexpr unique_ptr(nullptr_t) noexcept
	: _M_t()
        { }

      template<typename _Up, typename _Ep, typename = _Require<
	       __safe_conversion_up<_Up, _Ep>,
	       typename conditional<is_reference<_Dp>::value,
				    is_same<_Ep, _Dp>,
				    is_convertible<_Ep, _Dp>>::type>>
	unique_ptr(unique_ptr<_Up, _Ep>&& __u) noexcept
	: _M_t(__u.release(), std::forward<_Ep>(__u.get_deleter()))
	{ }

      /// Destructor, invokes the deleter if the stored pointer is not null.
      ~unique_ptr()
      {
	auto& __ptr = _M_t._M_ptr();
	if (__ptr != nullptr)
	  get_deleter()(__ptr);
	__ptr = pointer();
      }

      // Assignment.

      /** @brief Move assignment operator.
       *
       * @param __u  The object to transfer ownership from.
       *
       * Invokes the deleter first if this object owns a pointer.
       */
      unique_ptr&
      operator=(unique_ptr&& __u) noexcept
      {
	reset(__u.release());
	get_deleter() = std::forward<deleter_type>(__u.get_deleter());
	return *this;
      }

      /** @brief Assignment from another type.
       *
       * @param __u  The object to transfer ownership from, which owns a
       *             convertible pointer to an array object.
       *
       * Invokes the deleter first if this object owns a pointer.
       */
      template<typename _Up, typename _Ep>
	typename
	enable_if<__and_<__safe_conversion_up<_Up, _Ep>,
                         is_assignable<deleter_type&, _Ep&&>
                  >::value,
                  unique_ptr&>::type
	operator=(unique_ptr<_Up, _Ep>&& __u) noexcept
	{
	  reset(__u.release());
	  get_deleter() = std::forward<_Ep>(__u.get_deleter());
	  return *this;
	}

      /// Reset the %unique_ptr to empty, invoking the deleter if necessary.
      unique_ptr&
      operator=(nullptr_t) noexcept
      {
	reset();
	return *this;
      }

      // Observers.

      /// Access an element of owned array.
      typename std::add_lvalue_reference<element_type>::type
      operator[](size_t __i) const
      {
	__glibcxx_assert(get() != pointer());
	return get()[__i];
      }

      /// Return the stored pointer.
      pointer
      get() const noexcept
      { return _M_t._M_ptr(); }

      /// Return a reference to the stored deleter.
      deleter_type&
      get_deleter() noexcept
      { return _M_t._M_deleter(); }

      /// Return a reference to the stored deleter.
      const deleter_type&
      get_deleter() const noexcept
      { return _M_t._M_deleter(); }

      /// Return @c true if the stored pointer is not null.
      explicit operator bool() const noexcept
      { return get() == pointer() ? false : true; }

      // Modifiers.

      /// Release ownership of any stored pointer.
      pointer
      release() noexcept
      {
	pointer __p = get();
	_M_t._M_ptr() = pointer();
	return __p;
      }

      /** @brief Replace the stored pointer.
       *
       * @param __p  The new pointer to store.
       *
       * The deleter will be invoked if a pointer is already owned.
       */
      template <typename _Up,
                typename = _Require<
                  __or_<is_same<_Up, pointer>,
                        __and_<is_same<pointer, element_type*>,
                               is_pointer<_Up>,
                               is_convertible<
                                 typename remove_pointer<_Up>::type(*)[],
                                 element_type(*)[]
                               >
                        >
                  >
               >>
      void
      reset(_Up __p) noexcept
      {
	pointer __ptr = __p;
	using std::swap;
	swap(_M_t._M_ptr(), __ptr);
	if (__ptr != nullptr)
	  get_deleter()(__ptr);
      }

      void reset(nullptr_t = nullptr) noexcept
      {
        reset(pointer());
      }

      /// Exchange the pointer and deleter with another object.
      void
      swap(unique_ptr& __u) noexcept
      {
	static_assert(__is_swappable<_Dp>::value, "deleter must be swappable");
	_M_t.swap(__u._M_t);
      }

      // Disable copy from lvalue.
      unique_ptr(const unique_ptr&) = delete;
      unique_ptr& operator=(const unique_ptr&) = delete;
    };

  template<typename _Tp, typename _Dp>
    inline
#if __cplusplus > 201402L || !defined(__STRICT_ANSI__) // c++1z or gnu++11
    // Constrained free swap overload, see p0185r1
    typename enable_if<__is_swappable<_Dp>::value>::type
#else
    void
#endif
    swap(unique_ptr<_Tp, _Dp>& __x,
	 unique_ptr<_Tp, _Dp>& __y) noexcept
    { __x.swap(__y); }

#if __cplusplus > 201402L || !defined(__STRICT_ANSI__) // c++1z or gnu++11
  template<typename _Tp, typename _Dp>
    typename enable_if<!__is_swappable<_Dp>::value>::type
    swap(unique_ptr<_Tp, _Dp>&,
	 unique_ptr<_Tp, _Dp>&) = delete;
#endif

  template<typename _Tp, typename _Dp,
	   typename _Up, typename _Ep>
    _GLIBCXX_NODISCARD inline bool
    operator==(const unique_ptr<_Tp, _Dp>& __x,
	       const unique_ptr<_Up, _Ep>& __y)
    { return __x.get() == __y.get(); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator==(const unique_ptr<_Tp, _Dp>& __x, nullptr_t) noexcept
    { return !__x; }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator==(nullptr_t, const unique_ptr<_Tp, _Dp>& __x) noexcept
    { return !__x; }

  template<typename _Tp, typename _Dp,
	   typename _Up, typename _Ep>
    _GLIBCXX_NODISCARD inline bool
    operator!=(const unique_ptr<_Tp, _Dp>& __x,
	       const unique_ptr<_Up, _Ep>& __y)
    { return __x.get() != __y.get(); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator!=(const unique_ptr<_Tp, _Dp>& __x, nullptr_t) noexcept
    { return (bool)__x; }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator!=(nullptr_t, const unique_ptr<_Tp, _Dp>& __x) noexcept
    { return (bool)__x; }

  template<typename _Tp, typename _Dp,
	   typename _Up, typename _Ep>
    _GLIBCXX_NODISCARD inline bool
    operator<(const unique_ptr<_Tp, _Dp>& __x,
	      const unique_ptr<_Up, _Ep>& __y)
    {
      typedef typename
	std::common_type<typename unique_ptr<_Tp, _Dp>::pointer,
	                 typename unique_ptr<_Up, _Ep>::pointer>::type _CT;
      return std::less<_CT>()(__x.get(), __y.get());
    }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator<(const unique_ptr<_Tp, _Dp>& __x, nullptr_t)
    { return std::less<typename unique_ptr<_Tp, _Dp>::pointer>()(__x.get(),
								 nullptr); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator<(nullptr_t, const unique_ptr<_Tp, _Dp>& __x)
    { return std::less<typename unique_ptr<_Tp, _Dp>::pointer>()(nullptr,
								 __x.get()); }

  template<typename _Tp, typename _Dp,
	   typename _Up, typename _Ep>
    _GLIBCXX_NODISCARD inline bool
    operator<=(const unique_ptr<_Tp, _Dp>& __x,
	       const unique_ptr<_Up, _Ep>& __y)
    { return !(__y < __x); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator<=(const unique_ptr<_Tp, _Dp>& __x, nullptr_t)
    { return !(nullptr < __x); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator<=(nullptr_t, const unique_ptr<_Tp, _Dp>& __x)
    { return !(__x < nullptr); }

  template<typename _Tp, typename _Dp,
	   typename _Up, typename _Ep>
    _GLIBCXX_NODISCARD inline bool
    operator>(const unique_ptr<_Tp, _Dp>& __x,
	      const unique_ptr<_Up, _Ep>& __y)
    { return (__y < __x); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator>(const unique_ptr<_Tp, _Dp>& __x, nullptr_t)
    { return std::less<typename unique_ptr<_Tp, _Dp>::pointer>()(nullptr,
								 __x.get()); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator>(nullptr_t, const unique_ptr<_Tp, _Dp>& __x)
    { return std::less<typename unique_ptr<_Tp, _Dp>::pointer>()(__x.get(),
								 nullptr); }

  template<typename _Tp, typename _Dp,
	   typename _Up, typename _Ep>
    _GLIBCXX_NODISCARD inline bool
    operator>=(const unique_ptr<_Tp, _Dp>& __x,
	       const unique_ptr<_Up, _Ep>& __y)
    { return !(__x < __y); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator>=(const unique_ptr<_Tp, _Dp>& __x, nullptr_t)
    { return !(__x < nullptr); }

  template<typename _Tp, typename _Dp>
    _GLIBCXX_NODISCARD inline bool
    operator>=(nullptr_t, const unique_ptr<_Tp, _Dp>& __x)
    { return !(nullptr < __x); }

  /// std::hash specialization for unique_ptr.
  template<typename _Tp, typename _Dp>
    struct hash<unique_ptr<_Tp, _Dp>>
    : public __hash_base<size_t, unique_ptr<_Tp, _Dp>>,
    private __poison_hash<typename unique_ptr<_Tp, _Dp>::pointer>
    {
      size_t
      operator()(const unique_ptr<_Tp, _Dp>& __u) const noexcept
      {
	typedef unique_ptr<_Tp, _Dp> _UP;
	return std::hash<typename _UP::pointer>()(__u.get());
      }
    };

#if __cplusplus > 201103L

#define __cpp_lib_make_unique 201304

  template<typename _Tp>
    struct _MakeUniq
    { typedef unique_ptr<_Tp> __single_object; };

  template<typename _Tp>
    struct _MakeUniq<_Tp[]>
    { typedef unique_ptr<_Tp[]> __array; };

  template<typename _Tp, size_t _Bound>
    struct _MakeUniq<_Tp[_Bound]>
    { struct __invalid_type { }; };

  /// std::make_unique for single objects
  template<typename _Tp, typename... _Args>
    inline typename _MakeUniq<_Tp>::__single_object
    make_unique(_Args&&... __args)
    { return unique_ptr<_Tp>(new _Tp(std::forward<_Args>(__args)...)); }

  /// std::make_unique for arrays of unknown bound
  template<typename _Tp>
    inline typename _MakeUniq<_Tp>::__array
    make_unique(size_t __num)
    { return unique_ptr<_Tp>(new remove_extent_t<_Tp>[__num]()); }

  /// Disable std::make_unique for arrays of known bound
  template<typename _Tp, typename... _Args>
    inline typename _MakeUniq<_Tp>::__invalid_type
    make_unique(_Args&&...) = delete;
#endif

  // @} group pointer_abstractions

#if __cplusplus >= 201703L
  namespace __detail::__variant
  {
    template<typename> struct _Never_valueless_alt; // see <variant>

    // Provide the strong exception-safety guarantee when emplacing a
    // unique_ptr into a variant.
    template<typename _Tp, typename _Del>
      struct _Never_valueless_alt<std::unique_ptr<_Tp, _Del>>
      : std::true_type
      { };
  }  // namespace __detail::__variant
#endif // C++17

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace

#endif /* _UNIQUE_PTR_H */
