// -*- C++ -*- header.

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

/** @file bits/atomic_base.h
 *  This is an internal header file, included by other library headers.
 *  Do not attempt to use it directly. @headername{atomic}
 */

#ifndef _GLIBCXX_ATOMIC_BASE_H
#define _GLIBCXX_ATOMIC_BASE_H 1

#pragma GCC system_header

#include <bits/c++config.h>
#include <stdint.h>
#include <bits/atomic_lockfree_defines.h>

#ifndef _GLIBCXX_ALWAYS_INLINE
#define _GLIBCXX_ALWAYS_INLINE inline __attribute__((__always_inline__))
#endif

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  /**
   * @defgroup atomics Atomics
   *
   * Components for performing atomic operations.
   * @{
   */

  /// Enumeration for memory_order
#if __cplusplus > 201703L
  enum class memory_order : int
    {
      relaxed,
      consume,
      acquire,
      release,
      acq_rel,
      seq_cst
    };

  inline constexpr memory_order memory_order_relaxed = memory_order::relaxed;
  inline constexpr memory_order memory_order_consume = memory_order::consume;
  inline constexpr memory_order memory_order_acquire = memory_order::acquire;
  inline constexpr memory_order memory_order_release = memory_order::release;
  inline constexpr memory_order memory_order_acq_rel = memory_order::acq_rel;
  inline constexpr memory_order memory_order_seq_cst = memory_order::seq_cst;
#else
  typedef enum memory_order
    {
      memory_order_relaxed,
      memory_order_consume,
      memory_order_acquire,
      memory_order_release,
      memory_order_acq_rel,
      memory_order_seq_cst
    } memory_order;
#endif

  enum __memory_order_modifier
    {
      __memory_order_mask          = 0x0ffff,
      __memory_order_modifier_mask = 0xffff0000,
      __memory_order_hle_acquire   = 0x10000,
      __memory_order_hle_release   = 0x20000
    };

  constexpr memory_order
  operator|(memory_order __m, __memory_order_modifier __mod)
  {
    return memory_order(int(__m) | int(__mod));
  }

  constexpr memory_order
  operator&(memory_order __m, __memory_order_modifier __mod)
  {
    return memory_order(int(__m) & int(__mod));
  }

  // Drop release ordering as per [atomics.types.operations.req]/21
  constexpr memory_order
  __cmpexch_failure_order2(memory_order __m) noexcept
  {
    return __m == memory_order_acq_rel ? memory_order_acquire
      : __m == memory_order_release ? memory_order_relaxed : __m;
  }

  constexpr memory_order
  __cmpexch_failure_order(memory_order __m) noexcept
  {
    return memory_order(__cmpexch_failure_order2(__m & __memory_order_mask)
      | __memory_order_modifier(__m & __memory_order_modifier_mask));
  }

  _GLIBCXX_ALWAYS_INLINE void
  atomic_thread_fence(memory_order __m) noexcept
  { __atomic_thread_fence(int(__m)); }

  _GLIBCXX_ALWAYS_INLINE void
  atomic_signal_fence(memory_order __m) noexcept
  { __atomic_signal_fence(int(__m)); }

  /// kill_dependency
  template<typename _Tp>
    inline _Tp
    kill_dependency(_Tp __y) noexcept
    {
      _Tp __ret(__y);
      return __ret;
    }


  // Base types for atomics.
  template<typename _IntTp>
    struct __atomic_base;


#define ATOMIC_VAR_INIT(_VI) { _VI }

  template<typename _Tp>
    struct atomic;

  template<typename _Tp>
    struct atomic<_Tp*>;

    /* The target's "set" value for test-and-set may not be exactly 1.  */
#if __GCC_ATOMIC_TEST_AND_SET_TRUEVAL == 1
    typedef bool __atomic_flag_data_type;
#else
    typedef unsigned char __atomic_flag_data_type;
#endif

  /**
   *  @brief Base type for atomic_flag.
   *
   *  Base type is POD with data, allowing atomic_flag to derive from
   *  it and meet the standard layout type requirement. In addition to
   *  compatibility with a C interface, this allows different
   *  implementations of atomic_flag to use the same atomic operation
   *  functions, via a standard conversion to the __atomic_flag_base
   *  argument.
  */
  _GLIBCXX_BEGIN_EXTERN_C

  struct __atomic_flag_base
  {
    __atomic_flag_data_type _M_i;
  };

  _GLIBCXX_END_EXTERN_C

#define ATOMIC_FLAG_INIT { 0 }

  /// atomic_flag
  struct atomic_flag : public __atomic_flag_base
  {
    atomic_flag() noexcept = default;
    ~atomic_flag() noexcept = default;
    atomic_flag(const atomic_flag&) = delete;
    atomic_flag& operator=(const atomic_flag&) = delete;
    atomic_flag& operator=(const atomic_flag&) volatile = delete;

    // Conversion to ATOMIC_FLAG_INIT.
    constexpr atomic_flag(bool __i) noexcept
      : __atomic_flag_base{ _S_init(__i) }
    { }

    _GLIBCXX_ALWAYS_INLINE bool
    test_and_set(memory_order __m = memory_order_seq_cst) noexcept
    {
      return __atomic_test_and_set (&_M_i, int(__m));
    }

    _GLIBCXX_ALWAYS_INLINE bool
    test_and_set(memory_order __m = memory_order_seq_cst) volatile noexcept
    {
      return __atomic_test_and_set (&_M_i, int(__m));
    }

    _GLIBCXX_ALWAYS_INLINE void
    clear(memory_order __m = memory_order_seq_cst) noexcept
    {
      memory_order __b = __m & __memory_order_mask;
      __glibcxx_assert(__b != memory_order_consume);
      __glibcxx_assert(__b != memory_order_acquire);
      __glibcxx_assert(__b != memory_order_acq_rel);

      __atomic_clear (&_M_i, int(__m));
    }

    _GLIBCXX_ALWAYS_INLINE void
    clear(memory_order __m = memory_order_seq_cst) volatile noexcept
    {
      memory_order __b = __m & __memory_order_mask;
      __glibcxx_assert(__b != memory_order_consume);
      __glibcxx_assert(__b != memory_order_acquire);
      __glibcxx_assert(__b != memory_order_acq_rel);

      __atomic_clear (&_M_i, int(__m));
    }

  private:
    static constexpr __atomic_flag_data_type
    _S_init(bool __i)
    { return __i ? __GCC_ATOMIC_TEST_AND_SET_TRUEVAL : 0; }
  };


  /// Base class for atomic integrals.
  //
  // For each of the integral types, define atomic_[integral type] struct
  //
  // atomic_bool     bool
  // atomic_char     char
  // atomic_schar    signed char
  // atomic_uchar    unsigned char
  // atomic_short    short
  // atomic_ushort   unsigned short
  // atomic_int      int
  // atomic_uint     unsigned int
  // atomic_long     long
  // atomic_ulong    unsigned long
  // atomic_llong    long long
  // atomic_ullong   unsigned long long
  // atomic_char8_t  char8_t
  // atomic_char16_t char16_t
  // atomic_char32_t char32_t
  // atomic_wchar_t  wchar_t
  //
  // NB: Assuming _ITp is an integral scalar type that is 1, 2, 4, or
  // 8 bytes, since that is what GCC built-in functions for atomic
  // memory access expect.
  template<typename _ITp>
    struct __atomic_base
    {
      using value_type = _ITp;
      using difference_type = value_type;

    private:
      typedef _ITp 	__int_type;

      static constexpr int _S_alignment =
	sizeof(_ITp) > alignof(_ITp) ? sizeof(_ITp) : alignof(_ITp);

      alignas(_S_alignment) __int_type _M_i;

    public:
      __atomic_base() noexcept = default;
      ~__atomic_base() noexcept = default;
      __atomic_base(const __atomic_base&) = delete;
      __atomic_base& operator=(const __atomic_base&) = delete;
      __atomic_base& operator=(const __atomic_base&) volatile = delete;

      // Requires __int_type convertible to _M_i.
      constexpr __atomic_base(__int_type __i) noexcept : _M_i (__i) { }

      operator __int_type() const noexcept
      { return load(); }

      operator __int_type() const volatile noexcept
      { return load(); }

      __int_type
      operator=(__int_type __i) noexcept
      {
	store(__i);
	return __i;
      }

      __int_type
      operator=(__int_type __i) volatile noexcept
      {
	store(__i);
	return __i;
      }

      __int_type
      operator++(int) noexcept
      { return fetch_add(1); }

      __int_type
      operator++(int) volatile noexcept
      { return fetch_add(1); }

      __int_type
      operator--(int) noexcept
      { return fetch_sub(1); }

      __int_type
      operator--(int) volatile noexcept
      { return fetch_sub(1); }

      __int_type
      operator++() noexcept
      { return __atomic_add_fetch(&_M_i, 1, int(memory_order_seq_cst)); }

      __int_type
      operator++() volatile noexcept
      { return __atomic_add_fetch(&_M_i, 1, int(memory_order_seq_cst)); }

      __int_type
      operator--() noexcept
      { return __atomic_sub_fetch(&_M_i, 1, int(memory_order_seq_cst)); }

      __int_type
      operator--() volatile noexcept
      { return __atomic_sub_fetch(&_M_i, 1, int(memory_order_seq_cst)); }

      __int_type
      operator+=(__int_type __i) noexcept
      { return __atomic_add_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator+=(__int_type __i) volatile noexcept
      { return __atomic_add_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator-=(__int_type __i) noexcept
      { return __atomic_sub_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator-=(__int_type __i) volatile noexcept
      { return __atomic_sub_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator&=(__int_type __i) noexcept
      { return __atomic_and_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator&=(__int_type __i) volatile noexcept
      { return __atomic_and_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator|=(__int_type __i) noexcept
      { return __atomic_or_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator|=(__int_type __i) volatile noexcept
      { return __atomic_or_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator^=(__int_type __i) noexcept
      { return __atomic_xor_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      __int_type
      operator^=(__int_type __i) volatile noexcept
      { return __atomic_xor_fetch(&_M_i, __i, int(memory_order_seq_cst)); }

      bool
      is_lock_free() const noexcept
      {
	// Use a fake, minimally aligned pointer.
	return __atomic_is_lock_free(sizeof(_M_i),
	    reinterpret_cast<void *>(-_S_alignment));
      }

      bool
      is_lock_free() const volatile noexcept
      {
	// Use a fake, minimally aligned pointer.
	return __atomic_is_lock_free(sizeof(_M_i),
	    reinterpret_cast<void *>(-_S_alignment));
      }

      _GLIBCXX_ALWAYS_INLINE void
      store(__int_type __i, memory_order __m = memory_order_seq_cst) noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_acquire);
	__glibcxx_assert(__b != memory_order_acq_rel);
	__glibcxx_assert(__b != memory_order_consume);

	__atomic_store_n(&_M_i, __i, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE void
      store(__int_type __i,
	    memory_order __m = memory_order_seq_cst) volatile noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_acquire);
	__glibcxx_assert(__b != memory_order_acq_rel);
	__glibcxx_assert(__b != memory_order_consume);

	__atomic_store_n(&_M_i, __i, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __int_type
      load(memory_order __m = memory_order_seq_cst) const noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_release);
	__glibcxx_assert(__b != memory_order_acq_rel);

	return __atomic_load_n(&_M_i, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __int_type
      load(memory_order __m = memory_order_seq_cst) const volatile noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_release);
	__glibcxx_assert(__b != memory_order_acq_rel);

	return __atomic_load_n(&_M_i, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __int_type
      exchange(__int_type __i,
	       memory_order __m = memory_order_seq_cst) noexcept
      {
	return __atomic_exchange_n(&_M_i, __i, int(__m));
      }


      _GLIBCXX_ALWAYS_INLINE __int_type
      exchange(__int_type __i,
	       memory_order __m = memory_order_seq_cst) volatile noexcept
      {
	return __atomic_exchange_n(&_M_i, __i, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_weak(__int_type& __i1, __int_type __i2,
			    memory_order __m1, memory_order __m2) noexcept
      {
	memory_order __b2 = __m2 & __memory_order_mask;
	memory_order __b1 = __m1 & __memory_order_mask;
	__glibcxx_assert(__b2 != memory_order_release);
	__glibcxx_assert(__b2 != memory_order_acq_rel);
	__glibcxx_assert(__b2 <= __b1);

	return __atomic_compare_exchange_n(&_M_i, &__i1, __i2, 1,
					   int(__m1), int(__m2));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_weak(__int_type& __i1, __int_type __i2,
			    memory_order __m1,
			    memory_order __m2) volatile noexcept
      {
	memory_order __b2 = __m2 & __memory_order_mask;
	memory_order __b1 = __m1 & __memory_order_mask;
	__glibcxx_assert(__b2 != memory_order_release);
	__glibcxx_assert(__b2 != memory_order_acq_rel);
	__glibcxx_assert(__b2 <= __b1);

	return __atomic_compare_exchange_n(&_M_i, &__i1, __i2, 1,
					   int(__m1), int(__m2));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_weak(__int_type& __i1, __int_type __i2,
			    memory_order __m = memory_order_seq_cst) noexcept
      {
	return compare_exchange_weak(__i1, __i2, __m,
				     __cmpexch_failure_order(__m));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_weak(__int_type& __i1, __int_type __i2,
		   memory_order __m = memory_order_seq_cst) volatile noexcept
      {
	return compare_exchange_weak(__i1, __i2, __m,
				     __cmpexch_failure_order(__m));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_strong(__int_type& __i1, __int_type __i2,
			      memory_order __m1, memory_order __m2) noexcept
      {
	memory_order __b2 = __m2 & __memory_order_mask;
	memory_order __b1 = __m1 & __memory_order_mask;
	__glibcxx_assert(__b2 != memory_order_release);
	__glibcxx_assert(__b2 != memory_order_acq_rel);
	__glibcxx_assert(__b2 <= __b1);

	return __atomic_compare_exchange_n(&_M_i, &__i1, __i2, 0,
					   int(__m1), int(__m2));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_strong(__int_type& __i1, __int_type __i2,
			      memory_order __m1,
			      memory_order __m2) volatile noexcept
      {
	memory_order __b2 = __m2 & __memory_order_mask;
	memory_order __b1 = __m1 & __memory_order_mask;

	__glibcxx_assert(__b2 != memory_order_release);
	__glibcxx_assert(__b2 != memory_order_acq_rel);
	__glibcxx_assert(__b2 <= __b1);

	return __atomic_compare_exchange_n(&_M_i, &__i1, __i2, 0,
					   int(__m1), int(__m2));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_strong(__int_type& __i1, __int_type __i2,
			      memory_order __m = memory_order_seq_cst) noexcept
      {
	return compare_exchange_strong(__i1, __i2, __m,
				       __cmpexch_failure_order(__m));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_strong(__int_type& __i1, __int_type __i2,
		 memory_order __m = memory_order_seq_cst) volatile noexcept
      {
	return compare_exchange_strong(__i1, __i2, __m,
				       __cmpexch_failure_order(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_add(__int_type __i,
		memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_add(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_add(__int_type __i,
		memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_add(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_sub(__int_type __i,
		memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_sub(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_sub(__int_type __i,
		memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_sub(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_and(__int_type __i,
		memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_and(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_and(__int_type __i,
		memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_and(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_or(__int_type __i,
	       memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_or(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_or(__int_type __i,
	       memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_or(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_xor(__int_type __i,
		memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_xor(&_M_i, __i, int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __int_type
      fetch_xor(__int_type __i,
		memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_xor(&_M_i, __i, int(__m)); }
    };


  /// Partial specialization for pointer types.
  template<typename _PTp>
    struct __atomic_base<_PTp*>
    {
    private:
      typedef _PTp* 	__pointer_type;

      __pointer_type 	_M_p;

      // Factored out to facilitate explicit specialization.
      constexpr ptrdiff_t
      _M_type_size(ptrdiff_t __d) const { return __d * sizeof(_PTp); }

      constexpr ptrdiff_t
      _M_type_size(ptrdiff_t __d) const volatile { return __d * sizeof(_PTp); }

    public:
      __atomic_base() noexcept = default;
      ~__atomic_base() noexcept = default;
      __atomic_base(const __atomic_base&) = delete;
      __atomic_base& operator=(const __atomic_base&) = delete;
      __atomic_base& operator=(const __atomic_base&) volatile = delete;

      // Requires __pointer_type convertible to _M_p.
      constexpr __atomic_base(__pointer_type __p) noexcept : _M_p (__p) { }

      operator __pointer_type() const noexcept
      { return load(); }

      operator __pointer_type() const volatile noexcept
      { return load(); }

      __pointer_type
      operator=(__pointer_type __p) noexcept
      {
	store(__p);
	return __p;
      }

      __pointer_type
      operator=(__pointer_type __p) volatile noexcept
      {
	store(__p);
	return __p;
      }

      __pointer_type
      operator++(int) noexcept
      { return fetch_add(1); }

      __pointer_type
      operator++(int) volatile noexcept
      { return fetch_add(1); }

      __pointer_type
      operator--(int) noexcept
      { return fetch_sub(1); }

      __pointer_type
      operator--(int) volatile noexcept
      { return fetch_sub(1); }

      __pointer_type
      operator++() noexcept
      { return __atomic_add_fetch(&_M_p, _M_type_size(1),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator++() volatile noexcept
      { return __atomic_add_fetch(&_M_p, _M_type_size(1),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator--() noexcept
      { return __atomic_sub_fetch(&_M_p, _M_type_size(1),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator--() volatile noexcept
      { return __atomic_sub_fetch(&_M_p, _M_type_size(1),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator+=(ptrdiff_t __d) noexcept
      { return __atomic_add_fetch(&_M_p, _M_type_size(__d),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator+=(ptrdiff_t __d) volatile noexcept
      { return __atomic_add_fetch(&_M_p, _M_type_size(__d),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator-=(ptrdiff_t __d) noexcept
      { return __atomic_sub_fetch(&_M_p, _M_type_size(__d),
				  int(memory_order_seq_cst)); }

      __pointer_type
      operator-=(ptrdiff_t __d) volatile noexcept
      { return __atomic_sub_fetch(&_M_p, _M_type_size(__d),
				  int(memory_order_seq_cst)); }

      bool
      is_lock_free() const noexcept
      {
	// Produce a fake, minimally aligned pointer.
	return __atomic_is_lock_free(sizeof(_M_p),
	    reinterpret_cast<void *>(-__alignof(_M_p)));
      }

      bool
      is_lock_free() const volatile noexcept
      {
	// Produce a fake, minimally aligned pointer.
	return __atomic_is_lock_free(sizeof(_M_p),
	    reinterpret_cast<void *>(-__alignof(_M_p)));
      }

      _GLIBCXX_ALWAYS_INLINE void
      store(__pointer_type __p,
	    memory_order __m = memory_order_seq_cst) noexcept
      {
        memory_order __b = __m & __memory_order_mask;

	__glibcxx_assert(__b != memory_order_acquire);
	__glibcxx_assert(__b != memory_order_acq_rel);
	__glibcxx_assert(__b != memory_order_consume);

	__atomic_store_n(&_M_p, __p, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE void
      store(__pointer_type __p,
	    memory_order __m = memory_order_seq_cst) volatile noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_acquire);
	__glibcxx_assert(__b != memory_order_acq_rel);
	__glibcxx_assert(__b != memory_order_consume);

	__atomic_store_n(&_M_p, __p, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      load(memory_order __m = memory_order_seq_cst) const noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_release);
	__glibcxx_assert(__b != memory_order_acq_rel);

	return __atomic_load_n(&_M_p, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      load(memory_order __m = memory_order_seq_cst) const volatile noexcept
      {
	memory_order __b = __m & __memory_order_mask;
	__glibcxx_assert(__b != memory_order_release);
	__glibcxx_assert(__b != memory_order_acq_rel);

	return __atomic_load_n(&_M_p, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      exchange(__pointer_type __p,
	       memory_order __m = memory_order_seq_cst) noexcept
      {
	return __atomic_exchange_n(&_M_p, __p, int(__m));
      }


      _GLIBCXX_ALWAYS_INLINE __pointer_type
      exchange(__pointer_type __p,
	       memory_order __m = memory_order_seq_cst) volatile noexcept
      {
	return __atomic_exchange_n(&_M_p, __p, int(__m));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_strong(__pointer_type& __p1, __pointer_type __p2,
			      memory_order __m1,
			      memory_order __m2) noexcept
      {
	memory_order __b2 = __m2 & __memory_order_mask;
	memory_order __b1 = __m1 & __memory_order_mask;
	__glibcxx_assert(__b2 != memory_order_release);
	__glibcxx_assert(__b2 != memory_order_acq_rel);
	__glibcxx_assert(__b2 <= __b1);

	return __atomic_compare_exchange_n(&_M_p, &__p1, __p2, 0,
					   int(__m1), int(__m2));
      }

      _GLIBCXX_ALWAYS_INLINE bool
      compare_exchange_strong(__pointer_type& __p1, __pointer_type __p2,
			      memory_order __m1,
			      memory_order __m2) volatile noexcept
      {
	memory_order __b2 = __m2 & __memory_order_mask;
	memory_order __b1 = __m1 & __memory_order_mask;

	__glibcxx_assert(__b2 != memory_order_release);
	__glibcxx_assert(__b2 != memory_order_acq_rel);
	__glibcxx_assert(__b2 <= __b1);

	return __atomic_compare_exchange_n(&_M_p, &__p1, __p2, 0,
					   int(__m1), int(__m2));
      }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      fetch_add(ptrdiff_t __d,
		memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_add(&_M_p, _M_type_size(__d), int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      fetch_add(ptrdiff_t __d,
		memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_add(&_M_p, _M_type_size(__d), int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      fetch_sub(ptrdiff_t __d,
		memory_order __m = memory_order_seq_cst) noexcept
      { return __atomic_fetch_sub(&_M_p, _M_type_size(__d), int(__m)); }

      _GLIBCXX_ALWAYS_INLINE __pointer_type
      fetch_sub(ptrdiff_t __d,
		memory_order __m = memory_order_seq_cst) volatile noexcept
      { return __atomic_fetch_sub(&_M_p, _M_type_size(__d), int(__m)); }
    };

  // @} group atomics

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace std

#endif
