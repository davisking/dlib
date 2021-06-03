// <thread> -*- C++ -*-

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

/** @file include/thread
 *  This is a Standard C++ Library header.
 */

#ifndef _GLIBCXX_THREAD
#define _GLIBCXX_THREAD 1

#pragma GCC system_header

#if __cplusplus < 201103L
# include <bits/c++0x_warning.h>
#else

#include <chrono>
#include <memory>
#include <tuple>
#include <cerrno>
#include <bits/functexcept.h>
#include <bits/functional_hash.h>
#include <bits/invoke.h>
#include <bits/gthr.h>

#if defined(_GLIBCXX_HAS_GTHREADS)

namespace std _GLIBCXX_VISIBILITY(default)
{
_GLIBCXX_BEGIN_NAMESPACE_VERSION

  /**
   * @defgroup threads Threads
   * @ingroup concurrency
   *
   * Classes for thread support.
   * @{
   */

  /// thread
  class thread
  {
  public:
    // Abstract base class for types that wrap arbitrary functors to be
    // invoked in the new thread of execution.
    struct _State
    {
      virtual ~_State();
      virtual void _M_run() = 0;
    };
    using _State_ptr = unique_ptr<_State>;

    typedef __gthread_t			native_handle_type;

    /// thread::id
    class id
    {
      native_handle_type	_M_thread;

    public:
      id() noexcept : _M_thread() { }

      explicit
      id(native_handle_type __id) : _M_thread(__id) { }

    private:
      friend class thread;
      friend class hash<thread::id>;

      friend bool
      operator==(thread::id __x, thread::id __y) noexcept;

      friend bool
      operator<(thread::id __x, thread::id __y) noexcept;

      template<class _CharT, class _Traits>
	friend basic_ostream<_CharT, _Traits>&
	operator<<(basic_ostream<_CharT, _Traits>& __out, thread::id __id);
    };

  private:
    id				_M_id;

    // _GLIBCXX_RESOLVE_LIB_DEFECTS
    // 2097.  packaged_task constructors should be constrained
    // 3039. Unnecessary decay in thread and packaged_task
    template<typename _Tp>
      using __not_same = __not_<is_same<__remove_cvref_t<_Tp>, thread>>;

  public:
    thread() noexcept = default;

    template<typename _Callable, typename... _Args,
	     typename = _Require<__not_same<_Callable>>>
      explicit
      thread(_Callable&& __f, _Args&&... __args)
      {
	static_assert( __is_invocable<typename decay<_Callable>::type,
				      typename decay<_Args>::type...>::value,
	  "std::thread arguments must be invocable after conversion to rvalues"
	  );

#ifdef GTHR_ACTIVE_PROXY
	// Create a reference to pthread_create, not just the gthr weak symbol.
	auto __depend = reinterpret_cast<void(*)()>(&pthread_create);
#else
	auto __depend = nullptr;
#endif
        _M_start_thread(_S_make_state(
	      __make_invoker(std::forward<_Callable>(__f),
			     std::forward<_Args>(__args)...)),
	    __depend);
      }

    ~thread()
    {
      if (joinable())
	std::terminate();
    }

    thread(const thread&) = delete;

    thread(thread&& __t) noexcept
    { swap(__t); }

    thread& operator=(const thread&) = delete;

    thread& operator=(thread&& __t) noexcept
    {
      if (joinable())
	std::terminate();
      swap(__t);
      return *this;
    }

    void
    swap(thread& __t) noexcept
    { std::swap(_M_id, __t._M_id); }

    bool
    joinable() const noexcept
    { return !(_M_id == id()); }

    void
    join();

    void
    detach();

    thread::id
    get_id() const noexcept
    { return _M_id; }

    /** @pre thread is joinable
     */
    native_handle_type
    native_handle()
    { return _M_id._M_thread; }

    // Returns a value that hints at the number of hardware thread contexts.
    static unsigned int
    hardware_concurrency() noexcept;

  private:
    template<typename _Callable>
      struct _State_impl : public _State
      {
	_Callable		_M_func;

	_State_impl(_Callable&& __f) : _M_func(std::forward<_Callable>(__f))
	{ }

	void
	_M_run() { _M_func(); }
      };

    void
    _M_start_thread(_State_ptr, void (*)());

    template<typename _Callable>
      static _State_ptr
      _S_make_state(_Callable&& __f)
      {
	using _Impl = _State_impl<_Callable>;
	return _State_ptr{new _Impl{std::forward<_Callable>(__f)}};
      }
#if _GLIBCXX_THREAD_ABI_COMPAT
  public:
    struct _Impl_base;
    typedef shared_ptr<_Impl_base>	__shared_base_type;
    struct _Impl_base
    {
      __shared_base_type	_M_this_ptr;
      virtual ~_Impl_base() = default;
      virtual void _M_run() = 0;
    };

  private:
    void
    _M_start_thread(__shared_base_type, void (*)());

    void
    _M_start_thread(__shared_base_type);
#endif

  private:
    // A call wrapper that does INVOKE(forwarded tuple elements...)
    template<typename _Tuple>
      struct _Invoker
      {
	_Tuple _M_t;

	template<typename>
	  struct __result;
	template<typename _Fn, typename... _Args>
	  struct __result<tuple<_Fn, _Args...>>
	  : __invoke_result<_Fn, _Args...>
	  { };

	template<size_t... _Ind>
	  typename __result<_Tuple>::type
	  _M_invoke(_Index_tuple<_Ind...>)
	  { return std::__invoke(std::get<_Ind>(std::move(_M_t))...); }

	typename __result<_Tuple>::type
	operator()()
	{
	  using _Indices
	    = typename _Build_index_tuple<tuple_size<_Tuple>::value>::__type;
	  return _M_invoke(_Indices());
	}
      };

    template<typename... _Tp>
      using __decayed_tuple = tuple<typename decay<_Tp>::type...>;

  public:
    // Returns a call wrapper that stores
    // tuple{DECAY_COPY(__callable), DECAY_COPY(__args)...}.
    template<typename _Callable, typename... _Args>
      static _Invoker<__decayed_tuple<_Callable, _Args...>>
      __make_invoker(_Callable&& __callable, _Args&&... __args)
      {
	return { __decayed_tuple<_Callable, _Args...>{
	    std::forward<_Callable>(__callable), std::forward<_Args>(__args)...
	} };
      }
  };

  inline void
  swap(thread& __x, thread& __y) noexcept
  { __x.swap(__y); }

  inline bool
  operator==(thread::id __x, thread::id __y) noexcept
  {
    // pthread_equal is undefined if either thread ID is not valid, so we
    // can't safely use __gthread_equal on default-constructed values (nor
    // the non-zero value returned by this_thread::get_id() for
    // single-threaded programs using GNU libc). Assume EqualityComparable.
    return __x._M_thread == __y._M_thread;
  }

  inline bool
  operator!=(thread::id __x, thread::id __y) noexcept
  { return !(__x == __y); }

  inline bool
  operator<(thread::id __x, thread::id __y) noexcept
  {
    // Pthreads doesn't define any way to do this, so we just have to
    // assume native_handle_type is LessThanComparable.
    return __x._M_thread < __y._M_thread;
  }

  inline bool
  operator<=(thread::id __x, thread::id __y) noexcept
  { return !(__y < __x); }

  inline bool
  operator>(thread::id __x, thread::id __y) noexcept
  { return __y < __x; }

  inline bool
  operator>=(thread::id __x, thread::id __y) noexcept
  { return !(__x < __y); }

  // DR 889.
  /// std::hash specialization for thread::id.
  template<>
    struct hash<thread::id>
    : public __hash_base<size_t, thread::id>
    {
      size_t
      operator()(const thread::id& __id) const noexcept
      { return std::_Hash_impl::hash(__id._M_thread); }
    };

  template<class _CharT, class _Traits>
    inline basic_ostream<_CharT, _Traits>&
    operator<<(basic_ostream<_CharT, _Traits>& __out, thread::id __id)
    {
      if (__id == thread::id())
	return __out << "thread::id of a non-executing thread";
      else
	return __out << __id._M_thread;
    }

  /** @namespace std::this_thread
   *  @brief ISO C++ 2011 entities sub-namespace for thread.
   *  30.3.2 Namespace this_thread.
   */
  namespace this_thread
  {
    /// get_id
    inline thread::id
    get_id() noexcept
    {
#ifdef __GLIBC__
      // For the GNU C library pthread_self() is usable without linking to
      // libpthread.so but returns 0, so we cannot use it in single-threaded
      // programs, because this_thread::get_id() != thread::id{} must be true.
      // We know that pthread_t is an integral type in the GNU C library.
      if (!__gthread_active_p())
	return thread::id(1);
#endif
      return thread::id(__gthread_self());
    }

    /// yield
    inline void
    yield() noexcept
    {
#ifdef _GLIBCXX_USE_SCHED_YIELD
      __gthread_yield();
#endif
    }

    void
    __sleep_for(chrono::seconds, chrono::nanoseconds);

    /// sleep_for
    template<typename _Rep, typename _Period>
      inline void
      sleep_for(const chrono::duration<_Rep, _Period>& __rtime)
      {
	if (__rtime <= __rtime.zero())
	  return;
	auto __s = chrono::duration_cast<chrono::seconds>(__rtime);
	auto __ns = chrono::duration_cast<chrono::nanoseconds>(__rtime - __s);
#ifdef _GLIBCXX_USE_NANOSLEEP
	__gthread_time_t __ts =
	  {
	    static_cast<std::time_t>(__s.count()),
	    static_cast<long>(__ns.count())
	  };
	while (::nanosleep(&__ts, &__ts) == -1 && errno == EINTR)
	  { }
#else
	__sleep_for(__s, __ns);
#endif
      }

    /// sleep_until
    template<typename _Clock, typename _Duration>
      inline void
      sleep_until(const chrono::time_point<_Clock, _Duration>& __atime)
      {
	auto __now = _Clock::now();
	if (_Clock::is_steady)
	  {
	    if (__now < __atime)
	      sleep_for(__atime - __now);
	    return;
	  }
	while (__now < __atime)
	  {
	    sleep_for(__atime - __now);
	    __now = _Clock::now();
	  }
      }
  }

  // @} group threads

_GLIBCXX_END_NAMESPACE_VERSION
} // namespace

#endif // _GLIBCXX_HAS_GTHREADS

#endif // C++11

#endif // _GLIBCXX_THREAD
