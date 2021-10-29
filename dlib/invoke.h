// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_INVOKE_Hh_
#define DLIB_INVOKE_Hh_

#ifndef DLIB_CPLUSPLUS
    #if defined(_MSVC_LANG ) && !defined(__clang__)
        #define DLIB_CPLUSPLUS  (_MSC_VER == 1900 ? 201103L : _MSVC_LANG )
    #else
        #define DLIB_CPLUSPLUS  __cplusplus
    #endif
#endif

#if defined(_MSC_VER)
    #if _MSC_VER >= 1900
        #define DLIB_HAVE_INVOKE 1
    #endif
#else
    #if DLIB_CPLUSPLUS >= 201703L
        #define DLIB_HAVE_INVOKE 1
    #endif
#endif

#if DLIB_CPLUSPLUS >= 201703L
    #define DLIB_HAVE_INVOKE_RESULT 1
    #define DLIB_HAVE_APPLY 1
#endif

#include <functional>
#include <type_traits>
#include "utility.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

#if DLIB_HAVE_INVOKE
    using std::invoke;
#else
    namespace detail {
        template< typename F, typename ... Args >
        auto INVOKE(F&& fn, Args&& ... args)
        -> typename std::enable_if<
                std::is_member_pointer<typename std::decay<F>::type>::value,
                decltype(std::mem_fn(fn)(std::forward<Args>(args)...))>::type
        {
            return std::mem_fn(fn)(std::forward<Args>(args)...) ;
        }

        template< typename F, typename ... Args >
        auto INVOKE(F&& fn, Args&& ... args)
        -> typename std::enable_if<
                !std::is_member_pointer<typename std::decay<F>::type>::value,
                decltype(std::forward<F>(fn)(std::forward<Args>(args)...))>::type
        {
            return std::forward<F>(fn)(std::forward<Args>(args)...);
        }
    }

    template< typename F, typename... Args>
    auto invoke(F && f, Args &&... args)
    -> decltype(detail::INVOKE(std::forward<F>( f ), std::forward<Args>( args )...))
    {
        return detail::INVOKE(std::forward<F>( f ), std::forward<Args>( args )...);
    }
#endif

    // ----------------------------------------------------------------------------------------

#if DLIB_HAVE_INVOKE_RESULT
    using std::invoke_result;
#else
    namespace detail
    {
        template< typename AlwaysVoid, typename, typename...>
        struct invoke_result {};

        template< typename F, typename... Args >
        struct invoke_result< decltype( void(invoke(std::declval<F>(), std::declval<Args>()...)) ), F, Args...>
        {
            using type = decltype( invoke(std::declval<F>(), std::declval<Args>()...) );
        };
    }

    template< typename F, typename... Args >
    struct invoke_result : detail::invoke_result< void, F, Args...> {};
#endif

    // ----------------------------------------------------------------------------------------

#if DLIB_HAVE_APPLY
    using std::apply;
#else
    namespace detail {
        template<typename F, typename Tuple, std::size_t... I>
        auto apply_impl(F&& fn, Tuple&& tpl, index_sequence<I...>)
        -> decltype(invoke(std::forward<F>(fn),
                           std::get<I>(std::forward<Tuple>(tpl))...))
        {
            return invoke(std::forward<F>(fn),
                          std::get<I>(std::forward<Tuple>(tpl))...);
        }
    }

    template<typename F, typename Tuple>
    auto apply(F&& fn, Tuple&& tpl)
    -> decltype(detail::apply_impl(std::forward<F>(fn),
                                   std::forward<Tuple>(tpl),
                                   make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type >::value>{}))
    {
        return detail::apply_impl(std::forward<F>(fn),
                                  std::forward<Tuple>(tpl),
                                  make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type >::value>{});
    }
#endif

    // ----------------------------------------------------------------------------------------
}

#endif //DLIB_INVOKE_Hh_
