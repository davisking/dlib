// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_INVOKE_Hh_
#define DLIB_INVOKE_Hh_

#include <functional>
#include <type_traits>
#include "utility.h"

namespace dlib
{
    // ----------------------------------------------------------------------------------------

#if __cplusplus < 201703L
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
                not std::is_member_pointer<typename std::decay<F>::type>::value,
                decltype(std::forward<F>(fn)(std::forward<Args>(args)...))>::type
        {
            return std::forward<F>(fn)(std::forward<Args>(args)...);
        }

        template< typename AlwaysVoid, typename, typename...>
        struct invoke_result {};

        template< typename F, typename... Args >
        struct invoke_result< decltype( void(INVOKE(std::declval<F>(), std::declval<Args>()...)) ), F, Args...>
        {
            using type = decltype( INVOKE(std::declval<F>(), std::declval<Args>()...) );
        };
    }

    template< typename F, typename... Args>
    auto invoke(F && f, Args &&... args)
    -> decltype(detail::INVOKE(std::forward<F>( f ), std::forward<Args>( args )...))
    {
        return detail::INVOKE(std::forward<F>( f ), std::forward<Args>( args )...);
    }

    template< typename F, typename... Args >
    struct invoke_result : detail::invoke_result< void, F, Args...> {};

    template< typename > struct result_of;

    template< typename F, typename... Args >
    struct result_of< F(Args...) > : detail::invoke_result< void, F, Args...> {};

#else
    using std::invoke;
    using std::invoke_result;
    using std::result_of;
#endif

    // ----------------------------------------------------------------------------------------

#if __cplusplus < 201703L
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
#else
    using std::apply;
#endif

    // ----------------------------------------------------------------------------------------
}

#endif //DLIB_INVOKE_Hh_