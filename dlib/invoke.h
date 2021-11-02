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
    /*!
        ensures
            - identical to std::invoke(std::forward<F>(f), std::forward<Args>(args)...)
            - works with C++11 onwards
    !*/
    -> decltype(detail::INVOKE(std::forward<F>( f ), std::forward<Args>( args )...))
    {
        return detail::INVOKE(std::forward<F>( f ), std::forward<Args>( args )...);
    }

    // ----------------------------------------------------------------------------------------

    namespace detail
    {
        template< typename AlwaysVoid, typename, typename...>
        struct invoke_traits
        {
            static constexpr bool value = false;
        };

        template< typename F, typename... Args >
        struct invoke_traits< decltype( void(dlib::invoke(std::declval<F>(), std::declval<Args>()...)) ), F, Args...>
        {
            static constexpr bool value = true;
            using type = decltype( dlib::invoke(std::declval<F>(), std::declval<Args>()...) );
        };
    }

    template< typename F, typename... Args >
    struct invoke_result : detail::invoke_traits< void, F, Args...> {};
    /*!
        ensures
            - identical to std::invoke_result<F, Args..>
            - works with C++11 onwards
    !*/

    template< typename F, typename... Args >
    using invoke_result_t = typename invoke_result<F, Args...>::type;
    /*!
        ensures
            - identical to std::invoke_result_t<F, Args..>
            - works with C++11 onwards
    !*/

    // ----------------------------------------------------------------------------------------

    template< typename F, typename... Args >
    struct is_invocable : std::integral_constant<bool, detail::invoke_traits< void, F, Args...>::value> {};
    /*!
        ensures
            - identical to std::is_invocable<F, Args..>
            - works with C++11 onwards
    !*/

    // ----------------------------------------------------------------------------------------

    namespace detail
    {
        template<typename F, typename Tuple, std::size_t... I>
        auto apply_impl(F&& fn, Tuple&& tpl, index_sequence<I...>)
        -> decltype(dlib::invoke(std::forward<F>(fn),
                                 std::get<I>(std::forward<Tuple>(tpl))...))
        {
            return dlib::invoke(std::forward<F>(fn),
                                std::get<I>(std::forward<Tuple>(tpl))...);
        }
    }

    template<typename F, typename Tuple>
    auto apply(F&& fn, Tuple&& tpl)
    /*!
        ensures
            - identical to std::apply(std::forward<F>(f), std::forward<Tuple>(tpl))
            - works with C++11 onwards
    !*/
    -> decltype(detail::apply_impl(std::forward<F>(fn),
                                   std::forward<Tuple>(tpl),
                                   make_index_sequence<std::tuple_size<typename std::remove_reference<Tuple>::type >::value>{}))
    {
        return detail::apply_impl(std::forward<F>(fn),
                                  std::forward<Tuple>(tpl),
                                  make_index_sequence<std::tuple_size<typename std::remove_reference<Tuple>::type >::value>{});
    }

    // ----------------------------------------------------------------------------------------

    namespace detail
    {
        template <class T, class Tuple, std::size_t... I>
        constexpr T make_from_tuple_impl( Tuple&& t, index_sequence<I...> )
        {
            return T(std::get<I>(std::forward<Tuple>(t))...);
        }
    }

    template <class T, class Tuple>
    constexpr T make_from_tuple( Tuple&& t )
    /*!
        ensures
            - identical to std::make_from_tuple<T>(std::forward<Tuple>(t))
            - works with C++11 onwards
    !*/
    {
        return detail::make_from_tuple_impl<T>(std::forward<Tuple>(t),
                                               make_index_sequence<std::tuple_size<typename std::remove_reference<Tuple>::type >::value>{});
    }

    // ----------------------------------------------------------------------------------------
}

#endif //DLIB_INVOKE_Hh_