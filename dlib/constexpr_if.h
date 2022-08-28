// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IF_CONSTEXPR_H
#define DLIB_IF_CONSTEXPR_H

#include "overloaded.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------

    namespace detail
    {
        const auto _ = [](auto&& arg) -> decltype(auto) { return std::forward<decltype(arg)>(arg); };

        template<typename...T>
        using void_t = void;

        template<typename Void, template <class...> class Op, class... Args>
        struct is_detected : std::false_type{};

        template<template <class...> class Op, class... Args>
        struct is_detected<void_t<Op<Args...>>, Op, Args...> : std::true_type {};
    }

// ----------------------------------------------------------------------------------------

    template<typename... T>
    struct types_ {};
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a type list. Use this to pass types to the switch_() function.
   !*/

// ----------------------------------------------------------------------------------------

    template<bool... v>
    using bools_ = types_<std::integral_constant<bool,v>...>;
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a type alias which combines a set of compile time booleans.
   !*/

    template<bool... v>
    auto bools(std::integral_constant<bool, v>...)
    {
        return bools_<v...>{};
    }
    /*!
        ensures
            - returns a type list of compile time booleans.
              You probably want to use this instead of things like:
                - types_<std::is_same<T,U>>{} or
                - types_<std::is_constructible<T>>{}
              since these aren't trivially convertible to
                - types_<std::true_type>{} or
                - types_<std::false_type>{}

              Instead use the following:
                - bools(std::is_same<T,U>{})
                - bools(std::is_constructible<T>>{})
              This will convert to either either std::true_type or std::false_type
   !*/

// ----------------------------------------------------------------------------------------

    template <
        typename... T,
        typename... Cases
    >
    constexpr decltype(auto) switch_(
        types_<T...> meta_obj,
        Cases&&...   cases
    )
    {
        return overloaded(std::forward<Cases>(cases)...)(meta_obj, detail::_);
    }
    /*!
        requires
            - meta_obj combines a set of initial types
            - cases is a set of overload-able conditional branches.
            - at least one of the cases is callable given meta_obj.
        ensures
            - calls the correct conditional branch.
            - the correct conditional branch selected at compile-time.

            Here is an example:

            template<typename T>
            auto perform_correct_action(T& obj)
            {
                return switch(
                    types_<T>{},
                    [&](types_<A>, auto _) {
                        return _(obj).set_something_specific_to_A_and_return_something();
                    },
                    [&](types_<B>, auto _) {
                        return _(obj).set_something_specific_to_B_and_return_something();
                    },
                    [&](auto...) {
                        // Default case statement. Do something sensible.
                        return false;
                    }
                );
            }
      !*/
// ----------------------------------------------------------------------------------------

    template<template <class...> class Op, class... Args>
    using is_detected = detail::is_detected<void, Op, Args...>;
    /*!
        ensures
            - This is exactly the same as std::experimental::is_detected from library fundamentals v
   !*/

// ----------------------------------------------------------------------------------------
}

#endif //DLIB_IF_CONSTEXPR_H