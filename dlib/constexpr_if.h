// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IF_CONSTEXPR_H
#define DLIB_IF_CONSTEXPR_H

#include <utility>

namespace dlib
{
    namespace detail
    {
        const auto _ = [](auto&& arg) -> decltype(auto){ return std::forward<decltype(arg)>(arg); }; // same as std::identity in C++20 but better

        template <
            bool     Force,
            typename Type,
            typename Case
        >
        struct case_type_and_lambda
        {
            static constexpr bool force = Force;
            using type = Type;
            Case case_;

            constexpr decltype(auto) operator()() { return case_(_); }
        };

        template<
            typename type,
            typename Case
        >
        using valid_t = std::enable_if_t<std::decay_t<Case>::force || std::is_same<type, typename std::decay_t<Case>::type>::value, bool>;

        template<
            typename type,
            typename Case
        >
        using not_valid_t = std::enable_if_t<!std::decay_t<Case>::force && !std::is_same<type, typename std::decay_t<Case>::type>::value, bool>;
    }

    template <
        typename type,
        typename Case
    >
    constexpr auto case_type_(
        Case &&case_
    )
    {
        return detail::case_type_and_lambda<false, type, std::decay_t<Case>>{std::forward<Case>(case_)};
    }

    template <
        bool result,
        typename Case
    >
    constexpr auto case_(
        Case &&case_
    )
    {
        using bool_t = std::integral_constant<bool,result>;
        return case_type_<bool_t>(std::forward<Case>(case_));
    }

    template <typename Case>
    constexpr auto default_(Case &&case_)
    {
        return detail::case_type_and_lambda<true, bool, std::decay_t<Case>>{std::forward<Case>(case_)};
    }

    template <
        typename type,
        typename Case1,
        typename... CaseRest,
        detail::valid_t<type, Case1> = true
    >
    constexpr decltype(auto) switch_type_(
        Case1&& first,
        CaseRest&&...
    )
    {
        return first();
    }

    template <
        typename type,
        typename Case1,
        typename... CaseRest,
        detail::not_valid_t<type, Case1> = true
    >
    constexpr decltype(auto) switch_type_(
        Case1&&,
        CaseRest&&... cases
    )
    {
        return switch_type_<type>(std::forward<CaseRest>(cases)...);
    }

    template <
        typename type,
        typename CaseLast,
        detail::valid_t<type, CaseLast> = true
    >
    constexpr decltype(auto) switch_type_(
        CaseLast&& last
    )
    {
        return last();
    }

    template <
        typename type,
        typename CaseLast,
        detail::not_valid_t<type, CaseLast> = true
    >
    constexpr void switch_type_(CaseLast&&)
    {
    }

    template <
        typename...Cases
    >
    constexpr decltype(auto) switch_(
        Cases&&... cases
    )
    {
        return switch_type_<std::true_type>(std::forward<Cases>(cases)...);
    }

    namespace detail
    {
        template<typename Void, template <typename...> typename Op, typename... Args>
        struct is_detected : std::false_type{};

        template<template <typename...> typename Op, typename... Args>
        struct is_detected<std::void_t<Op<Args...>>, Op, Args...> : std::true_type {};
    }

    template<template <typename...> typename Op, typename... Args>
    using is_detected = detail::is_detected<void, Op, Args...>;
}

#endif //DLIB_IF_CONSTEXPR_H