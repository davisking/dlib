// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IF_CONSTEXPR_H
#define DLIB_IF_CONSTEXPR_H

#include "overloaded.h"

namespace dlib
{
    namespace detail
    {
//        const auto _ = [](auto&& arg) -> decltype(auto){ return std::forward<decltype(arg)>(arg); }; // same as std::identity in C++20 but better
        template<typename Void, template <class...> class Op, class... Args>
        struct is_detected : std::false_type{};

        template<template <class...> class Op, class... Args>
        struct is_detected<void(Op<Args...>), Op, Args...> : std::true_type {};
    }

    template<typename... T>
    struct types_ {};

    template<bool... v>
    using bools_ = types_<std::integral_constant<bool,v>...>;

    template<bool... v>
    auto bools(std::integral_constant<bool, v>...)
    {
        return bools_<v...>{};
    }

    template<template <class...> class Op, class... Args>
    using is_detected = detail::is_detected<void, Op, Args...>;

    template <
        typename... T,
        typename... Cases
    >
    constexpr decltype(auto) switch_(
        types_<T...> meta_obj,
        Cases&&...   cases
    )
    {
        return overloaded(std::forward<Cases>(cases)...)(meta_obj);
    }
}

#endif //DLIB_IF_CONSTEXPR_H