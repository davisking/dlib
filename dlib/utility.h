// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_UTILITY_Hh_
#define DLIB_UTILITY_Hh_

#include <cstddef>

/*
    This header contains back-ports of C++14/17 functions and type traits
    found in <utility> header of the standard library.
 */

namespace dlib
{
    // ---------------------------------------------------------------------

    template<std::size_t... Ints>
    struct index_sequence
    {
        using type = index_sequence;
        using value_type = std::size_t;
        static constexpr std::size_t size() noexcept {return sizeof...(Ints);}
    };

    template<class Sequence1, class Sequence2>
    struct merge_and_renumber;

    template<std::size_t... I1, std::size_t... I2>
    struct merge_and_renumber<index_sequence<I1...>, index_sequence<I2...>>
            : index_sequence < I1..., (sizeof...(I1) + I2)... > {};

    template<std::size_t N>
    struct make_index_sequence
        : merge_and_renumber < typename make_index_sequence < N / 2 >::type,
          typename make_index_sequence < N - N / 2 >::type > {};

    template<> struct make_index_sequence<0> : index_sequence<> {};
    template<> struct make_index_sequence<1> : index_sequence<0> {};

    template<typename... Ts>
    using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

    // ---------------------------------------------------------------------

    template <typename First, typename... Rest>
    struct are_nothrow_move_constructible
        : std::integral_constant<bool, std::is_nothrow_move_constructible<First>::value &&
                                       are_nothrow_move_constructible<Rest...>::value> {};

    template <typename T>
    struct are_nothrow_move_constructible<T> : std::is_nothrow_move_constructible<T> {};

    // ---------------------------------------------------------------------

    template <typename First, typename... Rest>
    struct are_nothrow_move_assignable
        : std::integral_constant<bool, std::is_nothrow_move_assignable<First>::value &&
                                       are_nothrow_move_assignable<Rest...>::value> {};

    template <typename T>
    struct are_nothrow_move_assignable<T> : std::is_nothrow_move_assignable<T> {};

    // ---------------------------------------------------------------------

    template <typename First, typename... Rest>
    struct are_nothrow_copy_constructible
        : std::integral_constant<bool, std::is_nothrow_copy_constructible<First>::value &&
                                       are_nothrow_copy_constructible<Rest...>::value> {};

    template <typename T>
    struct are_nothrow_copy_constructible<T> : std::is_nothrow_copy_constructible<T> {};

    // ---------------------------------------------------------------------

    template <typename First, typename... Rest>
    struct are_nothrow_copy_assignable
        : std::integral_constant<bool, std::is_nothrow_copy_assignable<First>::value &&
                                       are_nothrow_copy_assignable<Rest...>::value> {};

    template <typename T>
    struct are_nothrow_copy_assignable<T> : std::is_nothrow_copy_assignable<T> {};

    // ---------------------------------------------------------------------

    template< class... >
    using void_t = void;

    // ---------------------------------------------------------------------

    namespace swappable_details
    {
        using std::swap;

        template<typename T, typename = void>
        struct is_swappable : std::false_type {};

        template<typename T>
        struct is_swappable<T, void_t<decltype(swap(std::declval<T&>(), std::declval<T&>()))>> : std::true_type {};

        template<typename T>
        struct is_nothrow_swappable :
            std::integral_constant<bool, is_swappable<T>::value &&
                                         noexcept(swap(std::declval<T&>(), std::declval<T&>()))> {};
    }

    // ---------------------------------------------------------------------

    template<typename T>
    struct is_swappable : swappable_details::is_swappable<T>{};

    // ---------------------------------------------------------------------

    template<typename T>
    struct is_nothrow_swappable : swappable_details::is_nothrow_swappable<T>{};

    // ---------------------------------------------------------------------

    template <typename First, typename... Rest>
    struct are_nothrow_swappable
        : std::integral_constant<bool, is_nothrow_swappable<First>::value &&
                                       are_nothrow_swappable<Rest...>::value> {};

    template <typename T>
    struct are_nothrow_swappable<T> : is_nothrow_swappable<T> {};

    // ---------------------------------------------------------------------

    template<bool First, bool... Rest>
    struct And : std::integral_constant<bool, First && And<Rest...>::value> {};

    template<bool Value>
    struct And<Value> : std::integral_constant<bool, Value>{};

    // ---------------------------------------------------------------------

    template<std::size_t I>
    using size_ = std::integral_constant<std::size_t, I>;

    // ---------------------------------------------------------------------
}

#endif //DLIB_UTILITY_Hh_
