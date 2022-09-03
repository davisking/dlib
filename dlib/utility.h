// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_UTILITY_Hh_
#define DLIB_UTILITY_Hh_

#include <cstddef>
#include <type_traits>

/*
    This header contains back-ports of C++14/17 functions and type traits
    found in <utility> header of the standard library.
 */

namespace dlib
{
#ifdef __cpp_lib_integer_sequence
    template<std::size_t... Ints>
    using index_sequence = std::index_sequence<Ints...>;

    template<std::size_t N>
    using make_index_sequence = std::make_index_sequence<N>;

    template<class... T>
    using index_sequence_for = std::index_sequence_for<T...>;
#else
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
#endif

    // ---------------------------------------------------------------------

    template<bool First, bool... Rest>
    struct And : std::integral_constant<bool, First && And<Rest...>::value> {};

    template<bool Value>
    struct And<Value> : std::integral_constant<bool, Value>{};

    // ---------------------------------------------------------------------

    template<class...>
    struct conjunction : std::true_type {};

    template<class B1>
    struct conjunction<B1> : B1 {};

    template<class B1, class... Bn>
    struct conjunction<B1, Bn...> : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

    // ---------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_move_constructible : And<std::is_nothrow_move_constructible<Types>::value...> {};

    // ---------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_move_assignable : And<std::is_nothrow_move_assignable<Types>::value...> {};

    // ---------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_copy_constructible : And<std::is_nothrow_copy_constructible<Types>::value...> {};

    // ---------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_copy_assignable : And<std::is_nothrow_copy_assignable<Types>::value...> {};

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

    template <typename ...Types>
    struct are_nothrow_swappable : And<is_nothrow_swappable<Types>::value...> {};

    // ---------------------------------------------------------------------

    template<std::size_t I>
    using size_ = std::integral_constant<std::size_t, I>;

    // ---------------------------------------------------------------------
}

#endif //DLIB_UTILITY_Hh_
