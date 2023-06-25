// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_UTILITY_Hh_
#define DLIB_UTILITY_Hh_

#include <cstddef>
#include <utility>

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

    template<class Sequence>
    struct pop_front {};

    template<std::size_t I, std::size_t... Ints>
    struct pop_front<index_sequence<I, Ints...>>
    {
        using type = index_sequence<Ints...>;
    };

    template<class Sequence>
    using pop_front_t = typename pop_front<Sequence>::type;

    // ---------------------------------------------------------------------

    struct in_place_t 
    {
        explicit in_place_t() = default;
    };

    static constexpr in_place_t in_place{};

    // ---------------------------------------------------------------------
}

#endif //DLIB_UTILITY_Hh_
