// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_TRAITS_H_
#define DLIB_TYPE_TRAITS_H_

/*
    This header contains back-ports of C++14/17 type traits
    It also contains aliases for things found in <type_traits> which
    deprecate old dlib type traits.
*/

#include <type_traits>

namespace dlib
{
// ----------------------------------------------------------------------------------------

    template <typename T>
    using is_pointer_type = std::is_pointer<T>;

// ----------------------------------------------------------------------------------------

    template <typename T>
    using is_const_type = std::is_const<std::remove_reference_t<T>>;

// ----------------------------------------------------------------------------------------

    template <typename T>
    using is_reference_type = std::is_reference<std::remove_const_t<T>>;

// ----------------------------------------------------------------------------------------

    template<typename T>
    using is_function = std::is_function<T>;

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    using is_same_type = std::is_same<T,U>;

// ----------------------------------------------------------------------------------------

#ifdef __cpp_fold_expressions
    template<bool... v>
    struct And : std::integral_constant<bool, (... && v)> {};
#else
    template<bool First, bool... Rest>
    struct And : std::integral_constant<bool, First && And<Rest...>::value> {};

    template<bool Value>
    struct And<Value> : std::integral_constant<bool, Value>{};
#endif

// ----------------------------------------------------------------------------------------

#ifdef __cpp_fold_expressions
    template<bool... v>
    struct Or : std::integral_constant<bool, (... || v)> {};
#else
    template<bool First, bool... Rest>
    struct Or : std::integral_constant<bool, First || Or<Rest...>::value> {};

    template<bool Value>
    struct Or<Value> : std::integral_constant<bool, Value>{};
#endif

// ----------------------------------------------------------------------------------------

    /*!A is_any 

        This is a template where is_any<T,Rest...>::value == true when T is 
        the same type as any one of the types in Rest... 
    !*/

    template <typename T, typename... Types>
    struct is_any : Or<std::is_same<T,Types>::value...> {};

// ----------------------------------------------------------------------------------------

    template<typename T>
    using is_float_type = std::is_floating_point<T>;

// ----------------------------------------------------------------------------------------

    template <typename T>
    using is_unsigned_type = std::is_unsigned<T>;

// ----------------------------------------------------------------------------------------

    template <typename T>
    using is_signed_type = std::is_signed<T>;

// ----------------------------------------------------------------------------------------

    template <typename T> 
    using is_built_in_scalar_type = std::is_arithmetic<T>;

 // ----------------------------------------------------------------------------------------

    template< class T >
    using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

    template <typename T>
    struct basic_type { using type = remove_cvref_t<T>; };

 // ----------------------------------------------------------------------------------------

    template<class...>
    struct conjunction : std::true_type {};

    template<class B1>
    struct conjunction<B1> : B1 {};

    template<class B1, class... Bn>
    struct conjunction<B1, Bn...> : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

 // ----------------------------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_move_constructible : And<std::is_nothrow_move_constructible<Types>::value...> {};

 // ----------------------------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_move_assignable : And<std::is_nothrow_move_assignable<Types>::value...> {};

 // ----------------------------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_copy_constructible : And<std::is_nothrow_copy_constructible<Types>::value...> {};

 // ----------------------------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_copy_assignable : And<std::is_nothrow_copy_assignable<Types>::value...> {};

 // ----------------------------------------------------------------------------------------

    template< class... >
    using void_t = void;

 // ----------------------------------------------------------------------------------------

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

 // ----------------------------------------------------------------------------------------

    template<typename T>
    struct is_swappable : swappable_details::is_swappable<T>{};

 // ----------------------------------------------------------------------------------------

    template<typename T>
    struct is_nothrow_swappable : swappable_details::is_nothrow_swappable<T>{};

// ----------------------------------------------------------------------------------------

    template <typename ...Types>
    struct are_nothrow_swappable : And<is_nothrow_swappable<Types>::value...> {};

 // ----------------------------------------------------------------------------------------

    template<std::size_t I>
    using size_ = std::integral_constant<std::size_t, I>;

 // ----------------------------------------------------------------------------------------

    template <typename from, typename to>
    using is_convertible = std::is_convertible<from, to>;

 // ----------------------------------------------------------------------------------------
}

#endif //DLIB_TYPE_TRAITS_H_