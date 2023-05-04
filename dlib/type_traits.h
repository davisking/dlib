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

    template<class Byte>
    using is_byte = std::integral_constant<bool, std::is_same<Byte,char>::value
                                              || std::is_same<Byte,int8_t>::value
                                              || std::is_same<Byte,uint8_t>::value
#ifdef __cpp_lib_byte
                                              || std::is_same<Byte,std::byte>::value
#endif
                                          >;

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

        template<class T, class = void>
        struct swap_traits
        {
            constexpr static bool is_swappable{false};
            constexpr static bool is_nothrow{false};
        };

        template<class T>
        struct swap_traits<T, void_t<decltype(swap(std::declval<T&>(), std::declval<T&>()))>>
        {
            constexpr static bool is_swappable{true};
            constexpr static bool is_nothrow{noexcept(swap(std::declval<T&>(), std::declval<T&>()))};
        };
    }

// ----------------------------------------------------------------------------------------

    template<class T>
    using is_swappable = std::integral_constant<bool, swappable_details::swap_traits<T>::is_swappable>;

// ----------------------------------------------------------------------------------------

    template<class T>
    using is_nothrow_swappable = std::integral_constant<bool, swappable_details::swap_traits<T>::is_nothrow>;

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

    namespace details
    {
        template<class T, class AlwaysVoid = void>
        struct is_complete_type : std::false_type{};

        template<class T>
        struct is_complete_type<T, void_t<decltype(sizeof(T))>> : std::true_type{};
    }

    template<class T>
    using is_complete_type = details::is_complete_type<T>;

// ----------------------------------------------------------------------------------------
 
    template<typename... T>
    struct types_ {};
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a type list. You can use this for general-purpose meta-programming
            and it's used to pass types to the switch_() function.
    !*/

// ----------------------------------------------------------------------------------------

    namespace details
    {
        template<std::size_t I, class... Ts>
        struct nth_type;

        template<std::size_t I, class T0, class... Ts>
        struct nth_type<I, T0, Ts...> { using type = typename nth_type<I-1, Ts...>::type; };

        template<class T0, class... Ts>
        struct nth_type<0, T0, Ts...> { using type = T0; };
    }

    template<std::size_t I, class... Ts>
    struct nth_type;
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a type trait for getting the n'th argument of a parameter pack.
    !*/

    template<std::size_t I, class... Ts>
    struct nth_type<I, types_<Ts...>> : details::nth_type<I,Ts...> {};

    template<std::size_t I, class... Ts>
    struct nth_type : details::nth_type<I,Ts...> {};

    template<std::size_t I, class... Ts>
    using nth_type_t = typename nth_type<I,Ts...>::type;

// ----------------------------------------------------------------------------------------

    template<class F>
    struct callable_traits;

    template<class R, class... Args>
    struct callable_traits<R(Args...)>
    {
        using return_type = R;
        using args        = types_<Args...>;
    };  

    template<class R, class... Args>
    struct callable_traits<R(*)(Args...)> : public callable_traits<R(Args...)>{};

    template<class C, class R, class... Args>
    struct callable_traits<R(C::*)(Args...)> : public callable_traits<R(Args...)>{};

    template<class C, class R, class... Args>
    struct callable_traits<R(C::*)(Args...) const> : public callable_traits<R(Args...)>{};

    template<class F>
    struct callable_traits
    {
        using call_type     = callable_traits<decltype(&F::operator())>;
        using return_type   = typename call_type::return_type;
        using args          = typename call_type::args;
    };

    template<class Callable>
    using callable_args = typename callable_traits<Callable>::args;

    template<std::size_t I, class Callable>
    using callable_arg = nth_type_t<I, callable_args<Callable>>;

    template<class Callable>
    using callable_return = typename callable_traits<Callable>::return_type;

// ----------------------------------------------------------------------------------------

}

#endif //DLIB_TYPE_TRAITS_H_
