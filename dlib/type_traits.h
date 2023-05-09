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

    namespace details
    {
        template<typename Void, template <class...> class Op, class... Args>
        struct is_detected : std::false_type{};

        template<template <class...> class Op, class... Args>
        struct is_detected<dlib::void_t<Op<Args...>>, Op, Args...> : std::true_type {};
    }

    template<template <class...> class Op, class... Args>
    using is_detected = details::is_detected<void, Op, Args...>;
    /*!
        ensures
            - This is exactly the same as std::experimental::is_detected from library fundamentals v
    !*/

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
#if defined(__has_builtin)
#if __has_builtin(__type_pack_element)
    #define HAS_TYPE_PACK_ELEMENT 1
#endif
#endif

#if HAS_TYPE_PACK_ELEMENT
        template<std::size_t I, class... Ts>
        struct nth_type { using type = __type_pack_element<I,Ts...>; };
#else
        template<std::size_t I, class... Ts>
        struct nth_type;

        template<std::size_t I, class T0, class... Ts>
        struct nth_type<I, T0, Ts...> { using type = typename nth_type<I-1, Ts...>::type; };

        template<class T0, class... Ts>
        struct nth_type<0, T0, Ts...> { using type = T0; };
#endif
    }

    template<std::size_t I, class... Ts>
    struct nth_type;
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a type trait for getting the n'th argument of a parameter pack.
            In particular, nth_type<n, some_types...>::type is the nth type in some_types.
    !*/

    template<std::size_t I, class... Ts>
    struct nth_type<I, types_<Ts...>> : details::nth_type<I,Ts...> {};

    template<std::size_t I, class... Ts>
    struct nth_type : details::nth_type<I,Ts...> {};

    template<std::size_t I, class... Ts>
    using nth_type_t = typename nth_type<I,Ts...>::type;

// ----------------------------------------------------------------------------------------

    namespace details
    {
        template<class AlwaysVoid, class F>
        struct callable_traits
        {
            constexpr static bool is_callable = false;
        };

        template<class AlwaysVoid, class R, class... Args>
        struct callable_traits<AlwaysVoid, R(Args...)>
        {
            using return_type = R;
            using args        = types_<Args...>;
            constexpr static std::size_t    nargs       = sizeof...(Args);
            constexpr static bool           is_callable = true;
        }; 

        template<class AlwaysVoid, class R, class... Args>
        struct callable_traits<AlwaysVoid, R(*)(Args...)> 
        : public callable_traits<AlwaysVoid, R(Args...)>{};

        template<class AlwaysVoid, class C, class R, class... Args>
        struct callable_traits<AlwaysVoid, R(C::*)(Args...)> 
        : public callable_traits<AlwaysVoid, R(Args...)>{};

        template<class AlwaysVoid, class C, class R, class... Args>
        struct callable_traits<AlwaysVoid, R(C::*)(Args...) const> 
        : public callable_traits<AlwaysVoid, R(Args...)>{};

        template<class F>
        struct callable_traits<void_t<decltype(&std::decay_t<F>::operator())>, F>
        : public callable_traits<void, decltype(&std::decay_t<F>::operator())>{};
    }
   
    template<class F>
    struct callable_traits : details::callable_traits<void, F> {};
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a type trait for callable types.

            If the template parameter F is function pointer, functor or lambda then
            it provides the following types:
                return_type : the return type of the callable object
                args        : a parameter pack packaged in a types_<> meta container containing
                              all the function argument types
            It also provides the following static members:
                nargs       : the number of function arguments
                is_callable : a boolean which determines whether F is callable. In this case, it is true
            
            If the template parameter F is not function-like object, then it provides:
                is_callable : false
            
            For example, a function type F with signature R(T1, T2, T3) has the following traits:
                callable_traits<F>::return_type == R
                callable_traits<F>::args        == types_<T1,T2,T3>
                callable_traits<F>::nargs       == 3
                callable_traits<F>::is_callable == true

            Another example:
                callable_traits<int>::is_callable == false
                callable_traits<int>::return_type == does not exist. Compile error
                callable_traits<int>::args        == does not exist. Compile error
                callable_traits<int>::nargs       == does not exist. Compile error
    !*/

    template<class Callable>
    using callable_args = typename callable_traits<Callable>::args;

    template<std::size_t I, class Callable>
    using callable_arg = nth_type_t<I, callable_args<Callable>>;
    
    template<class Callable>
    using callable_nargs = std::integral_constant<std::size_t, callable_traits<Callable>::nargs>;

    template<class Callable>
    using callable_return = typename callable_traits<Callable>::return_type;

    template<class F>
    using is_callable = std::integral_constant<bool, callable_traits<F>::is_callable>;

// ----------------------------------------------------------------------------------------

}

#endif //DLIB_TYPE_TRAITS_H_
