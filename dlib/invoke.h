// Copyright (C) 2021  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_INVOKE_Hh_
#define DLIB_INVOKE_Hh_

#include <functional>
#include <type_traits>
#include "utility.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace detail 
    {
        template< typename T >
        struct is_reference_wrapper : std::false_type {};
        template< typename U >
        struct is_reference_wrapper<std::reference_wrapper<U>> : std::true_type {};

        template <
            typename Base,
            typename T,
            typename Derived,
            typename... Args
        >
        constexpr auto invoke_(
            T Base::*pmf, //pointer to member function
            Derived&& ref,
            Args&&... args
        )
        noexcept(noexcept((std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...)))
        -> typename std::enable_if<
                std::is_function<T>::value &&
                std::is_base_of<Base, typename std::decay<Derived>::type>::value,
                decltype((std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...))
           >::type
        {
            return (std::forward<Derived>(ref).*pmf)(std::forward<Args>(args)...);
        }

        template<
            typename Base,
            typename T,
            typename RefWrap,
            typename... Args
        >
        constexpr auto invoke_(
            T Base::*pmf, //pointer to member function
            RefWrap&& ref,
            Args&&... args
        )
        noexcept(noexcept((ref.get().*pmf)(std::forward<Args>(args)...)))
        -> typename std::enable_if<
                std::is_function<T>::value &&
                is_reference_wrapper<typename std::decay<RefWrap>::type>::value,
                decltype((ref.get().*pmf)(std::forward<Args>(args)...))>::type
        {
            return (ref.get().*pmf)(std::forward<Args>(args)...);
        }

        template<
            typename Base,
            typename T,
            typename Ptr,
            typename... Args
        >
        constexpr auto invoke_(
            T Base::*pmf, //pointer to member function
            Ptr&& ptr,
            Args&&... args
        )
        noexcept(noexcept(((*std::forward<Ptr>(ptr)).*pmf)( std::forward<Args>( args )...)))
        -> typename std::enable_if<
                std::is_function<T>::value &&
                !is_reference_wrapper<typename std::decay<Ptr>::type>::value &&
                !std::is_base_of<Base, typename std::decay<Ptr>::type>::value,
                decltype(((*std::forward<Ptr>(ptr)).*pmf)(std::forward<Args>(args)...))>::type
        {
            return ((*std::forward<Ptr>(ptr)).*pmf)( std::forward<Args>( args )...);
        }

        template<
            typename Base,
            typename T,
            typename Derived
        >
        constexpr auto invoke_(
            T Base::*pmd, //pointer to member data
            Derived&& ref
        )
        noexcept(noexcept(std::forward<Derived>(ref).*pmd))
        -> typename std::enable_if<
                std::is_object<T>::value &&
                std::is_base_of<Base, typename std::decay<Derived>::type>::value,
                decltype(std::forward<Derived>(ref).*pmd)>::type
        {
            return std::forward<Derived>(ref).*pmd;
        }

        template<
            typename Base,
            typename T,
            typename RefWrap
        >
        constexpr auto invoke_(
            T Base::*pmd, //pointer to member data
            RefWrap&& ref
        )
        noexcept(noexcept(ref.get().*pmd))
        -> typename std::enable_if<
                std::is_object<T>::value &&
                is_reference_wrapper<typename std::decay<RefWrap>::type>::value,
                decltype(ref.get().*pmd)>::type
        {
            return ref.get().*pmd;
        }

        template<
            typename Base,
            typename T,
            typename Ptr
        >
        constexpr auto invoke_(
            T Base::*pmd, //pointer to member data
            Ptr&& ptr
        )
        noexcept(noexcept((*std::forward<Ptr>(ptr)).*pmd))
        -> typename std::enable_if<
                std::is_object<T>::value &&
                !is_reference_wrapper<typename std::decay<Ptr>::type>::value &&
                !std::is_base_of<Base, typename std::decay<Ptr>::type>::value,
                decltype((*std::forward<Ptr>(ptr)).*pmd)>::type
        {
            return (*std::forward<Ptr>(ptr)).*pmd;
        }

        template<
            typename F,
            typename... Args
        >
        constexpr auto invoke_(
            F && f,
            Args&&... args
        )
        noexcept(noexcept(std::forward<F>( f )( std::forward<Args>( args )...)))
        -> typename std::enable_if<
                !std::is_member_pointer<typename std::decay<F>::type>::value,
                decltype(std::forward<F>(f)(std::forward<Args>(args)...))>::type
        {
            return std::forward<F>( f )( std::forward<Args>( args )...);
        }
    } // end namespace detail

// ----------------------------------------------------------------------------------------
    
    template< typename F, typename... Args>
    constexpr auto invoke(F && f, Args &&... args)
    /*!
        ensures
            - identical to std::invoke(std::forward<F>(f), std::forward<Args>(args)...)
            - works with C++11 onwards
    !*/
    noexcept(noexcept(detail::invoke_(std::forward<F>( f ), std::forward<Args>( args )...)))
    -> decltype(detail::invoke_(std::forward<F>( f ), std::forward<Args>( args )...))
    {
        return detail::invoke_(std::forward<F>( f ), std::forward<Args>( args )...);
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
    } // end namespace detail

// ----------------------------------------------------------------------------------------

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

    template <typename R, typename F, typename... Args>
    struct is_invocable_r : std::integral_constant<bool, dlib::is_invocable<F, Args...>::value &&
                                                         std::is_convertible<invoke_result_t<F, Args...>, R>::value> {};
    /*!
        ensures
            - identical to std::is_invocable_r<R, F, Args..>
            - works with C++11 onwards
    !*/

// ----------------------------------------------------------------------------------------

    template< typename R, typename F, typename... Args>
    constexpr typename std::enable_if<dlib::is_invocable_r<R, F, Args...>::value, R>::type
    invoke_r(F && f, Args &&... args)
    /*!
        ensures
            - identical to std::invoke_r<R>(std::forward<F>(f), std::forward<Args>(args)...)
            - works with C++11 onwards
    !*/
    noexcept(noexcept(dlib::invoke(std::forward<F>( f ), std::forward<Args>( args )...)))
    {
        return dlib::invoke(std::forward<F>( f ), std::forward<Args>( args )...);
    }

// ----------------------------------------------------------------------------------------

    namespace detail
    {
        template<typename F, typename Tuple, std::size_t... I>
        constexpr auto apply_impl(F&& fn, Tuple&& tpl, index_sequence<I...>)
        noexcept(noexcept(dlib::invoke(std::forward<F>(fn),
                                       std::get<I>(std::forward<Tuple>(tpl))...)))
        -> decltype(dlib::invoke(std::forward<F>(fn),
                                 std::get<I>(std::forward<Tuple>(tpl))...))
        {
            return dlib::invoke(std::forward<F>(fn),
                                std::get<I>(std::forward<Tuple>(tpl))...);
        }
    } // end namespace detail

// ----------------------------------------------------------------------------------------

    template<typename F, typename Tuple>
    constexpr auto apply(F&& fn, Tuple&& tpl)
    /*!
        ensures
            - identical to std::apply(std::forward<F>(f), std::forward<Tuple>(tpl))
            - works with C++11 onwards
    !*/
    noexcept(noexcept(detail::apply_impl(std::forward<F>(fn),
                                         std::forward<Tuple>(tpl),
                                         make_index_sequence<std::tuple_size<typename std::remove_reference<Tuple>::type >::value>{})))
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
    } // end namespace detail

// ----------------------------------------------------------------------------------------

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
