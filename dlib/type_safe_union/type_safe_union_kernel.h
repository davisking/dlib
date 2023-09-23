// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_SAFE_UNIOn_h_ 
#define DLIB_TYPE_SAFE_UNIOn_h_

#include "type_safe_union_kernel_abstract.h"
#include <new>
#include <iostream>
#include <functional>
#include "../serialize.h"
#include "../type_traits.h"
#include "../overloaded.h"

namespace dlib
{
    // ---------------------------------------------------------------------

    class bad_type_safe_union_cast : public std::bad_cast 
    {
    public:
          virtual const char * what() const throw()
          {
              return "bad_type_safe_union_cast";
          }
    };

    // ---------------------------------------------------------------------

    template<typename T>
    struct in_place_tag { using type = T;};

    // ---------------------------------------------------------------------

    template <typename... Types> class type_safe_union;

    template<typename Tsu>
    struct type_safe_union_size;

    template<typename... Types>
    struct type_safe_union_size<type_safe_union<Types...>> : std::integral_constant<size_t, sizeof...(Types)> {};

    template<typename Tsu> struct type_safe_union_size<const Tsu>           : type_safe_union_size<Tsu> {};
    template<typename Tsu> struct type_safe_union_size<volatile Tsu>        : type_safe_union_size<Tsu> {};
    template<typename Tsu> struct type_safe_union_size<const volatile Tsu>  : type_safe_union_size<Tsu> {};

    // ---------------------------------------------------------------------

    template <size_t I, typename TSU>
    struct type_safe_union_alternative;

    template <size_t I, typename... Types>
    struct type_safe_union_alternative<I, type_safe_union<Types...>> : nth_type<I, Types...>{};

    template<size_t I, typename TSU>
    using type_safe_union_alternative_t = typename type_safe_union_alternative<I, TSU>::type;

    template <size_t I, typename TSU>
    struct type_safe_union_alternative<I, const TSU>
    { using type = typename std::add_const<type_safe_union_alternative_t<I, TSU>>::type; };

    template <size_t I, typename TSU>
    struct type_safe_union_alternative<I, volatile TSU>
    { using type = typename std::add_volatile<type_safe_union_alternative_t<I, TSU>>::type; };

    template <size_t I, typename TSU>
    struct type_safe_union_alternative<I, const volatile TSU>
    { using type = typename std::add_cv<type_safe_union_alternative_t<I, TSU>>::type; };

    // ---------------------------------------------------------------------

    namespace detail
    {
        // ---------------------------------------------------------------------

        template <int nTs, typename T, typename... Ts>
        struct type_safe_union_type_id_impl
                : std::integral_constant<int, -1 - nTs> {};

        template <int nTs, typename T, typename T0, typename... Ts>
        struct type_safe_union_type_id_impl<nTs, T, T0, Ts...>
                : std::integral_constant<int, std::is_same<T,T0>::value ? 1 : type_safe_union_type_id_impl<nTs, T,Ts...>::value + 1> {};

        template <typename T, typename... Ts>
        struct type_safe_union_type_id : type_safe_union_type_id_impl<sizeof...(Ts),T,Ts...>{};

        template <typename T, typename... Ts>
        struct type_safe_union_type_id<in_place_tag<T>, Ts...> : type_safe_union_type_id<T,Ts...>{};

        // ---------------------------------------------------------------------
    }

    template <typename... Types>
    class type_safe_union
    {
        /*!
            CONVENTION
                - is_empty() ==  (type_identity == 0)
                - contains<T>() == (type_identity == get_type_id<T>())
                - mem == the aligned block of memory on the stack which is
                  where objects in the union are stored
        !*/
    public:
        template <typename T>
        static constexpr int get_type_id ()
        {
            return detail::type_safe_union_type_id<T,Types...>::value;
        }

        template <typename T>
        static constexpr int get_type_id (in_place_tag<T>)
        {
            return get_type_id<T>();
        }

    private:

        template<typename T>
        using is_valid_check = std::enable_if_t<is_any_type<T,Types...>::value, bool>;

        template <size_t I>
        using get_type_t = type_safe_union_alternative_t<I, type_safe_union>;

        typename std::aligned_union<0, Types...>::type mem;
        int type_identity = 0;

        template<typename F, typename TSU>
        struct dispatcher
        {
            constexpr static const std::size_t N = sizeof...(Types);
            using R = decltype(std::declval<F>()(std::declval<TSU>().template unchecked_get<get_type_t<0>>()));

            constexpr static const bool is_noexcept =
                And<std::is_default_constructible<R>::value &&
                    noexcept(std::declval<F>()(std::declval<TSU>().template unchecked_get<Types>()))...>::value;

            template<size_t I, typename std::enable_if<I == N, bool>::type = true>
            inline R operator()(F&&, TSU&&, size_<I>)
            noexcept(is_noexcept) { return R(); }

            template<size_t I, typename std::enable_if<I < N, bool>::type = true>
            inline R operator()(F&& f, TSU&& me, size_<I>)
            noexcept(is_noexcept)
            {
                if (me.is_empty())
                    return R();
                else if (me.get_current_type_id() == (I+1))
                    return std::forward<F>(f)(me.template unchecked_get<get_type_t<I>>());
                else
                    return (*this)(std::forward<F>(f), std::forward<TSU>(me), size_<I+1>{});
            }
        };

        template<typename F, typename TSU>
        static inline decltype(auto) dispatch(F&& f, TSU&& me)
        noexcept(noexcept(dispatcher<F&&,TSU&&>{}(std::forward<F>(f), std::forward<TSU>(me), size_<0>{}))) {
            return dispatcher<F&&,TSU&&>{}(std::forward<F>(f), std::forward<TSU>(me), size_<0>{});
        }

        template <typename T>
        const T& unchecked_get() const noexcept
        {
            return *reinterpret_cast<const T*>(&mem);
        }

        template <typename T>
        T& unchecked_get() noexcept
        {
            return *reinterpret_cast<T*>(&mem);
        }

        struct destruct_helper
        {
            template <typename T>
            void operator() (T& item) const
            {
                item.~T();
            }
        };

        void destruct ()
        {
            apply_to_contents(destruct_helper{});
            type_identity = 0;
        }

        template <typename T, typename... Args>
        void construct (
            Args&&... args
        )
        {
            destruct();
            new(&mem) T(std::forward<Args>(args)...);
            type_identity = get_type_id<T>();
        }

        struct assign_to
        {
            /*!
                This class assigns an object to `me` using std::forward.
            !*/
            assign_to(type_safe_union& me) : _me(me) {}

            template<typename T>
            void operator()(T&& x)
            {
                using U = std::decay_t<T>;

                if (_me.type_identity != get_type_id<U>())
                {
                    _me.construct<U>(std::forward<T>(x));
                }
                else
                {
                    _me.template unchecked_get<U>() = std::forward<T>(x);
                }
            }

            type_safe_union& _me;
        };

        struct move_to
        {
            /*!
                This class move assigns an object to `me`.
            !*/
            move_to(type_safe_union& me) : _me(me) {}

            template<typename T>
            void operator()(T& x)
            {
                if (_me.type_identity != get_type_id<T>())
                {
                    _me.construct<T>(std::move(x));
                }
                else
                {
                    _me.template unchecked_get<T>() = std::move(x);
                }
            }

            type_safe_union& _me;
        };

        struct swap_to
        {
            /*!
                This class swaps an object with `me`.
            !*/
            swap_to(type_safe_union& me) : _me(me) {}

            template<typename T>
            void operator()(T& x)
            /*!
                requires
                    - _me.contains<T>() == true
            !*/
            {
                using std::swap;
                swap(_me.unchecked_get<T>(), x);
            }

            type_safe_union& _me;
        };

    public:

        type_safe_union() = default;

        type_safe_union (
            const type_safe_union& item
        )
        noexcept(are_nothrow_copy_constructible<Types...>::value)
        : type_safe_union()
        {
            item.apply_to_contents(assign_to{*this});
        }

        type_safe_union& operator=(
            const type_safe_union& item
        )
        noexcept(are_nothrow_copy_constructible<Types...>::value &&
                 are_nothrow_copy_assignable<Types...>::value)
        {
            if (item.is_empty())
                destruct();
            else
                item.apply_to_contents(assign_to{*this});
            return *this;
        }

        type_safe_union (
            type_safe_union&& item
        )
        noexcept(are_nothrow_move_constructible<Types...>::value)
        : type_safe_union()
        {
            item.apply_to_contents(move_to{*this});
            item.destruct();
        }

        type_safe_union& operator= (
            type_safe_union&& item
        )
        noexcept(are_nothrow_move_constructible<Types...>::value &&
                 are_nothrow_move_assignable<Types...>::value)
        {
            if (item.is_empty())
            {
                destruct();
            }
            else
            {
                item.apply_to_contents(move_to{*this});
                item.destruct();
            }
            return *this;
        }

        template <
            typename T,
            is_valid_check<std::decay_t<T>> = true
        >
        type_safe_union (
            T&& item
        )
        noexcept(std::is_nothrow_constructible<std::decay_t<T>, T>::value)
        : type_safe_union()
        {
            assign_to{*this}(std::forward<T>(item));
        }

        template <
            typename T,
            is_valid_check<std::decay_t<T>> = true
        >
        type_safe_union& operator= (
            T&& item
        )
        noexcept(std::is_nothrow_constructible<std::decay_t<T>, T>::value &&
                 std::is_nothrow_assignable<std::decay_t<T>, T>::value)
        {
            assign_to{*this}(std::forward<T>(item));
            return *this;
        }

        template <
            typename T,
            typename... Args,
            is_valid_check<T> = true
        >
        type_safe_union (
            in_place_tag<T>,
            Args&&... args
        )
        noexcept(std::is_nothrow_constructible<T, Args...>::value)
        : type_safe_union()
        {
            construct<T>(std::forward<Args>(args)...);
        }

        ~type_safe_union()
        {
            destruct();
        }

        void clear()
        {
            destruct();
        }

        template <
            typename T,
            typename... Args,
            is_valid_check<T> = true
        >
        void emplace(
            Args&&... args
        )
        noexcept(std::is_nothrow_constructible<T, Args...>::value)
        {
            construct<T>(std::forward<Args>(args)...);
        }

        template <typename F>
        decltype(auto) apply_to_contents(
            F&& f
        ) noexcept(noexcept(dispatch(std::forward<F>(f), std::declval<type_safe_union&>()))) {
            return dispatch(std::forward<F>(f), *this);
        }

        template <typename F>
        decltype(auto) apply_to_contents(
            F&& f
        ) const noexcept(noexcept(dispatch(std::forward<F>(f), std::declval<const type_safe_union&>()))) {
            return dispatch(std::forward<F>(f), *this);
        }

        template <typename T>
        bool contains (
        ) const noexcept
        {
            return type_identity == get_type_id<T>();
        }

        bool is_empty (
        ) const noexcept
        {
            return type_identity == 0;
        }

        int get_current_type_id() const noexcept
        {
            return type_identity;
        }

        template <
            typename T,
            is_valid_check<T> = true
        >
        T& get(
        )
        noexcept(std::is_nothrow_default_constructible<T>::value)
        {
            if (type_identity != get_type_id<T>())
                construct<T>();
            return unchecked_get<T>();
        }

        template <
            typename T
        >
        T& get(
            in_place_tag<T>
        )
        noexcept(std::is_nothrow_default_constructible<T>::value)
        {
            return get<T>();
        }

        template <
            typename T,
            is_valid_check<T> = true
        >
        const T& cast_to (
        ) const
        {
            if (contains<T>())
                return unchecked_get<T>();
            else
                throw bad_type_safe_union_cast();
        }

        template <
            typename T,
            is_valid_check<T> = true
        >
        T& cast_to (
        )
        {
            if (contains<T>())
                return unchecked_get<T>();
            else
                throw bad_type_safe_union_cast();
        }

        void swap(
            type_safe_union& item
        ) noexcept(std::is_nothrow_move_constructible<type_safe_union>::value &&
                   are_nothrow_swappable<Types...>::value)
        {
            if (type_identity == item.type_identity)
            {
                apply_to_contents(swap_to{item});
            }
            else if (is_empty())
            {
                *this = std::move(item);
            }
            else if (item.is_empty())
            {
                item = std::move(*this);
            }
            else
            {
                type_safe_union tmp{std::move(*this)};
                *this = std::move(item);
                item  = std::move(tmp);
            }
        }
    };

    template <typename ...Types>
    inline void swap (
        type_safe_union<Types...>& a,
        type_safe_union<Types...>& b
    ) noexcept(noexcept(a.swap(b)))
    { a.swap(b); }

    namespace detail
    {
        template<
            typename F,
            typename TSU,
            std::size_t... I
        >
        void for_each_type_impl(
            F&& f,
            TSU&& tsu,
            std::index_sequence<I...>
        )
        {
            using Tsu = std::decay_t<TSU>;

#ifdef __cpp_fold_expressions
            (std::forward<F>(f)(
                in_place_tag<type_safe_union_alternative_t<I, Tsu>>{},
                std::forward<TSU>(tsu)),
            ...);
#else
            (void)std::initializer_list<int>{
                (std::forward<F>(f)(
                        in_place_tag<type_safe_union_alternative_t<I, Tsu>>{},
                        std::forward<TSU>(tsu)),
                 0
                )...
            };
#endif            
        }
    }

    template<
        typename TSU,
        typename F
    >
    void for_each_type(
        F&& f,
        TSU&& tsu
    )
    {
        using Tsu = std::decay_t<TSU>;
        static constexpr std::size_t Size = type_safe_union_size<Tsu>::value;
        detail::for_each_type_impl(std::forward<F>(f), std::forward<TSU>(tsu), std::make_index_sequence<Size>{});
    }

    template<typename F, typename TSU>
    decltype(auto) visit(
        F&& f,
        TSU&& tsu
    ) noexcept(noexcept(tsu.apply_to_contents(std::forward<F>(f)))) {
        return tsu.apply_to_contents(std::forward<F>(f));
    }

    template<typename... Types>
    inline void serialize (
        const type_safe_union<Types...>& item,
        std::ostream& out
    )
    {
        try
        {
            serialize(item.get_current_type_id(), out);
            item.apply_to_contents([&](auto&& x) {
                serialize(x, out);
            });
        }
        catch (serialization_error& e)
        {
            throw serialization_error(e.info + "\n   while serializing an object of type type_safe_union");
        }
    }

    template<typename... Types>
    inline void deserialize (
        type_safe_union<Types...>& item,
        std::istream& in
    )
    {
        try
        {
            int index = -1;
            deserialize(index, in);

            if (index == 0)
                item.clear();
            else if (index > 0 && index <= (int)sizeof...(Types))
                for_each_type([&](auto tag, auto&& me) {
                    if (index == me.get_type_id(tag))
                        deserialize(me.get(tag), in);
                }, item);
            else
                throw serialization_error("bad index value. Should be in range [0,sizeof...(Types))");
        }
        catch(serialization_error& e)
        {
            throw serialization_error(e.info + "\n   while deserializing an object of type type_safe_union");
        }
    }
}

#endif // DLIB_TYPE_SAFE_UNIOn_h_
