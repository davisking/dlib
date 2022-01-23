// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TYPE_SAFE_UNIOn_h_ 
#define DLIB_TYPE_SAFE_UNIOn_h_

#include "type_safe_union_kernel_abstract.h"
#include <new>
#include <iostream>
#include <type_traits>
#include <functional>
#include "../serialize.h"
#include "../invoke.h"

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

    namespace detail
    {
        template<size_t I, typename... Ts>
        struct nth_type;

        template<size_t I, typename T0, typename... Ts>
        struct nth_type<I, T0, Ts...> : nth_type<I-1, Ts...> {};

        template<typename T0, typename... Ts>
        struct nth_type<0, T0, Ts...> { using type = T0; };
    }

    template <size_t I, typename TSU>
    struct type_safe_union_alternative;

    template <size_t I, typename... Types>
    struct type_safe_union_alternative<I, type_safe_union<Types...>> : detail::nth_type<I, Types...>{};

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

        template <typename T, typename First, typename... Rest>
        struct is_any : std::integral_constant<bool, is_any<T,First>::value || is_any<T,Rest...>::value> {};

        template <typename T, typename First>
        struct is_any<T,First> : std::is_same<T,First> {};

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

    private:
        template<typename T>
        struct is_valid : detail::is_any<T,Types...> {};

        template<typename T>
        using is_valid_check = typename std::enable_if<is_valid<T>::value, bool>::type;

        template <size_t I>
        using get_type_t = type_safe_union_alternative_t<I, type_safe_union>;

        typename std::aligned_union<0, Types...>::type mem;
        int type_identity = 0;

        template<
            typename F,
            typename TSU,
            std::size_t I
        >
        static void apply_to_contents_as_type(
            F&& f,
            TSU&& me
        )
        {
            std::forward<F>(f)(me.template unchecked_get<get_type_t<I>>());
        }

        template<
            typename F,
            typename TSU,
            std::size_t... I
        >
        static void apply_to_contents_impl(
            F&& f,
            TSU&& me,
            dlib::index_sequence<I...>
        )
        {
            using func_t = void(*)(F&&, TSU&&);

            const func_t vtable[] = {
                /*! Empty (type_identity == 0) case !*/
                [](F&&, TSU&&) {
                },
                /*! Non-empty cases !*/
                &apply_to_contents_as_type<F&&,TSU&&,I>...
            };

            return vtable[me.get_current_type_id()](std::forward<F>(f), std::forward<TSU>(me));
        }

        template <typename T>
        const T& unchecked_get() const
        {
            return *reinterpret_cast<const T*>(&mem);
        }

        template <typename T>
        T& unchecked_get()
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
                using U = typename std::decay<T>::type;

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
        ) : type_safe_union()
        {
            item.apply_to_contents(assign_to{*this});
        }

        type_safe_union& operator=(
            const type_safe_union& item
        )
        {
            if (item.is_empty())
                destruct();
            else
                item.apply_to_contents(assign_to{*this});
            return *this;
        }

        type_safe_union (
            type_safe_union&& item
        ) : type_safe_union()
        {
            item.apply_to_contents(move_to{*this});
            item.destruct();
        }

        type_safe_union& operator= (
            type_safe_union&& item
        )
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
            is_valid_check<typename std::decay<T>::type> = true
        >
        type_safe_union (
            T&& item
        ) : type_safe_union()
        {
            assign_to{*this}(std::forward<T>(item));
        }

        template <
            typename T,
            is_valid_check<typename std::decay<T>::type> = true
        >
        type_safe_union& operator= (
            T&& item
        )
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
        {
            construct<T>(std::forward<Args>(args)...);
        }

        template <typename F>
        void apply_to_contents(
            F&& f
        )
        {
            apply_to_contents_impl(std::forward<F>(f), *this, dlib::make_index_sequence<sizeof...(Types)>{});
        }

        template <typename F>
        void apply_to_contents(
            F&& f
        ) const
        {
            apply_to_contents_impl(std::forward<F>(f), *this, dlib::make_index_sequence<sizeof...(Types)>{});
        }

        template <typename T>
        bool contains (
        ) const
        {
            return type_identity == get_type_id<T>();
        }

        bool is_empty (
        ) const
        {
            return type_identity == 0;
        }

        int get_current_type_id() const
        {
            return type_identity;
        }

        void swap (
            type_safe_union& item
        )
        {
            if (type_identity == item.type_identity)
            {
                item.apply_to_contents(swap_to{*this});
            }
            else if (is_empty())
            {
                item.apply_to_contents(move_to{*this});
                item.destruct();
            }
            else if (item.is_empty())
            {
                apply_to_contents(move_to{item});
                destruct();
            }
            else
            {
                type_safe_union tmp;
                swap(tmp);      // this -> tmp
                swap(item);     // item -> this
                tmp.swap(item); // tmp (this) -> item
            }
        }

        template <
            typename T,
            is_valid_check<T> = true
        >
        T& get(
        )
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
    };

    template <typename ...Types>
    inline void swap (
        type_safe_union<Types...>& a,
        type_safe_union<Types...>& b
    ) { a.swap(b); }

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
            dlib::index_sequence<I...>
        )
        {
            using Tsu = typename std::decay<TSU>::type;
            (void)std::initializer_list<int>{
                (std::forward<F>(f)(
                        in_place_tag<type_safe_union_alternative_t<I, Tsu>>{},
                        std::forward<TSU>(tsu)),
                 0
                )...
            };
        }

        template<
            typename R,
            typename F,
            typename TSU,
            std::size_t I
        >
        R visit_impl_as_type(
            F&& f,
            TSU&& tsu
        )
        {
            using Tsu = typename std::decay<TSU>::type;
            using T   = type_safe_union_alternative_t<I, Tsu>;
            return dlib::invoke(std::forward<F>(f), tsu.template cast_to<T>());
        }

        template<
            typename R,
            typename F,
            typename TSU,
            std::size_t... I
        >
        R visit_impl(
            F&& f,
            TSU&& tsu,
            dlib::index_sequence<I...>
        )
        {
            using func_t = R(*)(F&&, TSU&&);

            const func_t vtable[] = {
                /*! Empty (type_identity == 0) case !*/
                [](F&&, TSU&&) {
                    return R();
                },
                /*! Non-empty cases !*/
                &visit_impl_as_type<R,F&&,TSU&&,I>...
            };

            return vtable[tsu.get_current_type_id()](std::forward<F>(f), std::forward<TSU>(tsu));
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
        using Tsu = typename std::decay<TSU>::type;
        static constexpr std::size_t Size = type_safe_union_size<Tsu>::value;
        detail::for_each_type_impl(std::forward<F>(f), std::forward<TSU>(tsu), dlib::make_index_sequence<Size>{});
    }

    template<
        typename F,
        typename TSU,
        typename Tsu = typename std::decay<TSU>::type,
        typename T0  = type_safe_union_alternative_t<0, Tsu>
    >
    auto visit(
        F&& f,
        TSU&& tsu
    ) -> dlib::invoke_result_t<F, decltype(tsu.template cast_to<T0>())>
    {
        using ReturnType = dlib::invoke_result_t<F, decltype(tsu.template cast_to<T0>())>;
        static constexpr std::size_t Size = type_safe_union_size<Tsu>::value;
        return detail::visit_impl<ReturnType>(std::forward<F>(f), std::forward<TSU>(tsu), dlib::make_index_sequence<Size>{});
    }

    namespace detail
    {
        struct serialize_helper
        {
            serialize_helper(std::ostream& out_) : out(out_) {}

            template <typename T>
            void operator() (const T& item) const 
            { 
                serialize(item, out); 
            } 

            std::ostream& out;
        };  

        struct deserialize_helper
        {
            deserialize_helper(
                std::istream& in_,
                int index_
            ) : index(index_),
                in(in_)
            {}

            template<typename T, typename TSU>
            void operator()(in_place_tag<T>, TSU&& x)
            {
                if (index == x.template get_type_id<T>())
                    deserialize(x.template get<T>(), in);
            }

            const int index = -1;
            std::istream& in;
        };
    } // namespace detail

    template<typename... Types>
    inline void serialize (
        const type_safe_union<Types...>& item,
        std::ostream& out
    )
    {
        try
        {
            serialize(item.get_current_type_id(), out);
            item.apply_to_contents(detail::serialize_helper(out));
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
                for_each_type(detail::deserialize_helper(in, index), item);
            else
                throw serialization_error("bad index value. Should be in range [0,sizeof...(Types))");
        }
        catch(serialization_error& e)
        {
            throw serialization_error(e.info + "\n   while deserializing an object of type type_safe_union");
        }
    }

#if __cplusplus >= 201703L

    template<typename ...Base>
    struct overloaded_helper : Base...
    {
        template<typename... T>
        overloaded_helper(T&& ... t) : Base{std::forward<T>(t)}... {}

        using Base::operator()...;
    };

#else

    template<typename Base, typename ... BaseRest>
    struct overloaded_helper: Base, overloaded_helper<BaseRest...>
    {
        template<typename T, typename ... TRest>
        overloaded_helper(T&& t, TRest&& ...trest) :
            Base{std::forward<T>(t)},
            overloaded_helper<BaseRest...>{std::forward<TRest>(trest)...}
        {}

        using Base::operator();
        using overloaded_helper<BaseRest...>::operator();
    };

    template<typename Base>
    struct overloaded_helper<Base> : Base
    {
        template<typename T>
        overloaded_helper<Base>(T&& t) : Base{std::forward<T>(t)}
        {}

        using Base::operator();
    };

#endif //__cplusplus >= 201703L

    template<typename... T>
    overloaded_helper<typename std::decay<T>::type...> overloaded(T&&... t)
    {
        return overloaded_helper<typename std::decay<T>::type...>{std::forward<T>(t)...};
    }
}

#endif // DLIB_TYPE_SAFE_UNIOn_h_
