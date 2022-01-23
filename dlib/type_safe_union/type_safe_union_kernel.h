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

namespace dlib
{
    class bad_type_safe_union_cast : public std::bad_cast 
    {
    public:
          virtual const char * what() const throw()
          {
              return "bad_type_safe_union_cast";
          }
    };

    template<typename T>
    struct in_place_tag {};

    namespace internal
    {
        // ---------------------------------------------------------------------
        template <typename T, typename... Rest>
        struct is_any : std::false_type {};

        template <typename T, typename First>
        struct is_any<T,First> : std::is_same<T,First> {};

        template <typename T, typename First, typename... Rest>
        struct is_any<T,First,Rest...> : std::integral_constant<bool, std::is_same<T,First>::value || is_any<T,Rest...>::value> {};

        // ---------------------------------------------------------------------
        namespace detail
        {
            struct empty {};

            template<typename T>
            struct variant_type_placeholder {using type = T;};

            template <
                    size_t nVariantTypes,
                    size_t Counter,
                    size_t I,
                    typename... VariantTypes
            >
            struct variant_get_type_impl {};

            template <
                    size_t nVariantTypes,
                    size_t Counter,
                    size_t I,
                    typename VariantTypeFirst,
                    typename... VariantTypeRest
            >
            struct variant_get_type_impl<nVariantTypes, Counter, I, VariantTypeFirst, VariantTypeRest...>
                    :   std::conditional<(Counter < nVariantTypes),
                            typename std::conditional<(I == Counter),
                                    variant_type_placeholder<VariantTypeFirst>,
                                    variant_get_type_impl<nVariantTypes, Counter+1, I, VariantTypeRest...>
                            >::type,
                            empty
                    >::type {};
        }

        template <size_t I, typename... VariantTypes>
        struct variant_get_type : detail::variant_get_type_impl<sizeof...(VariantTypes), 0, I, VariantTypes...> {};

        // ---------------------------------------------------------------------
        namespace detail
        {
            template <
                    size_t nVariantTypes,
                    size_t Counter,
                    typename T,
                    typename... VariantTypes
            >
            struct variant_type_id_impl : std::integral_constant<int,-1> {};

            template <
                    size_t nVariantTypes,
                    size_t Counter,
                    typename T,
                    typename VariantTypeFirst,
                    typename... VariantTypesRest
            >
            struct variant_type_id_impl<nVariantTypes,Counter,T,VariantTypeFirst,VariantTypesRest...>
                    :   std::conditional<(Counter < nVariantTypes),
                            typename std::conditional<std::is_same<T,VariantTypeFirst>::value,
                                    std::integral_constant<int,Counter+1>,
                                    variant_type_id_impl<nVariantTypes,Counter + 1, T, VariantTypesRest...>
                            >::type,
                            std::integral_constant<int,-1>
                    >::type {};
        }

        template <typename T, typename... VariantTypes>
        struct variant_type_id : detail::variant_type_id_impl<sizeof...(VariantTypes), 0, T, VariantTypes...> {};
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
            return internal::variant_type_id<T,Types...>::value;
        }

    private:
        template<typename T>
        struct is_valid : internal::is_any<T,Types...> {};

        template<typename T>
        using is_valid_check = typename std::enable_if<is_valid<T>::value, bool>::type;

        template <size_t I>
        struct get_type : internal::variant_get_type<I,Types...> {};

        template <size_t I>
        using get_type_t = typename get_type<I>::type;

        using T0 = get_type_t<0>;

        template<typename F, typename T>
        struct return_type
        {
            using type = decltype(std::declval<F>()(std::declval<T>()));
        };

        template<typename F, typename T>
        using return_type_t = typename return_type<F,T>::type;

        typename std::aligned_union<0, Types...>::type mem;
        int type_identity = 0;

        template<
            size_t I,
            typename F
        >
        auto visit_impl(
            F&&
        ) -> typename std::enable_if<
                (I == sizeof...(Types)) &&
                std::is_same<void, return_type_t<F, T0&>>::value
        >::type
        {
        }

        template<
            size_t I,
            typename F
        >
        auto visit_impl(
            F&&
        ) -> typename std::enable_if<
                (I == sizeof...(Types)) &&
                ! std::is_same<void, return_type_t<F, T0&>>::value,
                return_type_t<F, T0&>
        >::type
        {
            return return_type_t<F, T0&>{};
        }

        template<
            size_t I,
            typename F
        >
        auto visit_impl(
            F&& f
        )  -> typename std::enable_if<
                (I < sizeof...(Types)),
                return_type_t<F, T0&>
        >::type
        {
            if (type_identity == (I+1))
                return std::forward<F>(f)(unchecked_get<get_type_t<I>>());
            else
                return visit_impl<I+1>(std::forward<F>(f));
        }

        template<
            size_t I,
            typename F
        >
        auto visit_impl(
            F&&
        ) const -> typename std::enable_if<
                (I == sizeof...(Types)) &&
                std::is_same<void, return_type_t<F, const T0&>>::value
        >::type
        {
        }

        template<
            size_t I,
            typename F
        >
        auto visit_impl(
            F&&
        ) const -> typename std::enable_if<
                (I == sizeof...(Types)) &&
                ! std::is_same<void, return_type_t<F, const T0&>>::value,
                return_type_t<F, const T0&>
        >::type
        {
            return return_type_t<F, const T0&>{};
        }

        template<
            size_t I,
            typename F
        >
        auto visit_impl(
            F&& f
        ) const -> typename std::enable_if<
                (I < sizeof...(Types)),
                return_type_t<F, const T0&>
        >::type
        {
            if (type_identity == (I+1))
                return std::forward<F>(f)(unchecked_get<get_type_t<I>>());
            else
                return visit_impl<I+1>(std::forward<F>(f));
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
            visit(destruct_helper{});
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
            item.visit(assign_to{*this});
        }

        type_safe_union& operator=(
            const type_safe_union& item
        )
        {
            if (item.is_empty())
                destruct();
            else
                item.visit(assign_to{*this});
            return *this;
        }

        type_safe_union (
            type_safe_union&& item
        ) : type_safe_union()
        {
            item.visit(move_to{*this});
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
                item.visit(move_to{*this});
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
            construct<T,Args...>(std::forward<Args>(args)...);
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
            construct<T,Args...>(std::forward<Args>(args)...);
        }

        template <typename F>
        auto visit(
            F&& f
        ) -> decltype(visit_impl<0>(std::forward<F>(f)))
        {
            return visit_impl<0>(std::forward<F>(f));
        }

        template <typename F>
        auto visit(
            F&& f
        ) const -> decltype(visit_impl<0>(std::forward<F>(f)))
        {
            return visit_impl<0>(std::forward<F>(f));
        }

        template <typename F>
        void apply_to_contents(
            F&& f
        )
        {
            visit(std::forward<F>(f));
        }

        template <typename F>
        void apply_to_contents(
            F&& f
        ) const
        {
            visit(std::forward<F>(f));
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
                item.visit(swap_to{*this});
            }
            else if (is_empty())
            {
                item.visit(move_to{*this});
                item.destruct();
            }
            else if (item.is_empty())
            {
                visit(move_to{item});
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

        template<
            size_t I,
            typename... Types
        >
        inline typename std::enable_if<(I == sizeof...(Types))>::type deserialize_helper(
            std::istream&, 
            int, 
            type_safe_union<Types...>&
        )
        {
        }
        
        template<
            size_t I,
            typename... Types
        >
        inline typename std::enable_if<(I < sizeof...(Types))>::type deserialize_helper(
            std::istream& in, 
            int index, 
            type_safe_union<Types...>& x
        )
        {
            using T = typename internal::variant_get_type<I, Types...>::type;
            
            if (index == (I+1))
            {
                deserialize(x.template get<T>(), in);
            }
            else
            {
                deserialize_helper<I+1>(in, index, x);
            }
        }        
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
            item.visit(detail::serialize_helper(out));
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
                detail::deserialize_helper<0>(in, index, item);
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
