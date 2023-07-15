// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_expected_H
#define DLIB_expected_H

#include <cstdint>
#include <exception>
#include <initializer_list>
#include "functional.h"

namespace dlib
{

// ---------------------------------------------------------------------------------------------------

    template <class E>
    class unexpected;

// ---------------------------------------------------------------------------------------------------

    namespace expected_details
    {
        template<class E>
        struct is_unexpected_type : std::false_type{};

        template<class E>
        struct is_unexpected_type<unexpected<E>> : std::true_type{};
    }

// ---------------------------------------------------------------------------------------------------

    template <class E> 
    class unexpected 
    {
    public:
        constexpr unexpected()                      = delete;
        constexpr unexpected( const unexpected& )   = default;
        constexpr unexpected( unexpected&& )        = default;
        ~unexpected()                     = default;

        template < 
          class Err = E,
          std::enable_if_t<
            !std::is_same<dlib::remove_cvref_t<Err>, unexpected>::value &&
            !std::is_same<dlib::remove_cvref_t<Err>, dlib::in_place_t>::value &&
            std::is_constructible<E, Err>::value,
            bool> = true
        >
        constexpr explicit unexpected ( 
            Err&& e 
        ) 
        noexcept(std::is_nothrow_constructible<E, Err>::value)
        : v{std::forward<Err>(e)}
        {
        }

        template < 
          class... Args,
          std::enable_if_t<std::is_constructible<E, Args...>::value,bool> = true
        >
        constexpr explicit unexpected ( 
            dlib::in_place_t, 
            Args&&... args 
        ) 
        noexcept(std::is_nothrow_constructible<E, Args...>::value)
        : v{std::forward<Args>(args)...}
        {
        }

        template < 
          class U, 
          class... Args,
          std::enable_if_t<std::is_constructible<E, std::initializer_list<U>&, Args...>::value, bool> = true
        >
        constexpr explicit unexpected (
            dlib::in_place_t,
            std::initializer_list<U> il, 
            Args&&... args 
        ) 
        noexcept(std::is_nothrow_constructible<E, std::initializer_list<U>&, Args...>::value)
        : v{il, std::forward<Args>(args)...}
        {
        }

        constexpr const E&  error() const&  noexcept {return v;}
        constexpr E&        error() &       noexcept {return v;}
        constexpr const E&& error() const&& noexcept {return std::move(v);}
        constexpr E&&       error() &&      noexcept {return std::move(v);}

        constexpr void swap( unexpected& other ) noexcept(dlib::is_nothrow_swappable<E>::value)
        {
            using std::swap;
            swap(v, other.v);
        }

    private:

        static_assert(!expected_details::is_unexpected_type<E>::value,   "E cannot be of type unexpected<>");
        static_assert(std::is_object<E>::value,                 "E must be an object type");
        static_assert(!std::is_array<E>::value,                 "E cannot be an array type");
        static_assert(!std::is_reference<E>::value,             "E cannot be a reference type");
        E v;
    };

    template< class E1, class E2 >
    constexpr bool operator==(unexpected<E1>& x, unexpected<E2>& y)
    {
        return x.error() == y.error();
    }

    template < 
      class E,
      std::enable_if_t<dlib::is_swappable<E>::value, bool> = true
    >
    constexpr void swap(unexpected<E>& x, unexpected<E>& y ) noexcept(noexcept(x.swap(y)))
    {
        x.swap(y);
    }

#ifdef __cpp_deduction_guides
    template <class E> 
    unexpected(E) -> unexpected<E>;
#endif

// ---------------------------------------------------------------------------------------------------

    template <class E> 
    class bad_expected_access : public std::exception 
    {
    public:
        explicit bad_expected_access(E e) : err(std::move(e)) {}

        virtual const char *what() const noexcept override {
            return "Bad expected access";
        }

        constexpr const E&  error() const &     { return err; }
        constexpr E&        error() &           { return err; }
        constexpr const E&& error() const &&    { return std::move(err); }
        constexpr E&&       error() &&          { return std::move(err); }

    private:
        E err;
    };

// ---------------------------------------------------------------------------------------------------

    struct unexpect_t 
    {
        explicit unexpect_t() = default;
    };

    static constexpr unexpect_t unexpect{};

// ---------------------------------------------------------------------------------------------------

    template<class T, class E>
    class expected;

// ---------------------------------------------------------------------------------------------------

    namespace expected_details
    {
        template<class T>
        struct is_expected_type : std::false_type{};

        template<class T, class E>
        struct is_expected_type<expected<T, E>> : std::true_type{};

        template<
          class T, class E,
          class U, class G, 
          class UF, class GF
        >
        using is_constructible = std::enable_if_t<And<
            ((std::is_void<T>::value && std::is_void<U>::value) || std::is_constructible<T, UF>::value),
            std::is_constructible<E, GF>::value,
            !std::is_constructible<T,       dlib::expected<U, G>&>::value,
            !std::is_constructible<T, const dlib::expected<U, G>&>::value,
            !std::is_constructible<T,       dlib::expected<U, G>&&>::value,
            !std::is_constructible<T, const dlib::expected<U, G>&&>::value,
            !std::is_convertible<      dlib::expected<U, G>&, T>::value,
            !std::is_convertible<const dlib::expected<U, G>&, T>::value,
            !std::is_convertible<      dlib::expected<U, G>&&, T>::value,
            !std::is_convertible<const dlib::expected<U, G>&&, T>::value,
            !std::is_constructible<dlib::unexpected<E>,       dlib::expected<U, G>&>::value,
            !std::is_constructible<dlib::unexpected<E>, const dlib::expected<U, G>&>::value,
            !std::is_constructible<dlib::unexpected<E>,       dlib::expected<U, G>&&>::value,
            !std::is_constructible<dlib::unexpected<E>, const dlib::expected<U, G>&&>::value 
        >::value,
        bool>;

        template <
          class T, class E,
          class U
        >
        using is_convert_constructible = std::enable_if_t<And<
            std::is_constructible<T, U>::value,
            !std::is_same<dlib::remove_cvref_t<U>, in_place_t>::value,
            !std::is_same<expected<T, E>, dlib::remove_cvref_t<U>>::value,
            !std::is_same<unexpected<E>, dlib::remove_cvref_t<U>>::value
        >::value,
        bool>;

// ---------------------------------------------------------------------------------------------------

        struct empty_initialization_tag{};

// ---------------------------------------------------------------------------------------------------

        static constexpr uint8_t IS_EMPTY{0};   // Technically expected<T,E> can never be empty, but one of the base classes can.
        static constexpr uint8_t IS_VAL{1};
        static constexpr uint8_t IS_ERROR{2};

// ---------------------------------------------------------------------------------------------------

        template <class T, class E>
        struct expected_base
        {
            constexpr expected_base(empty_initialization_tag) noexcept
            : e{}, state{IS_EMPTY}
            {}

            constexpr expected_base() noexcept(std::is_nothrow_default_constructible<T>::value)
            : val{}, state{IS_VAL}
            {}

            template<class ...U>
            constexpr expected_base(in_place_t, U&& ...u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            : val{std::forward<U>(u)...}, state{IS_VAL}
            {}

            template<class ...U>
            constexpr expected_base(unexpect_t, U&& ...u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            : error{std::forward<U>(u)...}, state{IS_ERROR} 
            {}    

            template <class... U> 
            constexpr void construct(U&&... u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            {
                new (std::addressof(val)) T(std::forward<U>(u)...);
                state = IS_VAL;
            }

            template <class... U> 
            constexpr void construct_error(U&&... u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            {
                new (std::addressof(error)) T(std::forward<U>(u)...);
                state = IS_ERROR;
            }

            constexpr void destruct() noexcept(std::is_nothrow_destructible<T>::value &&
                                               std::is_nothrow_destructible<E>::value)
            {
                switch(state)
                {
                    case IS_VAL:    val.~T();               break;
                    case IS_ERROR:  error.~unexpected<E>(); break;
                    default:                                break;
                }
            }    

            template<class Expected>
            constexpr void assign(Expected&& rhs)
            {
                /* This is screaming for pattern matching. */

                const uint8_t combined = state | (rhs.state << 4);
                
                switch(combined)
                {
                    case IS_VAL | (IS_VAL << 4): 
                        this->val = std::forward<Expected>(rhs).val; 
                        break;
                    
                    case IS_VAL | (IS_EMPTY << 4):
                        destruct();
                        break;
                    
                    case IS_VAL | (IS_ERROR << 4):
                        destruct();
                        construct_error(std::forward<Expected>(rhs).error);
                        break;
                    
                    case IS_EMPTY | (IS_VAL << 4): 
                        construct(std::forward<Expected>(rhs).val);
                        break;
                    
                    case IS_EMPTY | (IS_ERROR << 4): 
                        construct_error(std::forward<Expected>(rhs).error);
                        break;

                    case IS_ERROR | (IS_VAL << 4): 
                        destruct();
                        construct(std::forward<Expected>(rhs).val);
                        break;

                    case IS_ERROR | (IS_ERROR << 4): 
                        error = std::forward<Expected>(rhs).error; 
                        break;

                    case IS_ERROR | (IS_EMPTY << 4): 
                        destruct();
                        break;

                    default:
                        break;
                }
            }   

            struct empty{};
            union {T val; unexpected<E> error; empty e;};
            uint8_t state{IS_EMPTY};
        };

        template <
          class T,
          class E,
          bool = (std::is_trivially_destructible<T>::value && std::is_trivially_destructible<E>::value)
        >
        struct expected_destructor : expected_base<T,E>
        {
            using expected_base<T,E>::expected_base;

            ~expected_destructor() noexcept
            {
                this->destruct();
            }  
        };

        template <class T, class E>
        struct expected_destructor<T, E, true> : expected_base<T,E>
        {
            using expected_base<T,E>::expected_base;
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          class E,
          bool = (std::is_trivially_copy_constructible<T>::value && std::is_trivially_copy_constructible<E>::value)
        >
        struct expected_copy : expected_destructor<T,E> 
        {
            using expected_destructor<T,E>::expected_destructor;
        };

        template <class T, class E>
        struct expected_copy<T, E, false> : expected_destructor<T,E> 
        {
            using expected_destructor<T,E>::expected_destructor;

            constexpr expected_copy()                                       = default;
            constexpr expected_copy(expected_copy&& rhs)                    = default;
            constexpr expected_copy &operator=(const expected_copy& rhs)    = default;
            constexpr expected_copy &operator=(expected_copy&& rhs)         = default;

            constexpr expected_copy(const expected_copy& rhs) noexcept(std::is_nothrow_copy_constructible<T>::value &&
                                                                       std::is_nothrow_copy_constructible<E>::value)
            : expected_destructor<T, E>(empty_initialization_tag{})
            {
                switch(rhs.state)
                {
                    case IS_VAL:    this->construct(rhs.val);           break;
                    case IS_ERROR:  this->construct_error(rhs.error);   break;
                    default:                                            break;
                }
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          class E,
          bool = (std::is_trivially_move_constructible<T>::value && std::is_trivially_move_constructible<E>::value)
        >
        struct expected_move : expected_copy<T, E> 
        {
            using expected_copy<T, E>::expected_copy;
        };

        template <class T, class E> 
        struct expected_move<T, E, false> : expected_copy<T, E> 
        {
            using expected_copy<T, E>::expected_copy;

            constexpr expected_move()                                       = default;
            constexpr expected_move(const expected_move& rhs)               = default;
            constexpr expected_move& operator=(const expected_move& rhs)    = default;
            constexpr expected_move& operator=(expected_move&& rhs)         = default;

            constexpr expected_move(expected_move&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value &&
                                                                  std::is_nothrow_move_constructible<E>::value)
            : expected_copy<T, E>(empty_initialization_tag{})
            {
                switch(rhs.state)
                {
                    case IS_VAL:    this->construct(std::move(rhs.val));            break;
                    case IS_ERROR:  this->construct_error(std::move(rhs.error));    break;
                    default:                                                        break;
                }
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool = std::is_trivially_copy_assignable<T>::value    &&
                 std::is_trivially_copy_constructible<T>::value &&
                 std::is_trivially_destructible<T>::value       &&
                 std::is_trivially_copy_assignable<E>::value    &&
                 std::is_trivially_copy_constructible<E>::value &&
                 std::is_trivially_destructible<E>::value
        >
        struct expected_copy_assign : expected_move<T, E> 
        {
            using expected_move<T, E>::expected_move;
        };

        template <class T, class E>
        struct expected_copy_assign<T, E, false> : expected_move<T, E> 
        {
            using expected_move<T, E>::expected_move;

            constexpr expected_copy_assign()                                        = default;
            constexpr expected_copy_assign(const expected_copy_assign& rhs)         = default;
            constexpr expected_copy_assign(expected_copy_assign&& rhs)              = default;
            constexpr expected_copy_assign& operator=(expected_copy_assign &&rhs)   = default;

            constexpr expected_copy_assign& operator=(const expected_copy_assign &rhs) 
            noexcept(std::is_nothrow_copy_constructible<T>::value   && 
                     std::is_nothrow_copy_assignable<T>::value      &&
                     std::is_nothrow_copy_constructible<E>::value   && 
                     std::is_nothrow_copy_assignable<E>::value)
            {
                this->assign(rhs);
                return *this;
            }        
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool = std::is_trivially_move_assignable<T>::value    &&
                 std::is_trivially_move_constructible<T>::value &&
                 std::is_trivially_destructible<T>::value       &&
                 std::is_trivially_move_assignable<E>::value    &&
                 std::is_trivially_move_constructible<E>::value &&
                 std::is_trivially_destructible<E>::value
        >
        struct expected_move_assign : expected_copy_assign<T,E> 
        {
            using expected_copy_assign<T,E>::expected_copy_assign;
        };

        template <class T, class E>
        struct expected_move_assign<T, E, false> : expected_copy_assign<T, E> 
        {
            using expected_copy_assign<T,E>::expected_copy_assign;

            constexpr expected_move_assign()                                              = default;
            constexpr expected_move_assign(const expected_move_assign &rhs)               = default;
            constexpr expected_move_assign(expected_move_assign &&rhs)                    = default;
            constexpr expected_move_assign& operator=(const expected_move_assign &rhs)    = default;

            constexpr expected_move_assign& operator=(expected_move_assign &&rhs) 
            noexcept(std::is_nothrow_move_constructible<T>::value   && 
                     std::is_nothrow_move_assignable<T>::value      &&
                     std::is_nothrow_move_constructible<E>::value   && 
                     std::is_nothrow_move_assignable<E>::value)
            {
                this->assign(std::move(rhs));
                return *this;
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool = std::is_default_constructible<T>::value
        >
        struct expected_delete_default_constructor
        {
            constexpr expected_delete_default_constructor()                                                         = default;
            constexpr expected_delete_default_constructor(const expected_delete_default_constructor&)               = default;
            constexpr expected_delete_default_constructor(expected_delete_default_constructor&&)                    = default;
            constexpr expected_delete_default_constructor& operator=(const expected_delete_default_constructor &)   = default;
            constexpr expected_delete_default_constructor& operator=(expected_delete_default_constructor &&)        = default;
        };

        template <class T, class E>
        struct expected_delete_default_constructor<T, E, false>
        {
            constexpr expected_delete_default_constructor()                                                         = delete;
            constexpr expected_delete_default_constructor(const expected_delete_default_constructor&)               = default;
            constexpr expected_delete_default_constructor(expected_delete_default_constructor&&)                    = default;
            constexpr expected_delete_default_constructor& operator=(const expected_delete_default_constructor &)   = default;
            constexpr expected_delete_default_constructor& operator=(expected_delete_default_constructor &&)        = default;
        };

        template <
          class T, 
          class E,
          bool copyable = (std::is_copy_constructible<T>::value && std::is_copy_constructible<E>::value),
          bool moveable = (std::is_move_constructible<T>::value && std::is_move_constructible<E>::value)
        >
        struct expected_delete_constructors
        {
            constexpr expected_delete_constructors()                                                = default;
            constexpr expected_delete_constructors(const expected_delete_constructors&)             = default;
            constexpr expected_delete_constructors(expected_delete_constructors&&)                  = default;
            constexpr expected_delete_constructors& operator=(const expected_delete_constructors &) = default;
            constexpr expected_delete_constructors& operator=(expected_delete_constructors &&)      = default;
        };

        template <class T, class E> 
        struct expected_delete_constructors<T, E, true, false> 
        {
            constexpr expected_delete_constructors()                                                = default;
            constexpr expected_delete_constructors(const expected_delete_constructors&)             = default;
            constexpr expected_delete_constructors(expected_delete_constructors&&)                  = delete;
            constexpr expected_delete_constructors& operator=(const expected_delete_constructors &) = default;
            constexpr expected_delete_constructors& operator=(expected_delete_constructors &&)      = default;
        };

        template <class T, class E> 
        struct expected_delete_constructors<T, E, false, true>
        {
            constexpr expected_delete_constructors()                                                = default;
            constexpr expected_delete_constructors(const expected_delete_constructors&)             = delete;
            constexpr expected_delete_constructors(expected_delete_constructors&&)                  = default;
            constexpr expected_delete_constructors& operator=(const expected_delete_constructors &) = default;
            constexpr expected_delete_constructors& operator=(expected_delete_constructors &&)      = default;
        };

        template <class T, class E> 
        struct expected_delete_constructors<T, E, false, false>
        {
            constexpr expected_delete_constructors()                                                = default;
            constexpr expected_delete_constructors(const expected_delete_constructors&)             = delete;
            constexpr expected_delete_constructors(expected_delete_constructors&&)                  = delete;
            constexpr expected_delete_constructors& operator=(const expected_delete_constructors&)  = default;
            constexpr expected_delete_constructors& operator=(expected_delete_constructors &&)      = default;
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          class E,
          bool copyable = (std::is_copy_constructible<T>::value && 
                           std::is_copy_assignable<T>::value    &&
                           std::is_copy_constructible<E>::value && 
                           std::is_copy_assignable<E>::value),
          bool moveable = (std::is_move_constructible<T>::value && 
                           std::is_move_assignable<T>::value    &&
                           std::is_move_constructible<E>::value && 
                           std::is_move_assignable<E>::value)
        >
        struct expected_delete_assign
        {
            constexpr expected_delete_assign()                                          = default;
            constexpr expected_delete_assign(const expected_delete_assign &)            = default;
            constexpr expected_delete_assign(expected_delete_assign &&)                 = default;
            constexpr expected_delete_assign& operator=(const expected_delete_assign &) = default;
            constexpr expected_delete_assign& operator=(expected_delete_assign &&)      = default;
        };

        template <class T, class E> 
        struct expected_delete_assign<T, E, true, false>
        {
            constexpr expected_delete_assign()                                          = default;
            constexpr expected_delete_assign(const expected_delete_assign &)            = default;
            constexpr expected_delete_assign(expected_delete_assign &&)                 = default;
            constexpr expected_delete_assign& operator=(const expected_delete_assign &) = default;
            constexpr expected_delete_assign& operator=(expected_delete_assign &&)      = delete;
        };

        template <class T, class E> 
        struct expected_delete_assign<T, E, false, true>
        {
            constexpr expected_delete_assign()                                          = default;
            constexpr expected_delete_assign(const expected_delete_assign &)            = default;
            constexpr expected_delete_assign(expected_delete_assign &&)                 = default;
            constexpr expected_delete_assign& operator=(const expected_delete_assign &) = delete;
            constexpr expected_delete_assign& operator=(expected_delete_assign &&)      = default;
        };

        template <class T, class E> 
        struct expected_delete_assign<T, E, false, false>
        {
            constexpr expected_delete_assign()                                          = default;
            constexpr expected_delete_assign(const expected_delete_assign &)            = default;
            constexpr expected_delete_assign(expected_delete_assign &&)                 = default;
            constexpr expected_delete_assign& operator=(const expected_delete_assign &) = delete;
            constexpr expected_delete_assign& operator=(expected_delete_assign &&)      = delete;
        };

// ---------------------------------------------------------------------------------------------------

    }

// ---------------------------------------------------------------------------------------------------

    template <class T, class E>
    class expected : private expected_details::expected_move_assign<T,E>,
                     private expected_details::expected_delete_default_constructor<T,E>,
                     private expected_details::expected_delete_constructors<T,E>,
                     private expected_details::expected_delete_assign<T,E> 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS 
                This is a standard's compliant backport of std::expected that works with C++14.
                It includes C++23 monadic interfaces
        !*/

        using base = expected_details::expected_move_assign<T,E>;

        static_assert(!std::is_reference<T>::value,             "expected<T&,E&> not allowed");
        static_assert(!std::is_reference<E>::value,             "expected<T&,E&> not allowed");
        static_assert(!std::is_same<T, in_place_t>::value,      "expected<in_place_t> not allowed");
        static_assert(!std::is_same<T, unexpect_t>::value,      "expected<unexpect_t> not allowed");
        static_assert(!std::is_same<T, unexpected<E>>::value,   "T cannot be unexpected<E>");

    public:
        using value_type        = T;
        using error_type        = E;
        using unexpected_type   = unexpected<E>;

        template< class U >
        using rebind = expected<U, error_type>;
        
        constexpr expected()                                = default;
        constexpr expected(const expected &rhs)             = default;
        constexpr expected(expected &&rhs)                  = default;
        constexpr expected& operator=(const expected &rhs)  = default;
        constexpr expected& operator=(expected &&rhs)       = default;
        ~expected()                                         = default;

        template <
          class U, 
          class G,
          class UF = std::add_lvalue_reference_t<const U>,
          class GF = const G&,
          expected_details::is_constructible<T,E,U,G,UF,GF> = true,
          std::enable_if_t<!std::is_convertible<UF, T>::value || !std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr explicit expected(const expected<U,G> &rhs)
        {
            if (rhs)
                this->contruct(std::forward<UF>(*rhs));
            else
                this->construct_error(std::forward<GF>(rhs.error()));
        }

        template <
          class U, 
          class G,
          class UF = std::add_lvalue_reference_t<const U>,
          class GF = const G&,
          expected_details::is_constructible<T,E,U,G,UF,GF> = true,
          std::enable_if_t<std::is_convertible<UF, T>::value && std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr expected(const expected<U,G> &rhs)
        {
            if (rhs)
                this->contruct(std::forward<UF>(*rhs));
            else
                this->construct_error(std::forward<GF>(rhs.error()));
        }

        template <
          class U, 
          class G,
          class UF = U,
          class GF = G,
          expected_details::is_constructible<T,E,U,G,UF,GF> = true,
          std::enable_if_t<!std::is_convertible<UF, T>::value || !std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr explicit expected(expected<U,G>&& rhs)
        {
            if (rhs)
                this->contruct(std::forward<UF>(*rhs));
            else
                this->construct_error(std::forward<GF>(rhs.error()));
        }

        template <
          class U, 
          class G,
          class UF = U,
          class GF = G,
          expected_details::is_constructible<T,E,U,G,UF,GF> = true,
          std::enable_if_t<std::is_convertible<UF, T>::value && std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr expected(expected<U,G>&& rhs)
        {
            if (rhs)
                this->contruct(std::forward<UF>(*rhs));
            else
                this->construct_error(std::forward<GF>(rhs.error()));
        }

        template< 
          class U = T,
          expected_details::is_convert_constructible<T, E, U> = true,
          std::enable_if_t<!std::is_convertible<U, T>::value, bool> = true
        >
        constexpr explicit expected( U&& v )
        : base(in_place, std::forward<U>(v))
        {
        }

        template< 
          class U = T,
          expected_details::is_convert_constructible<T, E, U> = true,
          std::enable_if_t<std::is_convertible<U, T>::value, bool> = true
        >
        constexpr expected( U&& v )
        : base(in_place, std::forward<U>(v))
        {
        }

        template< 
          class G,
          class GF = const G&,
          std::enable_if_t<std::is_constructible<E, GF>::value, bool> = true,
          std::enable_if_t<!std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr explicit expected( const unexpected<G>& e )
        : base(unexpect, std::forward<GF>(e.error()))
        {
        }

        template< 
          class G,
          class GF = const G&,
          std::enable_if_t<std::is_constructible<E, GF>::value, bool> = true,
          std::enable_if_t<std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr expected( const unexpected<G>& e )
        : base(unexpect, std::forward<GF>(e.error()))
        {
        }

        template< 
          class G,
          class GF = G,
          std::enable_if_t<std::is_constructible<E, GF>::value, bool> = true,
          std::enable_if_t<!std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr explicit expected( unexpected<G>&& e )
        : base(unexpect, std::forward<GF>(e.error()))
        {
        }

        template< 
          class G,
          class GF = G,
          std::enable_if_t<std::is_constructible<E, GF>::value, bool> = true,
          std::enable_if_t<std::is_convertible<GF, E>::value, bool> = true
        >
        constexpr expected( unexpected<G>&& e )
        : base(unexpect, std::forward<GF>(e.error()))
        {
        }

        template < 
          class... Args,
          std::enable_if_t<std::is_constructible<T, Args...>::value, bool> = true
        >
        constexpr explicit expected( 
            in_place_t, 
            Args&&... args 
        ) noexcept(std::is_nothrow_constructible<T, Args...>::value)
        : base(in_place, std::forward<Args>(args)...)
        {
        }

        template < 
          class U, 
          class... Args,
          std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value, bool> = true
        >
        constexpr explicit expected( 
            in_place_t,
            std::initializer_list<U> il,
            Args&&... args 
        ) noexcept(std::is_nothrow_constructible<T, std::initializer_list<U>&, Args...>::value)
        : base(in_place, il, std::forward<Args>(args)...)
        {
        }

        template < 
          class... Args,
          std::enable_if_t<std::is_constructible<E, Args...>::value, bool> = true
        >
        constexpr explicit expected( 
            unexpect_t, 
            Args&&... args 
        ) noexcept(std::is_nothrow_constructible<E, Args...>::value)
        : base(unexpect, std::forward<Args>(args)...)
        {
        }

        template < 
          class U, 
          class... Args,
          std::enable_if_t<std::is_constructible<E, std::initializer_list<U>&, Args...>::value, bool> = true
        >
        constexpr explicit expected( 
            unexpect_t,
            std::initializer_list<U> il,
            Args&&... args 
        ) noexcept(std::is_nothrow_constructible<E, std::initializer_list<U>&, Args...>::value)
        : base(unexpect, il, std::forward<Args>(args)...)
        {
        }

        constexpr const T*  operator->() const  noexcept { return &this->val; }
        constexpr T*        operator->()        noexcept { return &this->val; }
        constexpr T&        operator*() &       noexcept { return this->val; }
        constexpr const T&  operator*() const&  noexcept { return this->val; }
        constexpr T&&       operator*() &&      noexcept { return std::move(this->val); }
        constexpr const T&& operator*() const&& noexcept { return std::move(this->val); }
        constexpr explicit  operator bool() const noexcept { return this->state == expected_details::IS_VAL; }
        constexpr bool      has_value()     const noexcept { return this->state == expected_details::IS_VAL; }
        constexpr T&        error() &       noexcept { return this->error; }
        constexpr const T&  error() const&  noexcept { return this->error; }
        constexpr T&&       error() &&      noexcept { return std::move(this->error); }
        constexpr const T&& error() const&& noexcept { return std::move(this->error); }

        constexpr T& value() & 
        {
            if (*this)
                return **this;
            throw bad_expected_access<E>(error());
        }

        constexpr const T& value() const & 
        {
            if (*this)
                return **this;
            throw bad_expected_access<E>(error());
        }

        constexpr T&& value() && 
        {
            if (*this)
                return std::move(**this);
            throw bad_expected_access<E>(error());
        }

        constexpr const T&& value() const && 
        {
            if (*this)
                return std::move(**this);
            throw bad_expected_access<E>(error());
        }

        template <class U> 
        constexpr T value_or(U &&u) const & 
        {
            return *this ? **this : static_cast<T>(std::forward<U>(u));
        }

        template <class U> 
        constexpr T value_or(U &&u) && 
        {
            return *this ? std::move(**this) : static_cast<T>(std::forward<U>(u));
        }

        void reset() noexcept(std::is_nothrow_destructible<T>::value)
        {
            this->destruct();
        }
    };

// ---------------------------------------------------------------------------------------------------

}

#endif