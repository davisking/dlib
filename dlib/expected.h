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
        /*!
            WHAT THIS OBJECT REPRESENTS 
                This is a standard's compliant backport of std::unexpected that works with C++14.
        !*/

    public:
        constexpr unexpected()                                  = delete;
        constexpr unexpected( const unexpected& )               = default;
        constexpr unexpected( unexpected&& )                    = default;
        constexpr unexpected& operator=( const unexpected& )    = default;
        constexpr unexpected& operator=( unexpected&& )         = default;
        ~unexpected()                                           = default;

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

        static_assert(!expected_details::is_unexpected_type<E>::value,  "You cannot have unexpected<unexpected<E>>");
        static_assert(!std::is_void<E>::value,                          "E cannot be void");
        static_assert(std::is_object<E>::value,                         "E must be an object type");
        static_assert(!std::is_array<E>::value,                         "E cannot be an array type");
        static_assert(!std::is_reference<E>::value,                     "E cannot be a reference type");
        E v;
    };

    template< class E1, class E2 >
    constexpr bool operator==(unexpected<E1>& x, unexpected<E2>& y) noexcept(noexcept(x.error() == y.error()))
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
    
    template <class E>
    constexpr auto make_unexpected(E &&e) noexcept(std::is_nothrow_constructible<std::decay_t<E>, E&&>::value)
    {
        return unexpected<std::decay_t<E>>{std::forward<E>(e)};
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

        constexpr const E&  error() const &  noexcept { return err; }
        constexpr E&        error() &        noexcept { return err; }
        constexpr const E&& error() const && noexcept { return std::move(err); }
        constexpr E&&       error() &&       noexcept { return std::move(err); }

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
          class T, class E, class U, class U_ = dlib::remove_cvref_t<U>
        >
        using is_convert_constructible = std::enable_if_t<And<
            !std::is_void<T>::value,
            !std::is_same<U_, dlib::in_place_t>::value,
            !std::is_same<expected<T, E>, U_>::value,
            std::is_constructible<T, U>::value,
            !is_unexpected_type<U_>::value,
            std::is_same<bool, dlib::remove_cvref_t<T>>::value || !is_expected_type<U_>::value
        >::value,
        bool>;

        template <
          class T, class E, class GF
        >
        using is_constructible_from_error = std::enable_if_t<And<
            std::is_constructible<E, GF>::value,
            std::is_assignable<E&, GF>::value,
            Or<
              std::is_void<T>::value,
              std::is_nothrow_constructible<E, GF>::value,
              std::is_nothrow_move_constructible<T>::value,
              std::is_nothrow_move_constructible<E>::value
            >::value
          >::value,
        bool>;

// ---------------------------------------------------------------------------------------------------

        struct empty_initialization_tag{};

// ---------------------------------------------------------------------------------------------------

        static constexpr uint8_t IS_EMPTY{0};   // Technically expected<T,E> can never be valueless, but one of the base classes can.
        static constexpr uint8_t IS_VAL{1};
        static constexpr uint8_t IS_ERROR{2};

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool = std::is_trivially_destructible<T>::value,
          bool = std::is_trivially_destructible<E>::value
        >
        class expected_base
        {
        public:
            ~expected_base()
            {
                switch(state)
                {
                    case IS_VAL:    val.~T();               break;
                    case IS_ERROR:  error.~unexpected<E>(); break;
                    default:                                break;
                }
            }

            constexpr expected_base() noexcept(std::is_nothrow_default_constructible<T>::value)
            : val{}, state{IS_VAL}
            {}

        protected:

            constexpr expected_base(empty_initialization_tag) noexcept
            : e{}, state{IS_EMPTY}
            {}

            template<class ...U>
            constexpr expected_base(in_place_t, U&& ...u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            : val{std::forward<U>(u)...}, state{IS_VAL}
            {}

            template<class ...U>
            constexpr expected_base(unexpect_t, U&& ...u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            : error{std::forward<U>(u)...}, state{IS_ERROR} 
            {}    

            struct empty{};
            union {T val; unexpected<E> error; empty e;};
            uint8_t state{IS_EMPTY};
        };

        template <class T, class E>
        class expected_base<T, E, true, true>
        {
        public:
            constexpr expected_base() noexcept(std::is_nothrow_default_constructible<T>::value) 
            : val{}, state{IS_VAL}
            {}

        protected:
            constexpr expected_base(empty_initialization_tag) noexcept
            : e{}, state{IS_EMPTY}
            {}

            template<class ...U>
            constexpr expected_base(in_place_t, U&& ...u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            : val{std::forward<U>(u)...}, state{IS_VAL}
            {}

            template<class ...U>
            constexpr expected_base(unexpect_t, U&& ...u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            : error{std::forward<U>(u)...}, state{IS_ERROR} 
            {}  

            struct empty{};
            union {T val; unexpected<E> error; empty e;};
            uint8_t state{IS_EMPTY};
        };

        template <class E>
        class expected_base<void, E, false, false> // LOL void is not trivially destructible
        {
        public:
            ~expected_base()
            {
                if (state == IS_ERROR)
                    error.~unexpected<E>();
            }

            constexpr expected_base() noexcept
            : state{IS_VAL}
            {}

        protected:
            constexpr expected_base(empty_initialization_tag) noexcept
            : e{}, state{IS_EMPTY}
            {}

            constexpr expected_base(in_place_t) noexcept
            : state{IS_VAL}
            {}

            template<class ...U>
            constexpr expected_base(unexpect_t, U&& ...u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            : error{std::forward<U>(u)...}, state{IS_ERROR} 
            {}  

            struct empty{};
            union {unexpected<E> error; empty e;};
            uint8_t state{IS_EMPTY};
        };

        template <class E>
        class expected_base<void, E, false, true>
        {
        public:
            constexpr expected_base() noexcept
            : state{IS_VAL}
            {}
        
        protected:
            constexpr expected_base(empty_initialization_tag) noexcept
            : e{}, state{IS_EMPTY}
            {}

            constexpr expected_base(in_place_t) noexcept
            : state{IS_VAL}
            {}

            template<class ...U>
            constexpr expected_base(unexpect_t, U&& ...u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            : error{std::forward<U>(u)...}, state{IS_ERROR} 
            {}  

            struct empty{};
            union {unexpected<E> error; empty e;};
            uint8_t state{IS_EMPTY};
        };

// ---------------------------------------------------------------------------------------------------

        template <class T, class E>
        class expected_operations : public expected_base<T,E>
        {
        public:
            using expected_base<T,E>::expected_base;

            constexpr bool      has_value()   const noexcept { return this->state == IS_VAL; }
            constexpr T&        operator*() &       noexcept { return this->val; }
            constexpr const T&  operator*() const&  noexcept { return this->val; }
            constexpr T&&       operator*() &&      noexcept { return std::move(this->val); }
            constexpr const T&& operator*() const&& noexcept { return std::move(this->val); }

            constexpr const T*  operator->() const  noexcept { return &this->val; }
            constexpr T*        operator->()        noexcept { return &this->val; }

            constexpr T& value() & 
            {
                if (!has_value())
                    throw bad_expected_access<std::decay_t<E>>(dlib::as_const(this->error.error()));
                return **this;
            }

            constexpr const T& value() const & 
            {
                if (!has_value())
                    throw bad_expected_access<std::decay_t<E>>(dlib::as_const(this->error.error()));
                return **this;
            }

            constexpr T&& value() && 
            {
                if (!has_value())
                    throw bad_expected_access<std::decay_t<E>>(std::move(this->error.error()));
                return std::move(**this);
            }

            constexpr const T&& value() const && 
            {
                if (!has_value())
                    throw bad_expected_access<std::decay_t<E>>(std::move(this->error.error()));
                return std::move(**this);
            }

            template <class U> 
            constexpr T value_or(U &&u) const & 
            {
                return has_value() ? **this : static_cast<T>(std::forward<U>(u));
            }

            template <class U> 
            constexpr T value_or(U &&u) && 
            {
                return has_value() ? std::move(**this) : static_cast<T>(std::forward<U>(u));
            }

            template< 
              class... Args,
              std::enable_if_t<std::is_nothrow_constructible<T, Args...>::value, bool> = true
            >
            constexpr T& emplace( Args&&... args ) noexcept
            {
                this->destruct();
                this->construct_value(std::forward<Args>(args)...);
            }

            template < 
              class U, 
              class... Args,
              std::enable_if_t<std::is_nothrow_constructible<T, std::initializer_list<U>&, Args...>::value, bool> = true
            >
            constexpr T& emplace( std::initializer_list<U>& il, Args&&... args ) noexcept
            {
                this->destruct();
                this->construct_value(il, std::forward<Args>(args)...);
            }

        protected:
            constexpr void destruct() noexcept(std::is_nothrow_destructible<T>::value &&
                                               std::is_nothrow_destructible<E>::value)
            {
                switch(this->state)
                {
                    case IS_VAL:    this->val.~T();               break;
                    case IS_ERROR:  this->error.~unexpected<E>(); break;
                    default:                                      break;
                }
                this->state = IS_EMPTY;
            }  

            template <class... U> 
            constexpr void construct_value(U&&... u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            {
                new (std::addressof(this->val)) T(std::forward<U>(u)...);
                this->state = IS_VAL;
            }

            template <class... U> 
            constexpr void construct_error(U&&... u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            {
                new (std::addressof(this->error)) unexpected<E>(std::forward<U>(u)...);
                this->state = IS_ERROR;
            }    

            template <class Expected>
            constexpr void construct(Expected&& rhs)
            {
                switch(rhs.state)
                {
                    case IS_VAL:    construct_value(std::forward<Expected>(rhs).val);           break;
                    case IS_ERROR:  construct_error(std::forward<Expected>(rhs).error.error()); break;
                    default:        this->state = IS_ERROR;                                     break;
                }
            }          

            template<class Expected>
            constexpr void assign(Expected&& rhs)
            {   
                /* This is screaming for pattern matching. */

                const uint8_t combined = this->state | (rhs.state << 4);
                
                switch(combined)
                {
                    case IS_EMPTY | (IS_VAL << 4): 
                        construct_value(std::forward<Expected>(rhs).val);
                        break;

                    case IS_ERROR | (IS_VAL << 4): 
                        destruct();
                        construct_value(std::forward<Expected>(rhs).val);
                        break;

                    case IS_VAL | (IS_VAL << 4): 
                        this->val = std::forward<Expected>(rhs).val; 
                        break;
                    
                    case IS_EMPTY | (IS_EMPTY << 4):
                        break;

                    case IS_ERROR | (IS_EMPTY << 4): 
                        destruct();
                        break;

                    case IS_VAL | (IS_EMPTY << 4):
                        destruct();
                        break;
                    
                    case IS_EMPTY | (IS_ERROR << 4): 
                        construct_error(std::forward<Expected>(rhs).error);
                        break;

                    case IS_ERROR | (IS_ERROR << 4): 
                        this->error = std::forward<Expected>(rhs).error; 
                        break;

                    case IS_VAL | (IS_ERROR << 4):
                        destruct();
                        construct_error(std::forward<Expected>(rhs).error);
                        break;

                    default:
                        break;
                }
            }  
        };

        template <class E>
        class expected_operations<void,E> : public expected_base<void,E>
        {
        public:
            using expected_base<void,E>::expected_base;

            constexpr bool has_value() const noexcept { return this->state == IS_VAL; }
            constexpr void operator*() const noexcept {}

            constexpr void value() const &
            {
                if (!has_value())
                    throw bad_expected_access<std::decay_t<E>>(dlib::as_const(this->error));
            }

            constexpr void value() &&
            {
                if (!has_value())
                    throw bad_expected_access<std::decay_t<E>>(std::move(this->error));
            }

            constexpr void emplace() noexcept
            {
                this->destruct();
            }

        protected:
            constexpr void destruct() noexcept(std::is_nothrow_destructible<E>::value)
            {
                if (std::exchange(this->state, IS_EMPTY) == IS_ERROR)
                    this->error.~unexpected<E>();
            }  

            template <class... U> 
            constexpr void construct_error(U&&... u) noexcept(std::is_nothrow_constructible<E,U...>::value)
            {
                new (std::addressof(this->error)) unexpected<E>(std::forward<U>(u)...);
                this->state = IS_ERROR;
            }    

            template <class Expected>
            constexpr void construct(Expected&& rhs)
            {
                switch(rhs.state)
                {
                    case IS_VAL:    this->state = IS_VAL;                                       break;
                    case IS_ERROR:  construct_error(std::forward<Expected>(rhs).error.error()); break;
                    default:        this->state = IS_ERROR;                                     break;
                }
            }          

            template<class Expected>
            constexpr void assign(Expected&& rhs)
            {                    
                /* This is screaming for pattern matching. */

                const uint8_t combined = this->state | (rhs.state << 4);
                
                switch(combined)
                {
                    case IS_EMPTY | (IS_VAL << 4): 
                        this->state = IS_VAL;
                        break;

                    case IS_ERROR | (IS_VAL << 4): 
                        destruct();
                        this->state = IS_VAL;
                        break;
                    
                    case IS_ERROR | (IS_EMPTY << 4): 
                        destruct();
                        break;

                    case IS_VAL | (IS_EMPTY << 4):
                        destruct();
                        break;
                    
                    case IS_VAL | (IS_ERROR << 4):
                        destruct();
                        construct_error(std::forward<Expected>(rhs).error);
                        break;
                                        
                    case IS_EMPTY | (IS_ERROR << 4): 
                        construct_error(std::forward<Expected>(rhs).error);
                        break;

                    case IS_ERROR | (IS_ERROR << 4): 
                        this->error = std::forward<Expected>(rhs).error; 
                        break;

                    default:
                        break;
                }
            }  
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          class E,
          bool = (disjunction<std::is_void<T>, std::is_trivially_copy_constructible<T>>::value && std::is_trivially_copy_constructible<E>::value)
        >
        struct expected_copy : public expected_operations<T,E> 
        {
            using expected_operations<T,E>::expected_operations;
        };

        template <class T, class E>
        struct expected_copy<T, E, false> : public expected_operations<T,E> 
        {
            using expected_operations<T,E>::expected_operations;

            constexpr expected_copy()                                       = default;
            constexpr expected_copy(expected_copy&& rhs)                    = default;
            constexpr expected_copy &operator=(const expected_copy& rhs)    = default;
            constexpr expected_copy &operator=(expected_copy&& rhs)         = default;

            constexpr expected_copy(const expected_copy& rhs) noexcept(std::is_nothrow_copy_constructible<T>::value &&
                                                                       std::is_nothrow_copy_constructible<E>::value)
            : expected_operations<T, E>(empty_initialization_tag{})
            {
                this->construct(rhs);
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          class E,
          bool = (disjunction<std::is_void<T>, std::is_trivially_move_constructible<T>>::value && std::is_trivially_move_constructible<E>::value)
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
                this->construct(std::move(rhs));
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool = disjunction<std::is_void<T>,
                    conjunction<std::is_trivially_copy_assignable<T>,
                                std::is_trivially_copy_constructible<T>,
                                std::is_trivially_destructible<T>>>::value  && 
                 std::is_trivially_copy_assignable<E>::value                &&
                 std::is_trivially_copy_constructible<E>::value             &&
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
                     std::is_nothrow_destructible<T>::value         &&
                     std::is_nothrow_copy_constructible<E>::value   && 
                     std::is_nothrow_copy_assignable<E>::value      && 
                     std::is_nothrow_destructible<E>::value)
            {
                this->assign(rhs);
                return *this;
            }        
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool = disjunction<std::is_void<T>,
                    conjunction<std::is_trivially_move_assignable<T>,
                                std::is_trivially_move_constructible<T>,
                                std::is_trivially_destructible<T>>>::value  && 
                 std::is_trivially_move_assignable<E>::value                &&
                 std::is_trivially_move_constructible<E>::value             &&
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
                     std::is_nothrow_destructible<T>::value         &&
                     std::is_nothrow_move_constructible<E>::value   && 
                     std::is_nothrow_move_assignable<E>::value      && 
                     std::is_nothrow_destructible<E>::value)
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

        template <class E>
        struct expected_delete_default_constructor<void, E, false>
        {
            constexpr expected_delete_default_constructor()                                                         = default;
            constexpr expected_delete_default_constructor(const expected_delete_default_constructor&)               = default;
            constexpr expected_delete_default_constructor(expected_delete_default_constructor&&)                    = default;
            constexpr expected_delete_default_constructor& operator=(const expected_delete_default_constructor &)   = default;
            constexpr expected_delete_default_constructor& operator=(expected_delete_default_constructor &&)        = default;
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          class E,
          bool copyable = (disjunction<std::is_void<T>, std::is_copy_constructible<T>>::value && std::is_copy_constructible<E>::value),
          bool moveable = (disjunction<std::is_void<T>, std::is_move_constructible<T>>::value && std::is_move_constructible<E>::value)
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
          bool copyable = (disjunction<std::is_void<T>, std::is_copy_constructible<T>>::value   &&
                           disjunction<std::is_void<T>, std::is_copy_assignable<T>>::value      &&
                           std::is_copy_constructible<E>::value                                 && 
                           std::is_copy_assignable<E>::value),
          bool moveable = (disjunction<std::is_void<T>, std::is_move_constructible<T>>::value   && 
                           disjunction<std::is_void<T>, std::is_move_assignable<T>>::value      &&
                           std::is_move_constructible<E>::value                                 && 
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
    class expected : public expected_details::expected_move_assign<T,E>,
                     private expected_details::expected_delete_default_constructor<T,E>,
                     private expected_details::expected_delete_constructors<T,E>,
                     private expected_details::expected_delete_assign<T,E>
    {
        /*!
            WHAT THIS OBJECT REPRESENTS 
                This is a standard's compliant backport of std::expected that works with C++14.
                It includes C++23 monadic interfaces.
                This is an optimized vocabulary type, designed for error handling and value semantics.
                It correctly propagates constexpr, noexcept-ness and triviality.
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
            this->construct(rhs);
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
            this->construct(rhs);
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
            this->construct(std::move(rhs));
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
            this->construct(std::move(rhs));
        }

        template< 
          class U = T,
          expected_details::is_convert_constructible<T, E, U> = true,
          std::enable_if_t<!std::is_convertible<U, T>::value, bool> = true
        >
        constexpr explicit expected( U&& v )
        : base(dlib::in_place, std::forward<U>(v))
        {
        }

        template< 
          class U = T,
          expected_details::is_convert_constructible<T, E, U> = true,
          std::enable_if_t<std::is_convertible<U, T>::value, bool> = true
        >
        constexpr expected( U&& v )
        : base(dlib::in_place, std::forward<U>(v))
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
            dlib::in_place_t, 
            Args&&... args 
        ) noexcept(std::is_nothrow_constructible<T, Args...>::value)
        : base(dlib::in_place, std::forward<Args>(args)...)
        {
        }

        template < 
          class U, 
          class... Args,
          std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args...>::value, bool> = true
        >
        constexpr explicit expected( 
            dlib::in_place_t,
            std::initializer_list<U> il,
            Args&&... args 
        ) noexcept(std::is_nothrow_constructible<T, std::initializer_list<U>&, Args...>::value)
        : base(dlib::in_place, il, std::forward<Args>(args)...)
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

        template <
          class U  = T,
          class G  = T,
          class U_ = dlib::remove_cvref_t<U>,
          std::enable_if_t<And<
            !std::is_void<G>::value,
            !std::is_same<expected<G,E>, U_>::value,
            !expected_details::is_unexpected_type<U_>::value,
            std::is_constructible<G,U>::value,
            std::is_assignable<G&,U>::value,
            Or<std::is_nothrow_constructible<G, U>::value,
               std::is_nothrow_move_constructible<G>::value,
               std::is_nothrow_move_constructible<E>::value>::value
          >::value, bool> = true
        >
        constexpr expected& operator=( U&& v )
        {
            using namespace expected_details;

            if (*this)
            {
                **this = std::forward<U>(v); 
            }
            else
            {
                this->destruct();
                this->construct_value(std::forward<U>(v));
            }
            
            return *this;
        }

        template< 
          class G,
          class GF = const G&,
          expected_details::is_constructible_from_error<T,E,GF> = true
        >
        constexpr expected& operator=( const unexpected<G>& e )
        {
            if (!*this)
            {
                error() = std::forward<GF>(e.error());
            }
            else
            {
                this->destruct();
                this->construct_error(std::forward<GF>(e.error()));
            }
            return *this;
        }

        template< 
          class G,
          class GF = G,
          expected_details::is_constructible_from_error<T,E,GF> = true
        >
        constexpr expected& operator=( unexpected<G>&& e )
        {
            if (!*this)
            {
                error() = std::forward<GF>(e.error());
            }
            else
            {
                this->destruct();
                this->construct_error(std::forward<GF>(e.error()));
            }
            return *this;
        }
        
        constexpr explicit  operator bool() const noexcept { return base::has_value(); }
        constexpr E&        error() &       noexcept { return base::error.error(); }
        constexpr const E&  error() const&  noexcept { return base::error.error(); }
        constexpr E&&       error() &&      noexcept { return std::move(base::error.error()); }
        constexpr const E&& error() const&& noexcept { return std::move(base::error.error()); }

        void reset() noexcept(std::is_nothrow_destructible<T>::value)
        {
            this->destruct();
        }
    };

// ---------------------------------------------------------------------------------------------------

}

#endif