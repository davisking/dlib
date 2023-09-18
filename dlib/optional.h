// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIONAL_H
#define DLIB_OPTIONAL_H

#include <exception>
#include <initializer_list>
#include "functional.h"

namespace dlib
{

// ---------------------------------------------------------------------------------------------------

    class bad_optional_access : public std::exception 
    {
    public:
        bad_optional_access() = default;
        const char *what() const noexcept { return "Optional has no value"; }
    };

// ---------------------------------------------------------------------------------------------------

    struct nullopt_t 
    {
        nullopt_t() = delete;
        constexpr explicit nullopt_t(int) noexcept {}
    };

    static constexpr nullopt_t nullopt{int{}};

// ---------------------------------------------------------------------------------------------------

    template<class T>
    class optional;
    /*!
        WHAT THIS OBJECT REPRESENTS 
            This is a standard's compliant backport of std::optional that works with C++14.
            It includes C++23 monadic interfaces

            Therefore, refer to https://en.cppreference.com/w/cpp/utility/optional for docs on the
            interface of optional.
    !*/


// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
//                                IMPLEMENTATION DETAILS BELOW
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------

    namespace details
    {
        template<class T>
        struct is_optional : std::false_type{};

        template<class T>
        struct is_optional<dlib::optional<T>> : std::true_type{};

        template<class T, class U>
        using is_constructible_from = And<
            !std::is_constructible<T,       dlib::optional<U>&>::value,
            !std::is_constructible<T, const dlib::optional<U>&>::value,
            !std::is_constructible<T,       dlib::optional<U>&&>::value,
            !std::is_constructible<T, const dlib::optional<U>&&>::value,
            !std::is_convertible<      dlib::optional<U>&, T>::value,
            !std::is_convertible<const dlib::optional<U>&, T>::value,
            !std::is_convertible<      dlib::optional<U>&&, T>::value,
            !std::is_convertible<const dlib::optional<U>&&, T>::value
        >;

        template<class T, class U>
        using is_assignable_from = And<
            is_constructible_from<T,U>::value,
            std::is_assignable<T&,       dlib::optional<U>&>::value,
            std::is_assignable<T&, const dlib::optional<U>&>::value,
            std::is_assignable<T&,       dlib::optional<U>&&>::value,
            std::is_assignable<T&, const dlib::optional<U>&&>::value
        >;

        template<class T, class U>
        using is_copy_convertible = std::enable_if_t<
            is_constructible_from<T,U>::value &&
            std::is_constructible<T, const U&>::value,
            bool
        >;

        template<class T, class U>
        using is_move_convertible = std::enable_if_t<
            is_constructible_from<T,U>::value &&
            std::is_constructible<T, U&&>::value,
            bool
        >;

        template<class T, class U>
        using is_copy_assignable = std::enable_if_t<
            is_assignable_from<T,U>::value              &&
            std::is_constructible<T, const U&>::value   &&
            std::is_assignable<T&, const U&>::value,
            bool
        >;

        template<class T, class U>
        using is_move_assignable = std::enable_if_t<
            is_assignable_from<T,U>::value      &&
            std::is_constructible<T, U>::value  &&
            std::is_assignable<T&, U>::value,
            bool
        >;

        template<class T, class U, class U_ = std::decay_t<U>>
        using is_construct_convertible_from = std::enable_if_t<
            std::is_constructible<T, U&&>::value &&
            !std::is_same<U_, in_place_t>::value &&
            !std::is_same<U_, dlib::optional<T>>::value,
            bool
        >;

        template <class T, class U, class U_ = std::decay_t<U>>
        using is_assign_convertible_from = std::enable_if_t<
            std::is_constructible<T, U>::value  && 
            std::is_assignable<T&, U>::value    &&
            !std::is_same<U_, dlib::optional<T>>::value &&
            (!std::is_scalar<T>::value || !std::is_same<T, U_>::value),
            bool
        >;

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          bool = std::is_trivially_destructible<T>::value
        >
        struct optional_storage
        {
            constexpr optional_storage() noexcept
            : e{}, active{false}
            {}

            template<class ...U>
            constexpr optional_storage(in_place_t, U&& ...u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            : val{std::forward<U>(u)...}, active{true} 
            {}

            ~optional_storage() noexcept(std::is_nothrow_destructible<T>::value)
            {
                if (active)
                    val.~T();
            }           

            struct empty{};
            union {T val; empty e;};
            bool active{false};
        };

        template <class T>
        struct optional_storage<T, true>
        {
            constexpr optional_storage() noexcept
            : e{}, active{false}
            {}

            template<class ...U>
            constexpr optional_storage(in_place_t, U&& ...u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            : val{std::forward<U>(u)...}, active{true} 
            {}

            struct empty{};
            union {T val; empty e;};
            bool active{false};
        };

// ---------------------------------------------------------------------------------------------------

        template <class T>
        struct optional_ops : optional_storage<T> 
        {
            using optional_storage<T>::optional_storage;

            template <class... U> 
            constexpr void construct(U&&... u) noexcept(std::is_nothrow_constructible<T,U...>::value)
            {
                new (std::addressof(this->val)) T(std::forward<U>(u)...);
                this->active = true;
            }

            template<class Optional>
            constexpr void assign(Optional&& rhs) noexcept(std::is_nothrow_constructible<T,Optional>::value &&
                                                           std::is_nothrow_assignable<T&,Optional>::value)
            {
                if (this->active && rhs.active)
                    this->val = std::forward<Optional>(rhs).val;
                else if (!this->active && rhs.active)
                    construct(std::forward<Optional>(rhs).val);
                else if (this->active && !rhs.active)
                    destruct();
            }

            constexpr void destruct() noexcept(std::is_nothrow_destructible<T>::value)
            {
                if (this->active)
                {
                    this->val.~T();
                    this->active = false;
                }
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <class T, bool = std::is_trivially_copy_constructible<T>::value>
        struct optional_copy : optional_ops<T> 
        {
            using optional_ops<T>::optional_ops;
        };

        template <class T>
        struct optional_copy<T, false> : optional_ops<T> 
        {
            using optional_ops<T>::optional_ops;

            constexpr optional_copy()                                       = default;
            constexpr optional_copy(optional_copy&& rhs)                    = default;
            constexpr optional_copy &operator=(const optional_copy& rhs)    = default;
            constexpr optional_copy &operator=(optional_copy&& rhs)         = default;

            constexpr optional_copy(const optional_copy& rhs) noexcept(std::is_nothrow_copy_constructible<T>::value)
            : optional_ops<T>() 
            {
                if (rhs.active)
                    this->construct(rhs.val);
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <class T, bool = std::is_trivially_move_constructible<T>::value>
        struct optional_move : optional_copy<T> 
        {
            using optional_copy<T>::optional_copy;
        };

        template <class T> 
        struct optional_move<T, false> : optional_copy<T> 
        {
            using optional_copy<T>::optional_copy;

            constexpr optional_move()                                       = default;
            constexpr optional_move(const optional_move& rhs)               = default;
            constexpr optional_move& operator=(const optional_move& rhs)    = default;
            constexpr optional_move& operator=(optional_move&& rhs)         = default;

            constexpr optional_move(optional_move&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value)
            {
                if (rhs.active)
                    this->construct(std::move(rhs.val));
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          bool = std::is_trivially_copy_assignable<T>::value       &&
                 std::is_trivially_copy_constructible<T>::value    &&
                 std::is_trivially_destructible<T>::value
        >
        struct optional_copy_assign : optional_move<T> 
        {
            using optional_move<T>::optional_move;
        };

        template <class T>
        struct optional_copy_assign<T, false> : optional_move<T> 
        {
            using optional_move<T>::optional_move;

            constexpr optional_copy_assign()                                        = default;
            constexpr optional_copy_assign(const optional_copy_assign& rhs)         = default;
            constexpr optional_copy_assign(optional_copy_assign&& rhs)              = default;
            constexpr optional_copy_assign& operator=(optional_copy_assign &&rhs)   = default;

            constexpr optional_copy_assign& operator=(const optional_copy_assign &rhs) 
            noexcept(std::is_nothrow_copy_constructible<T>::value && 
                    std::is_nothrow_copy_assignable<T>::value)
            {
                this->assign(rhs);
                return *this;
            }        
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          bool = std::is_trivially_destructible<T>::value       &&
                 std::is_trivially_move_constructible<T>::value &&
                 std::is_trivially_move_assignable<T>::value
        >
        struct optional_move_assign : optional_copy_assign<T> 
        {
            using optional_copy_assign<T>::optional_copy_assign;
        };

        template <class T>
        struct optional_move_assign<T, false> : optional_copy_assign<T> 
        {
            using optional_copy_assign<T>::optional_copy_assign;

            constexpr optional_move_assign()                                              = default;
            constexpr optional_move_assign(const optional_move_assign &rhs)               = default;
            constexpr optional_move_assign(optional_move_assign &&rhs)                    = default;
            constexpr optional_move_assign& operator=(const optional_move_assign &rhs)    = default;

            constexpr optional_move_assign& operator=(optional_move_assign &&rhs) 
            noexcept(std::is_nothrow_move_constructible<T>::value && 
                    std::is_nothrow_move_assignable<T>::value)
            {
                this->assign(std::move(rhs));
                return *this;
            }
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T, 
          bool copyable = std::is_copy_constructible<T>::value,
          bool moveable = std::is_move_constructible<T>::value
        >
        struct optional_delete_constructors
        {
            constexpr optional_delete_constructors()                                                = default;
            constexpr optional_delete_constructors(const optional_delete_constructors&)             = default;
            constexpr optional_delete_constructors(optional_delete_constructors&&)                  = default;
            constexpr optional_delete_constructors& operator=(const optional_delete_constructors &) = default;
            constexpr optional_delete_constructors& operator=(optional_delete_constructors &&)      = default;
        };

        template <class T> 
        struct optional_delete_constructors<T, true, false> 
        {
            constexpr optional_delete_constructors()                                                = default;
            constexpr optional_delete_constructors(const optional_delete_constructors&)             = default;
            constexpr optional_delete_constructors(optional_delete_constructors&&)                  = delete;
            constexpr optional_delete_constructors& operator=(const optional_delete_constructors &) = default;
            constexpr optional_delete_constructors& operator=(optional_delete_constructors &&)      = default;
        };

        template <class T> 
        struct optional_delete_constructors<T, false, true>
        {
            constexpr optional_delete_constructors()                                                = default;
            constexpr optional_delete_constructors(const optional_delete_constructors&)             = delete;
            constexpr optional_delete_constructors(optional_delete_constructors&&)                  = default;
            constexpr optional_delete_constructors& operator=(const optional_delete_constructors &) = default;
            constexpr optional_delete_constructors& operator=(optional_delete_constructors &&)      = default;
        };

        template <class T> 
        struct optional_delete_constructors<T, false, false>
        {
            constexpr optional_delete_constructors()                                                = default;
            constexpr optional_delete_constructors(const optional_delete_constructors&)             = delete;
            constexpr optional_delete_constructors(optional_delete_constructors&&)                  = delete;
            constexpr optional_delete_constructors& operator=(const optional_delete_constructors&)  = default;
            constexpr optional_delete_constructors& operator=(optional_delete_constructors &&)      = default;
        };

// ---------------------------------------------------------------------------------------------------

        template <
          class T,
          bool copyable = (std::is_copy_constructible<T>::value && std::is_copy_assignable<T>::value),
          bool moveable = (std::is_move_constructible<T>::value && std::is_move_assignable<T>::value)
        >
        struct optional_delete_assign
        {
            constexpr optional_delete_assign()                                          = default;
            constexpr optional_delete_assign(const optional_delete_assign &)            = default;
            constexpr optional_delete_assign(optional_delete_assign &&)                 = default;
            constexpr optional_delete_assign& operator=(const optional_delete_assign &) = default;
            constexpr optional_delete_assign& operator=(optional_delete_assign &&)      = default;
        };

        template <class T> 
        struct optional_delete_assign<T, true, false>
        {
            constexpr optional_delete_assign()                                          = default;
            constexpr optional_delete_assign(const optional_delete_assign &)            = default;
            constexpr optional_delete_assign(optional_delete_assign &&)                 = default;
            constexpr optional_delete_assign& operator=(const optional_delete_assign &) = default;
            constexpr optional_delete_assign& operator=(optional_delete_assign &&)      = delete;
        };

        template <class T> 
        struct optional_delete_assign<T, false, true>
        {
            constexpr optional_delete_assign()                                          = default;
            constexpr optional_delete_assign(const optional_delete_assign &)            = default;
            constexpr optional_delete_assign(optional_delete_assign &&)                 = default;
            constexpr optional_delete_assign& operator=(const optional_delete_assign &) = delete;
            constexpr optional_delete_assign& operator=(optional_delete_assign &&)      = default;
        };

        template <class T> 
        struct optional_delete_assign<T, false, false>
        {
            constexpr optional_delete_assign()                                          = default;
            constexpr optional_delete_assign(const optional_delete_assign &)            = default;
            constexpr optional_delete_assign(optional_delete_assign &&)                 = default;
            constexpr optional_delete_assign& operator=(const optional_delete_assign &) = delete;
            constexpr optional_delete_assign& operator=(optional_delete_assign &&)      = delete;
        };

// ---------------------------------------------------------------------------------------------------

    }

// ---------------------------------------------------------------------------------------------------

    template <class T>
    class optional : private details::optional_move_assign<T>,
                     private details::optional_delete_constructors<T>,
                     private details::optional_delete_assign<T> 
    {
        using base = details::optional_move_assign<T>;

        static_assert(!std::is_reference<T>::value,         "optional<T&> not allowed");
        static_assert(!std::is_same<T, in_place_t>::value,  "optional<in_place_t> not allowed");
        static_assert(!std::is_same<T, nullopt_t>::value,   "optional<nullopt_t> not allowed");

    public:
        using value_type = T;
        
        constexpr optional()                                = default;
        constexpr optional(const optional &rhs)             = default;
        constexpr optional(optional &&rhs)                  = default;
        constexpr optional& operator=(const optional &rhs)  = default;
        constexpr optional& operator=(optional &&rhs)       = default;
        ~optional()                                         = default;

        constexpr optional(nullopt_t) noexcept {}

        constexpr optional& operator=(nullopt_t) noexcept 
        {
            if (*this)
                reset();
            return *this;
        }

        template <
          class U, 
          details::is_copy_convertible<T,U> = true,
          std::enable_if_t<!std::is_convertible<const U&, T>::value, bool> = true
        >
        constexpr explicit optional(const optional<U> &rhs) noexcept(std::is_nothrow_constructible<T,const U&>::value)
        {
            if (rhs)
                this->construct(*rhs);
        }

        template <
          class U, 
          details::is_copy_convertible<T,U> = true,
          std::enable_if_t<std::is_convertible<const U&, T>::value, bool> = true
        >
        constexpr optional(const optional<U> &rhs) noexcept(std::is_nothrow_constructible<T,const U&>::value)
        {
            if (rhs)
                this->construct(*rhs);
        }

        template <
          class U, 
          details::is_move_convertible<T,U> = true,
          std::enable_if_t<!std::is_convertible<U&&, T>::value, bool> = true
        >
        constexpr explicit optional(optional<U>&& rhs) noexcept(std::is_nothrow_constructible<T,U&&>::value)
        {
            if (rhs)
                this->construct(std::move(*rhs));
        }

        template <
          class U, 
          details::is_move_convertible<T,U> = true,
          std::enable_if_t<std::is_convertible<U&&, T>::value, bool> = true
        >
        constexpr optional(optional<U>&& rhs) noexcept(std::is_nothrow_constructible<T,U&&>::value)
        {
            if (rhs)
                this->construct(std::move(*rhs));
        }

        template <
          class... Args,
          std::enable_if_t<std::is_constructible<T, Args...>::value, bool> = true
        >
        constexpr explicit optional (
            in_place_t,
            Args&&... args
        ) noexcept(std::is_nothrow_constructible<T,Args&&...>::value)
        : base(in_place, std::forward<Args>(args)...)
        {
        }

        template <
          class U, 
          class... Args,
          std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value, bool> = true
        >
        constexpr explicit optional (
            in_place_t,
            std::initializer_list<U> il,
            Args &&... args
        ) noexcept(std::is_nothrow_constructible<T,std::initializer_list<U>&,Args&&...>::value)
        {
            this->construct(il, std::forward<Args>(args)...);
        }

        template<
          class U,
          details::is_construct_convertible_from<T,U> = true,
          std::enable_if_t<!std::is_convertible<U&&, T>::value, bool> = true
        >
        constexpr explicit optional(U &&u) noexcept(std::is_nothrow_constructible<T, U&&>::value)
        : base(in_place, std::forward<U>(u))
        {
        }

        template<
          class U,
          details::is_construct_convertible_from<T,U> = true,
          std::enable_if_t<std::is_convertible<U&&, T>::value, bool> = true
        >
        constexpr optional(U &&u) noexcept(std::is_nothrow_constructible<T, U&&>::value)
        : base(in_place, std::forward<U>(u))
        {
        }      

        template <
          class U,
          details::is_copy_assignable<T, U> = true
        >
        constexpr optional &operator=(const optional<U>& rhs) noexcept(std::is_nothrow_constructible<T, const U&>::value &&
                                                                       std::is_nothrow_assignable<T, const U&>::value)
        {
            this->assign(rhs);
            return *this;
        }

        template <
          class U,
          details::is_move_assignable<T, U> = true
        >
        constexpr optional &operator=(optional<U>&& rhs) noexcept(std::is_nothrow_constructible<T, U>::value &&
                                                                  std::is_nothrow_assignable<T, U>::value)
        {
            this->assign(std::move(rhs));
            return *this;
        }
        
        template <
          class U, 
          details::is_assign_convertible_from<T,U> = true
        >
        constexpr optional& operator=(U &&u) noexcept(std::is_nothrow_constructible<T, U>::value &&
                                                      std::is_nothrow_assignable<T, U>::value)
        {
            if (*this)
                **this = std::forward<U>(u);
            else
                this->construct(std::forward<U>(u));
            return *this;
        }

        template <class... Args> 
        constexpr T& emplace(Args &&... args) noexcept(std::is_nothrow_constructible<T, Args...>::value)
        {
            reset();
            this->construct(std::forward<Args>(args)...);
            return **this;
        }

        template <class U, class... Args>
        constexpr T& emplace(std::initializer_list<U> il, Args &&... args) 
        {
            reset();
            this->construct(il, std::forward<Args>(args)...);
            return **this;   
        }

        void swap(optional& rhs) noexcept(std::is_nothrow_move_constructible<T>::value &&
                                          dlib::is_nothrow_swappable<T>::value) 
        {
            using std::swap;

            if (*this && rhs)
            {
                swap(**this, *rhs);
            }
                
            else if (*this && !rhs)
            {
                rhs = std::move(**this);
                reset();
            }
            
            else if (!*this && rhs)
            {
                *this = std::move(*rhs);
                rhs.reset();
            }
        }

        constexpr const T*  operator->() const  noexcept { return &this->val; }
        constexpr T*        operator->()        noexcept { return &this->val; }
        constexpr T&        operator*() &       noexcept { return this->val; }
        constexpr const T&  operator*() const&  noexcept { return this->val; }
        constexpr T&&       operator*() &&      noexcept { return std::move(this->val); }
        constexpr const T&& operator*() const&& noexcept { return std::move(this->val); }
        constexpr explicit  operator bool() const noexcept { return this->active; }
        constexpr bool      has_value()     const noexcept { return this->active; }

        constexpr T& value() & 
        {
            if (*this)
                return **this;
            throw bad_optional_access();
        }

        constexpr const T& value() const & 
        {
            if (*this)
                return **this;
            throw bad_optional_access();
        }

        constexpr T&& value() && 
        {
            if (*this)
                return std::move(**this);
            throw bad_optional_access();
        }

        constexpr const T&& value() const && 
        {
            if (*this)
                return std::move(**this);
            throw bad_optional_access();
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

        template <
          class F,
          class Return = dlib::remove_cvref_t<dlib::invoke_result_t<F,T&>>,
          std::enable_if_t<details::is_optional<Return>::value, bool> = true
        >
        constexpr auto and_then(F&& f) &
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), **this);
            else
                return Return{};
        }

        template <
          class F,
          class Return = dlib::remove_cvref_t<dlib::invoke_result_t<F,const T&>>,
          std::enable_if_t<details::is_optional<Return>::value, bool> = true
        >
        constexpr auto and_then(F&& f) const&
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), **this);
            else
                return Return{};
        }

        template <
          class F,
          class Return = dlib::remove_cvref_t<dlib::invoke_result_t<F,T>>,
          std::enable_if_t<details::is_optional<Return>::value, bool> = true
        >
        constexpr auto and_then(F&& f) &&
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), std::move(**this));
            else
                return Return{};
        }

        template <
          class F,
          class Return = dlib::remove_cvref_t<dlib::invoke_result_t<F,const T>>,
          std::enable_if_t<details::is_optional<Return>::value, bool> = true
        >
        constexpr auto and_then(F&& f) const&&
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), std::move(**this));
            else
                return Return{};
        }

        template <
          class F,
          class U = dlib::remove_cvref_t<dlib::invoke_result_t<F, T&>>,
          std::enable_if_t<!std::is_same<U, dlib::in_place_t>::value, bool> = true,
          std::enable_if_t<!std::is_same<U, dlib::nullopt_t>::value, bool> = true
        >
        constexpr dlib::optional<U> transform(F&& f) &
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), **this);
            else
                return dlib::optional<U>{};
        }

        template <
          class F,
          class U = dlib::remove_cvref_t<dlib::invoke_result_t<F, const T&>>,
          std::enable_if_t<!std::is_same<U, dlib::in_place_t>::value, bool> = true,
          std::enable_if_t<!std::is_same<U, dlib::nullopt_t>::value, bool> = true
        >
        constexpr dlib::optional<U> transform(F&& f) const&
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), **this);
            else
                return dlib::optional<U>{};
        }

        template <
          class F,
          class U = dlib::remove_cvref_t<dlib::invoke_result_t<F,T>>,
          std::enable_if_t<!std::is_same<U, dlib::in_place_t>::value, bool> = true,
          std::enable_if_t<!std::is_same<U, dlib::nullopt_t>::value, bool> = true
        >
        constexpr dlib::optional<U> transform(F&& f) &&
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), std::move(**this));
            else
                return dlib::optional<U>{};
        }

        template <
          class F,
          class U = dlib::remove_cvref_t<dlib::invoke_result_t<F,const T>>,
          std::enable_if_t<!std::is_same<U, dlib::in_place_t>::value, bool> = true,
          std::enable_if_t<!std::is_same<U, dlib::nullopt_t>::value, bool> = true
        >
        constexpr dlib::optional<U> transform(F&& f) const&&
        {
            if (*this)
                return dlib::invoke(std::forward<F>(f), std::move(**this));
            else
                return dlib::optional<U>{};
        }

        template < 
          class F,
          class U = dlib::remove_cvref_t<dlib::invoke_result_t<F>>,
          std::enable_if_t<std::is_same<U, dlib::optional<T>>::value, bool> = true
        >
        constexpr optional or_else( F&& f ) const&
        {
            return *this ? *this : std::forward<F>(f)();
        }

        template < 
          class F,
          class U = dlib::remove_cvref_t<dlib::invoke_result_t<F>>,
          std::enable_if_t<std::is_same<U, dlib::optional<T>>::value, bool> = true
        >
        constexpr optional or_else( F&& f ) &&
        {
            return *this ? std::move(*this) : std::forward<F>(f)();
        }
    };

// ---------------------------------------------------------------------------------------------------

    template <class T, class U>
    constexpr bool operator==(const optional<T> &lhs, const optional<U> &rhs) noexcept(noexcept(std::declval<T>() == std::declval<T>()))
    {        
        return bool(lhs) == bool(rhs) && (!bool(lhs) || *lhs == *rhs);
    }

    template <class T, class U> 
    constexpr bool operator!=(const optional<T> &lhs, const optional<U> &rhs) noexcept(noexcept(std::declval<T>() != std::declval<T>()))
    { 
        return !(lhs == rhs); 
    }
        
    template <class T, class U> 
    constexpr bool operator<(const optional<T> &lhs, const optional<U> &rhs) noexcept(noexcept(std::declval<T>() < std::declval<T>()))
    {
        return bool(rhs) && (!bool(lhs) || *lhs < *rhs);
    }

    template <class T, class U>
    constexpr bool operator>=(const optional<T> &lhs, const optional<U> &rhs) noexcept(noexcept(std::declval<T>() >= std::declval<T>()))
    {
        return !(lhs < rhs);
    }
    
    template <class T, class U>
    constexpr bool operator>(const optional<T> &lhs, const optional<U> &rhs) noexcept(noexcept(std::declval<T>() > std::declval<T>()))
    {
        return bool(lhs) && (!bool(rhs) || *lhs > *rhs);
    }

    template <class T, class U>
    constexpr bool operator<=(const optional<T> &lhs, const optional<U> &rhs) noexcept(noexcept(std::declval<T>() <= std::declval<T>()))
    {
        return !(lhs > rhs);
    }

// ---------------------------------------------------------------------------------------------------

    template <class T>
    constexpr bool operator==(const optional<T> &lhs, nullopt_t) noexcept
    {
        return !bool(lhs);
    }

    template <class T>
    constexpr bool operator==(nullopt_t, const optional<T> &rhs) noexcept 
    {
        return !bool(rhs);
    }
        
    template <class T>
    constexpr bool operator!=(const optional<T> &lhs, nullopt_t) noexcept 
    {
        return bool(lhs);
    }

    template <class T>
    constexpr bool operator!=(nullopt_t, const optional<T> &rhs) noexcept 
    {
        return bool(rhs);
    }
        
    template <class T>
    constexpr bool operator<(const optional<T> &, nullopt_t) noexcept 
    {
        return false;
    }
        
    template <class T>
    constexpr bool operator<(nullopt_t, const optional<T> &rhs) noexcept 
    {
        return bool(rhs);
    }

    template <class T>
    constexpr bool operator<=(const optional<T> &lhs, nullopt_t) noexcept
    {
        return !bool(lhs);
    }

    template <class T>
    constexpr bool operator<=(nullopt_t, const optional<T> &) noexcept 
    {
        return true;
    }

    template <class T>
    constexpr bool operator>(const optional<T> &lhs, nullopt_t) noexcept
    {
        return bool(lhs);
    }

    template <class T>
    constexpr bool operator>(nullopt_t, const optional<T> &) noexcept 
    {
        return false;
    }

    template <class T>
    constexpr bool operator>=(const optional<T> &, nullopt_t) noexcept 
    {
        return true;
    }

    template <class T>
    constexpr bool operator>=(nullopt_t, const optional<T> &rhs) noexcept 
    {
        return !rhs.has_value();
    }

// ---------------------------------------------------------------------------------------------------

    template <class T, class U>
    constexpr bool operator==(const optional<T> &lhs, const U &rhs) 
    {
        return bool(lhs) ? *lhs == rhs : false;
    }

    template <class T, class U>
    constexpr bool operator==(const U &lhs, const optional<T> &rhs) 
    {
        return bool(rhs) ? lhs == *rhs : false;
    }

    template <class T, class U>
    constexpr bool operator!=(const optional<T> &lhs, const U &rhs) 
    {
        return bool(lhs) ? *lhs != rhs : true;
    }

    template <class T, class U>
    constexpr bool operator!=(const U &lhs, const optional<T> &rhs) 
    {
        return bool(rhs) ? lhs != *rhs : true;
    }

    template <class T, class U>
    constexpr bool operator<(const optional<T> &lhs, const U &rhs) 
    {
        return bool(lhs) ? *lhs < rhs : true;
    }

    template <class T, class U>
    constexpr bool operator<(const U &lhs, const optional<T> &rhs) 
    {
        return bool(rhs) ? lhs < *rhs : false;
    }

    template <class T, class U>
    constexpr bool operator<=(const optional<T> &lhs, const U &rhs)
    {
        return bool(lhs) ? *lhs <= rhs : true;
    }

    template <class T, class U>
    constexpr bool operator<=(const U &lhs, const optional<T> &rhs)
    {
        return bool(rhs) ? lhs <= *rhs : false;
    }

    template <class T, class U>
    constexpr bool operator>(const optional<T> &lhs, const U &rhs) 
    {
        return bool(lhs) ? *lhs > rhs : false;
    }

    template <class T, class U>
    constexpr bool operator>(const U &lhs, const optional<T> &rhs)
    {
        return bool(rhs) ? lhs > *rhs : true;
    }

    template <class T, class U>
    constexpr bool operator>=(const optional<T> &lhs, const U &rhs) 
    {
        return bool(lhs) ? *lhs >= rhs : false;
    }

    template <class T, class U>
    constexpr bool operator>=(const U &lhs, const optional<T> &rhs)
    {
        return bool(rhs) ? lhs >= *rhs : true;
    }

// ---------------------------------------------------------------------------------------------------

    template <
        class T,
        std::enable_if_t<
        std::is_move_constructible<T>::value &&
        dlib::is_swappable<T>::value,
        bool> = true
    >
    void swap(dlib::optional<T>& lhs, dlib::optional<T>& rhs) noexcept(noexcept(lhs.swap(rhs))) 
    {
        return lhs.swap(rhs);
    }

// ---------------------------------------------------------------------------------------------------

    template< class T >
    constexpr dlib::optional<std::decay_t<T>> make_optional( T&& value )
    {
        return dlib::optional<std::decay_t<T>>(std::forward<T>(value));
    }

    template< class T, class... Args >
    constexpr dlib::optional<T> make_optional( Args&&... args )
    {
        return dlib::optional<T>(dlib::in_place, std::forward<Args>(args)...);
    }

    template< class T, class U, class... Args >
    constexpr dlib::optional<T> make_optional( std::initializer_list<U> il, Args&&... args )
    {
        return dlib::optional<T>(dlib::in_place, il, std::forward<Args>(args)...);
    }

// ---------------------------------------------------------------------------------------------------

}

#endif
