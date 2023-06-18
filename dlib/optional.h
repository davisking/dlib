// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OPTIONAL_H
#define DLIB_OPTIONAL_H

#include <exception>
#include <initializer_list>
#include "type_traits.h"
#include "utility.h"

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
        explicit constexpr nullopt_t(int) noexcept {}
    };

    static constexpr nullopt_t nullopt{int{}};

// ---------------------------------------------------------------------------------------------------

    template<class T>
    class optional
    {
    private:

// ---------------------------------------------------------------------------------------------------

        template<class U>
        using is_assignable_base = And<
            !std::is_constructible<T,       dlib::optional<U>&>::value,
            !std::is_constructible<T, const dlib::optional<U>&>::value,
            !std::is_constructible<T,       dlib::optional<U>&&>::value,
            !std::is_constructible<T, const dlib::optional<U>&&>::value,
            !std::is_convertible<      dlib::optional<U>&, T>::value,
            !std::is_convertible<const dlib::optional<U>&, T>::value,
            !std::is_convertible<      dlib::optional<U>&&, T>::value,
            !std::is_convertible<const dlib::optional<U>&&, T>::value,
            !std::is_assignable<T&,       dlib::optional<U>&>::value,
            !std::is_assignable<T&, const dlib::optional<U>&>::value,
            !std::is_assignable<T&,       dlib::optional<U>&&>::value,
            !std::is_assignable<T&, const dlib::optional<U>&&>::value
        >;

        template<class U>
        using is_copy_assignable_from = std::enable_if_t<
            is_assignable_base<U>::value &&
            std::is_constructible<T, const U&>::value &&
            std::is_assignable<T&, const U&>::value,
            bool
        >;

        template<class U>
        using is_move_assignable_from = std::enable_if_t<
            is_assignable_base<U>::value &&
            std::is_constructible<T, U>::value &&
            std::is_assignable<T&, U>::value,
            bool
        >;

// ---------------------------------------------------------------------------------------------------

    public:

// ---------------------------------------------------------------------------------------------------

        constexpr optional() noexcept = default;

// ---------------------------------------------------------------------------------------------------

        constexpr optional(dlib::nullopt_t) noexcept = default;

// ---------------------------------------------------------------------------------------------------

        constexpr optional& operator=(dlib::nullopt_t) noexcept
        {
            reset();
            return *this;
        }

// ---------------------------------------------------------------------------------------------------

        template <
          class U,
          std::enable_if_t<std::is_constructible<T, const U&>::value, bool> = true
        >
        constexpr optional (const optional<U>& other) noexcept(std::is_nothrow_constructible<T, const U&>::value)
        {
            if (other)
                construct(*other);                
        }

// ---------------------------------------------------------------------------------------------------

        template < 
          class U,
          is_copy_assignable_from<U> = true
        >
        constexpr optional& operator=( const optional<U>& other ) noexcept(std::is_nothrow_constructible<T, const U&>::value &&
                                                                           std::is_nothrow_assignable<T&, const U&>::value)
        {
            if (!has_value() && other)
                construct(*other);
            else if (has_value() && other)
                **this = *other;
            else if (has_value() && !other)
                reset();
            return *this;
        }

// ---------------------------------------------------------------------------------------------------

        template <
          class U,
          std::enable_if_t<std::is_constructible<T, U&&>::value, bool> = true
        >
        constexpr optional( optional<U>&& other ) noexcept(std::is_nothrow_constructible<T, U&&>::value)
        {
            if (other.has_value())
                construct(std::move(other.value()));  
        }

// ---------------------------------------------------------------------------------------------------

        template < 
          class U,
          is_move_assignable_from<U> = true
        >
        constexpr optional& operator=( optional<U>&& other ) noexcept(std::is_nothrow_constructible<T, U&&>::value &&
                                                                      std::is_nothrow_assignable<T&, U&&>::value)
        {
            if (!has_value() && other)
                construct(std::move(*other));
            else if (has_value() && other)
                **this = std::move(*other);
            else if (has_value() && !other)
                reset();
            return *this;
        }

// ---------------------------------------------------------------------------------------------------

        template < 
          class... Args,
          std::enable_if_t<std::is_constructible<T, Args...>::value, bool> = true
        >
        constexpr explicit optional ( 
            dlib::in_place_t,
            Args&&... args
        ) 
        noexcept(std::is_nothrow_constructible<T, Args&&...>::value)
        {
            construct(std::forward<Args>(args)...);
        }

// ---------------------------------------------------------------------------------------------------

        template< class... Args >
        constexpr T& emplace( Args&&... args )
        {
            reset();
            construct(std::forward<Args>(args)...);
            return **this;
        }

// ---------------------------------------------------------------------------------------------------

        template < 
          class U, 
          class... Args,
          std::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value, bool> = true
        >
        constexpr explicit optional ( 
            dlib::in_place_t,
            std::initializer_list<U> ilist,
            Args&&... args 
        )
        {
            construct(ilist, std::forward<Args>(args)...);
        }

// ---------------------------------------------------------------------------------------------------

        template< class U, class... Args >
        constexpr T& emplace( std::initializer_list<U> ilist, Args&&... args )
        {
            reset();
            construct(ilist, std::forward<Args>(args)...);
            return **this;
        }

// ---------------------------------------------------------------------------------------------------

        template < 
          class U,
          class U_ = dlib::remove_cvref_t<U>,
          std::enable_if_t<std::is_constructible<T, U&&>::value &&
                           !std::is_same<U_, dlib::in_place_t>::value &&
                           !std::is_same<U_, dlib::optional<T>>::value, bool> = true
        >
        constexpr optional( U&& u ) noexcept(std::is_nothrow_constructible<T, U&&>::value)
        {
            construct(std::forward<U>(u));
        }

// ---------------------------------------------------------------------------------------------------

        template < 
          class U,
          class U_ = dlib::remove_cvref_t<U>,
          std::enable_if_t<std::is_constructible<T, U>::value &&
                           std::is_assignable<T&, U>::value &&
                           !std::is_same<U_, dlib::optional<T>>::value,bool> = true
        >
        constexpr optional& operator=( U&& u ) noexcept(std::is_nothrow_constructible<T, U&&>::value &&
                                                        std::is_nothrow_assignable<T,U&&>::value)
        {
            if (has_value())
                **this = std::forward<U>(u);
            else
                construct(std::forward<U>(u));
        }

// ---------------------------------------------------------------------------------------------------

        ~optional() noexcept
        {
            reset();
        }

// ---------------------------------------------------------------------------------------------------

        constexpr void reset() noexcept(std::is_nothrow_destructible<T>::value)
        {
            if (has_value())
            {
                (**this).~T();
                has_value_ = false;
            }
        }

// ---------------------------------------------------------------------------------------------------

        constexpr explicit  operator bool() const noexcept { return has_value_; }

// ---------------------------------------------------------------------------------------------------

        constexpr bool      has_value()     const noexcept { return has_value_; }

// ---------------------------------------------------------------------------------------------------

        constexpr const T*  operator->()    const noexcept { return reinterpret_cast<const T*>(&mem); }
        constexpr T*        operator->()          noexcept { return reinterpret_cast<const T*>(&mem); }

// ---------------------------------------------------------------------------------------------------

        constexpr const T&  operator*()     const noexcept { return *reinterpret_cast<const T*>(&mem); }
        constexpr T&        operator*()           noexcept { return *reinterpret_cast<const T*>(&mem); }

// ---------------------------------------------------------------------------------------------------

        constexpr T& value() 
        {
            if (has_value())
                return **this;
            throw bad_optional_access();
        }
        
        constexpr const T& value() const
        {
            if (has_value())
                return **this;
            throw bad_optional_access();
        }

// ---------------------------------------------------------------------------------------------------

        template< class U >
        constexpr T value_or( U&& u ) const&
        {
            return has_value_ ? **this : static_cast<T>(std::forward<U>(u));
        }

        template< class U >
        constexpr T value_or( U&& u ) &&
        {
            return has_value_ ? std::move(**this) : static_cast<T>(std::forward<U>(u));
        }

// ---------------------------------------------------------------------------------------------------

        constexpr void swap( optional& other ) noexcept(std::is_nothrow_move_constructible<T>::value &&
                                                        dlib::is_nothrow_swappable<T>::value)
        {
            if (has_value() && other.has_value())
                std::swap(**this, *other);
            else if (has_value() && !other.has_value())
                other = std::move(*this);
            else if (!has_value() && other.has_value())
                *this = std::move(other);
        }

// ---------------------------------------------------------------------------------------------------

    private:

        static_assert(!std::is_reference<T>::value, "optional of reference is forbidden");

// ---------------------------------------------------------------------------------------------------

        template< class... Args>
        constexpr void construct(Args&&... args) noexcept(std::is_nothrow_constructible<T,Args&&...>::value)
        {
            new (&mem) T{std::forward<Args>(args)...};
            has_value_ = true;
        }

// ---------------------------------------------------------------------------------------------------

        std::aligned_storage_t<sizeof(T), alignof(T)> mem;
        bool has_value_{false};
    };

// ---------------------------------------------------------------------------------------------------

    template<class T>
    void swap(dlib::optional<T>& a, dlib::optional<T>& b)
    {
        a.swap(b);
    }

// ---------------------------------------------------------------------------------------------------

}

#endif