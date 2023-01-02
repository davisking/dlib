// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_FUNCTION_Hh_
#define DLIB_AnY_FUNCTION_Hh_

#include "../assert.h"
#include "../functional.h"
#include "any.h"
#include "any_function_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        class Storage, 
        class F
    > 
    class any_function_basic;

    template <
        class Storage,
        class R, 
        class... Args
    > 
    class any_function_basic<Storage, R(Args...)> 
    {
    private:
        template<class F>
        using is_valid = std::enable_if_t<!std::is_same<std::decay_t<F>, any_function_basic>::value &&
                                          dlib::is_invocable_r<R, F, Args...>::value,
                                          bool>;

        template<typename Func>
        static auto make_invoker()
        {
            return [](void* self, Args... args) -> R {
                return dlib::invoke(*reinterpret_cast<std::add_pointer_t<Func>>(self),
                                    std::forward<Args>(args)...);
            };
        }

        Storage str;
        R (*func)(void*, Args...) = nullptr;

    public:

        using result_type = R;
        
        constexpr any_function_basic(std::nullptr_t) noexcept {}
        constexpr any_function_basic()                                              = default;
        constexpr any_function_basic(const any_function_basic& other)               = default;
        constexpr any_function_basic& operator=(const any_function_basic& other)    = default;

        constexpr any_function_basic(any_function_basic&& other) 
        :   str{std::move(other.str)}, 
            func{std::exchange(other.func, nullptr)} 
        {
        }

        constexpr any_function_basic& operator=(any_function_basic&& other)
        {
            if (this != &other)
            {
                str     = std::move(other.str);
                func    = std::exchange(other.func, nullptr);
            }

            return *this;
        }

        template<class F, is_valid<F> = true>
        any_function_basic(
            F&& f
        ) : str{std::forward<F>(f)},
            func{make_invoker<F&&>()}
        {
        }

        template<class F, is_valid<F> = true>
        any_function_basic(
            F* f
        ) : str{f},
            func{make_invoker<F*>()}
        {
        }

        R operator()(Args... args) const {
            return func(const_cast<void*>(str.get_ptr()), std::forward<Args>(args)...);
        }

        void clear()                            { str.clear(); }
        void swap (any_function_basic& item)    { std::swap(*this, item); }
        bool is_empty()          const noexcept { return str.is_empty() || func == nullptr; }
        bool is_set()            const noexcept { return !is_empty(); }
        explicit operator bool() const noexcept { return is_set(); }

        template <typename T>     bool contains() const { return str.template contains<T>();}
        template <typename T>       T& cast_to()        { return str.template cast_to<T>(); }
        template <typename T> const T& cast_to() const  { return str.template cast_to<T>(); }
        template <typename T>       T& get()            { return str.template get<T>(); }
    };

// ----------------------------------------------------------------------------------------

    template <class T, class Storage, class F> 
    T& any_cast(any_function_basic<Storage, F>& a) { return a.template cast_to<T>(); }

    template <class T, class Storage, class F> 
    const T& any_cast(const any_function_basic<Storage, F>& a) { return a.template cast_to<T>(); }

// ----------------------------------------------------------------------------------------

    template <class F> 
    using any_function = any_function_basic<te::storage_sbo<16>, F>;

    template <class F> 
    using any_function_view = any_function_basic<te::storage_view, F>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_AnY_FUNCTION_Hh_

