// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_FUNCTION_H_
#define DLIB_FUNCTION_H_

#include <cstddef>
#include "te.h"
#include "invoke.h"

namespace dlib
{
    template <
        class Storage, 
        class F
    > 
    class function_basic;

    template <
        class Storage,
        class R, 
        class... Args
    > 
    class function_basic<Storage, R(Args...)> 
    {
    public:
        constexpr function_basic(std::nullptr_t) {}
        constexpr function_basic()                                          = default;
        constexpr function_basic(const function_basic& other)               = default;
        constexpr function_basic& operator=(const function_basic& other)    = default;
        constexpr function_basic(function_basic&& other)                    = default;        
        constexpr function_basic& operator=(function_basic&& other)         = default;

        template <
            typename F,
            std::enable_if_t<!std::is_same<std::decay_t<F>, function_basic>{} &&
                             dlib::is_invocable_r<R, F&&, Args...>{},
                             bool> = true
        >
        function_basic(
            F&& f
        ) : str{std::forward<F>(f)},
            func{make_invoker<F&&>()}
        {
        }

        template <
            typename F,
            std::enable_if_t<!std::is_same<F, function_basic>{} &&
                             dlib::is_invocable_r<R, F, Args...>{},
                             bool> = true
        >
        function_basic(
            F* f
        ) : str{f},
            func{make_invoker<F*>()}
        {
        }

        explicit operator bool() const noexcept
        {
            return !str.is_empty() && func;
        }

        R operator()(Args... args) const {
            return func(const_cast<void*>(str.get_ptr()), std::forward<Args>(args)...);
        }

    private:
        template<typename Func>
        static auto make_invoker()
        {
            return [](void* self, Args... args) -> R {
                return dlib::invoke(*reinterpret_cast<std::add_pointer_t<Func>>(self),
                                    std::forward<Args>(args)...);
            };
        }
        
        Storage str;
        te::managed_move_pointer<R(void*, Args...), true> func;
    };

    template <class F> 
    using function_heap  = function_basic<te::storage_heap, F>;

    template <class F, std::size_t Size, std::size_t Alignment = 8> 
    using function_stack = function_basic<te::storage_stack<Size, Alignment>, F>;
    
    template <class F, std::size_t Size, std::size_t Alignment = 8> 
    using function_sbo   = function_basic<te::storage_sbo<Size, Alignment>, F>;

    template <class F>
    using function_shared = function_basic<te::storage_shared, F>;

    template <class F> 
    using function_view  = function_basic<te::storage_view, F>;
}

#endif //DLIB_FUNCTION_H_
