// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SCOPE_H_
#define DLIB_SCOPE_H_ 

#include <utility>
#include <functional>
#include <type_traits>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template<class Fn>
    class scope_exit
    {
    private:
        Fn f_;
        bool active_{true};

    public:
        constexpr scope_exit()                                = delete;
        constexpr scope_exit(const scope_exit &)              = delete;
        constexpr scope_exit &operator=(const scope_exit &)   = delete;
        constexpr scope_exit &operator=(scope_exit &&)        = delete;

        constexpr scope_exit(scope_exit &&other) noexcept(std::is_nothrow_move_constructible<Fn>::value)
        : f_{std::move(other.f_)}, active_{std::exchange(other.active_, false)}
        {}

        template<
        class F, 
        std::enable_if_t<!std::is_same<std::decay_t<F>, scope_exit>::value, bool> = true
        >
        explicit scope_exit(F&& f) noexcept(std::is_nothrow_constructible<Fn,F>::value)
        : f_{std::forward<F>(f)}, active_{true}
        {}
    
        ~scope_exit() noexcept 
        {
            if (active_)
                f_();
        }

        void release() noexcept { active_ = false; }
    };

    template<class Fn>
    auto make_scope_exit(Fn&& f)
    {
        return scope_exit<std::decay_t<Fn>>(std::forward<Fn>(f));
    }

#ifdef __cpp_deduction_guides
    template<class Fn>
    scope_exit(Fn) -> scope_exit<Fn>;
#endif

// ----------------------------------------------------------------------------------------

    using scope_exit_erased = scope_exit<std::function<void()>>;

// ----------------------------------------------------------------------------------------

}

#endif //DLIB_SCOPE_H_