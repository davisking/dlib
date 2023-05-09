// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IF_CONSTEXPR_H
#define DLIB_IF_CONSTEXPR_H

#include "overloaded.h"
#include "type_traits.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------

    namespace detail
    {
        const auto _ = [](auto&& arg) -> decltype(auto) { return std::forward<decltype(arg)>(arg); };
    }

// ----------------------------------------------------------------------------------------

    template<bool... v>
    constexpr auto bools(std::integral_constant<bool, v>...)
    /*!
        ensures
            - returns a type list of compile time booleans.
    !*/
    {
        return types_<std::integral_constant<bool,v>...>{};
    }

    using true_t  = types_<std::true_type>;
    using false_t = types_<std::false_type>;

// ----------------------------------------------------------------------------------------

    template <
        typename... T,
        typename... Cases
    >
    constexpr decltype(auto) switch_(
        types_<T...> /*meta_obj*/,
        Cases&&...   cases
    )
    /*!
        requires
            - meta_obj combines a set of initial types. These are used as compile-time initial conditions.
            - cases is a set of overload-able conditional branches.
            - at least one of the cases is callable given meta_obj.
            - each case statement has signature auto(types_<>..., auto _) where _ is an identity function
              with identical behaviour to std::identity. This is used to make each generic lambda artificially
              dependent on the function body. This allows semantic analysis of the lambdas to be performed AFTER
              the correct lambda is chosen depending on meta_obj. This is the crucial bit that makes switch_() behave
              in a similar way to "if constexpr()" in C++17. Make sure to use _ on one of the objects in the lambdas.
        ensures
            - calls the correct conditional branch.
            - the correct conditional branch is selected at compile-time.
            - Note, each branch can return different types, and the return type of the switch_() function
              is that of the compile-time selected branch.

            Here is an example:

            template<typename T>
            auto perform_correct_action(T& obj)
            {
                return switch(
                    types_<T>{},
                    [&](types_<A>, auto _) {
                        return _(obj).set_something_specific_to_A_and_return_something();
                    },
                    [&](types_<B>, auto _) {
                        return _(obj).set_something_specific_to_B_and_return_something();
                    },
                    [&](auto...) {
                        // Default case statement. Do something sensible.
                        return false;
                    }
                );
            }

            Here is another example:

            template<typename T>
            auto transfer_state(T& a, T& b)
            {
                return switch(
                    bools(std::is_move_constructible<T>{}, std::is_copy_constructible<T>{}),
                    [&](true_t, auto, auto _) {
                        // T is both move-constructible. Copy semantics can be anything
                        a = std::move(_(b));
                        return move_tag{}; // Just for fun, we return different types in each branch.
                    },
                    [&](auto, true_t, auto _) {
                        // T is copy-constructible, Move semantics can be anything. Though in this case,
                        // if it had been move-constructible, the first branch would have been selected.
                        // So in this case, it is not move-constructible.
                        a = _(b);
                        return copy_tag{};
                    },
                    [&](auto...) {
                        // Default case statement
                        return dont_care_tag{};
                    }
                );
            }
      !*/
    {
        return overloaded(std::forward<Cases>(cases)...)(types_<T>{}..., detail::_);
    }

// ----------------------------------------------------------------------------------------

}

#endif //DLIB_IF_CONSTEXPR_H