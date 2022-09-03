// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_METApROGRAMMING_Hh_
#define DLIB_METApROGRAMMING_Hh_

#include "constexpr_if.h"
#include "invoke.h"

namespace dlib
{
// ----------------------------------------------------------------------------------------
    namespace impl
    {
        template<typename List>
        struct add_one {};

        template<std::size_t... n>
        struct add_one<std::index_sequence<n...>> { using type = std::index_sequence<(n+1)...>; };

        template<typename List>
        using add_one_t = typename add_one<List>::type;
    }

// ----------------------------------------------------------------------------------------

    template <size_t... n>
    using compile_time_integer_list = std::index_sequence<n...>;

// ----------------------------------------------------------------------------------------

    template <size_t max>
    struct make_compile_time_integer_range
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object makes a compile_time_integer_list containing the integers in the range [1,max] inclusive.
                For example:
                    make_compile_time_integer_range<4>::type
                evaluates to:
                    compile_time_integer_list<1,2,3,4>
        !*/

        using type = impl::add_one_t<std::make_index_sequence<max>>;
    };

// ----------------------------------------------------------------------------------------

    template <typename Funct, typename... Args>
    bool call_if_valid(Funct&& f, Args&&... args) 
    /*!
        ensures
            - if f(std::forward<Args>(args)...) is a valid expression then we evaluate it and return
              true.  Otherwise we do nothing and return false.
    !*/
    {
        return switch_(bools(is_invocable<Funct, Args...>{}),
            [&](true_t, auto _) {
                _(std::forward<Funct>(f))(std::forward<Args>(args)...);
                return true;
            },
            [](auto...) {
                return false;
            }
        );
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_METApROGRAMMING_Hh_


