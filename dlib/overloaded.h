// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OVERLOADED_H_
#define DLIB_OVERLOADED_H_

#include <utility>
#include <type_traits>

namespace dlib
{
#if __cpp_fold_expressions

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

#endif //__cpp_fold_expressions

    template<typename... T>
    auto overloaded(T&&... t)
    {
        return overloaded_helper<std::decay_t<T>...>{std::forward<T>(t)...};
    }
}

#endif //DLIB_OVERLOADED_H_
