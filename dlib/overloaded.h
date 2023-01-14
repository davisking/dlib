// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_OVERLOADED_H_
#define DLIB_OVERLOADED_H_

#include <utility>
#include <type_traits>

namespace dlib
{
    namespace detail
    {
#if __cpp_fold_expressions
        template<typename ...Base>
        struct overloaded_helper : Base...
        {
            template<typename... T>
            constexpr overloaded_helper(T&& ... t)
            noexcept((std::is_nothrow_constructible<Base,T&&>::value && ...))
            : Base{std::forward<T>(t)}... {}

            using Base::operator()...;
        };
#else
        template<typename Base, typename ... BaseRest>
        struct overloaded_helper: Base, overloaded_helper<BaseRest...>
        {
            template<typename T, typename ... TRest>
            constexpr overloaded_helper(T&& t, TRest&& ...trest) 
            noexcept(std::is_nothrow_constructible<Base, T&&>::value && std::is_nothrow_constructible<overloaded_helper<BaseRest...>, TRest&&...>::value)
            : Base{std::forward<T>(t)},
              overloaded_helper<BaseRest...>{std::forward<TRest>(trest)...}
            {}

            using Base::operator();
            using overloaded_helper<BaseRest...>::operator();
        };

        template<typename Base>
        struct overloaded_helper<Base> : Base
        {
            template<typename T>
            constexpr overloaded_helper<Base>(T&& t) 
            noexcept(std::is_nothrow_constructible<Base, T&&>::value)
            : Base{std::forward<T>(t)}
            {}

            using Base::operator();
        };
#endif //__cpp_fold_expressions
    }
    
    template<typename... T>
    constexpr auto overloaded(T&&... t)
    /*!
        This is a helper function for combining many callable objects (usually lambdas), into
        an overload-able set. This can be used in visitor patterns like
            - dlib::type_safe_union::apply_to_contents()
            - dlib::visit()
            - dlib::for_each_type()
            - dlib::switch_()

        A picture paints a thousand words:

        using tsu = type_safe_union<int,float,std::string>;

        tsu a = std::string("hello there");

        std::string result;

        a.apply_to_contents(overloaded(
            [&result](int) {
                result = std::string("int");
            },
            [&result](float) {
                result = std::string("float");
            },
            [&result](const std::string& item) {
                result = item;
            }
        ));

        assert(result == "hello there");
        result = "";

        result = visit(overloaded(
            [](int) {
                return std::string("int");
            },
            [](float) {
                return std::string("float");
            },
            [](const std::string& item) {
                return item;
            }
        ), a);

        assert(result == "hello there");

        std::vector<int> type_ids;

        for_each_type(a, overloaded(
            [&type_ids](in_place_tag<int>, tsu& me) {
                type_ids.push_back(me.get_type_id<int>());
            },
            [&type_ids](in_place_tag<float>, tsu& me) {
                type_ids.push_back(me.get_type_id<float>());
            },
            [&type_ids](in_place_tag<std::string>, tsu& me) {
                type_ids.push_back(me.get_type_id<std::string>());
            }
        ));

        assert(type_ids == vector<int>({0,1,2}));
    !*/
    noexcept(std::is_nothrow_constructible<detail::overloaded_helper<std::decay_t<T>...>, T&&...>::value)
    {
        return detail::overloaded_helper<std::decay_t<T>...>{std::forward<T>(t)...};
    }
}

#endif //DLIB_OVERLOADED_H_
