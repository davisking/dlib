// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <string>
#include <dlib/constexpr_if.h>
#include <dlib/functional.h>
#include <cstdio>
#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.constexpr_if");

    struct A
    {
        int i;
    };

    struct B
    {
        float f;
    };

    struct C
    {
        std::string str;
    };

    struct D
    {
        int i;
        void set_i(int j) {i = j;}
    };

    template<typename T>
    auto handle_type_and_return1(T obj)
    {
        return switch_(types_<T>{},
            [&](types_<A>, auto _) {
                return _(obj).i;
            },
            [&](types_<B>, auto _) {
                return _(obj).f;
            },
            [&](types_<C>, auto _) {
                return _(obj).str;
            },
            [&](auto...) {
                printf("Don't know what this type is\n");
            }
        );
    }

    template<typename T>
    auto handle_type_and_return2(T obj)
    {
        return switch_(bools(std::is_same<T,A>{}, std::is_same<T,B>{}, std::is_same<T,C>{}),
            [&](true_t, auto, auto, auto _) {
               return _(obj).i;
            },
            [&](auto, true_t, auto, auto _) {
               return _(obj).f;
            },
            [&](auto, auto, true_t, auto _) {
               return _(obj).str;
            },
            [&](auto...) {
               printf("Don't know what this type is\n");
            }
        );
    }

    void test_switch_type()
    {
        A a{1};
        B b{2.5f};
        C c{"hello there!"};

        {
            auto ret = handle_type_and_return1(a);
            static_assert(std::is_same<decltype(ret), int>::value, "failed test");
            DLIB_TEST(ret == a.i);
        }

        {
            auto ret = handle_type_and_return2(a);
            static_assert(std::is_same<decltype(ret), int>::value, "failed test");
            DLIB_TEST(ret == a.i);
        }

        {
            auto ret = handle_type_and_return1(b);
            static_assert(std::is_same<decltype(ret), float>::value, "failed test");
            DLIB_TEST(ret == b.f);
        }

        {
            auto ret = handle_type_and_return2(b);
            static_assert(std::is_same<decltype(ret), float>::value, "failed test");
            DLIB_TEST(ret == b.f);
        }

        {
            auto ret = handle_type_and_return1(c);
            static_assert(std::is_same<decltype(ret), std::string>::value, "failed test");
            DLIB_TEST(ret == c.str);
        }

        {
            auto ret = handle_type_and_return2(c);
            static_assert(std::is_same<decltype(ret), std::string>::value, "failed test");
            DLIB_TEST(ret == c.str);
        }
    }

    template <typename Func, typename... Args>
    bool try_invoke(Func&& f, Args&&... args)
    {
        return switch_(bools(is_invocable<Func, Args...>{}),
            [&](true_t, auto _) {
                _(std::forward<Func>(f))(std::forward<Args>(args)...);
                return true;
            },
            [](auto...) {
                return false;
            }
        );
    }

    void test_try_invoke()
    {
        int value = 0;

        auto foo = [&]{ value++; };
        auto bar = [&](int i) { value += i; };
        auto baz = [&](int i, int j) { value += (i+j); };

        DLIB_TEST(try_invoke(foo));
        DLIB_TEST(value == 1);
        DLIB_TEST(!try_invoke(foo, 1));
        DLIB_TEST(value == 1);
        DLIB_TEST(!try_invoke(foo, 1, 2));
        DLIB_TEST(value == 1);

        DLIB_TEST(!try_invoke(bar));
        DLIB_TEST(value == 1);
        DLIB_TEST(try_invoke(bar, 1));
        DLIB_TEST(value == 2);
        DLIB_TEST(!try_invoke(bar, 1, 2));
        DLIB_TEST(value == 2);

        DLIB_TEST(!try_invoke(baz));
        DLIB_TEST(value == 2);
        DLIB_TEST(!try_invoke(baz, 1));
        DLIB_TEST(value == 2);
        DLIB_TEST(try_invoke(baz, 1, 2));
        DLIB_TEST(value == 5);
    }

    template<typename T>
    using set_i_pred = decltype(std::declval<T>().set_i(int{}));

    template<typename T>
    bool try_set_i(T& obj, int i)
    {
        return switch_(bools(is_detected<set_i_pred, T>{}),
            [&](true_t, auto _) {
                _(obj).set_i(i);
                return true;
            },
            [](auto...){
                return false;
            }
        );
    }

    void test_set_i()
    {
        A a{1};
        D d{1};
        DLIB_TEST(!try_set_i(a, 2));
        DLIB_TEST(a.i == 1);
        DLIB_TEST(try_set_i(d, 2));
        DLIB_TEST(d.i == 2);
    }

    class constexpr_if_test : public tester
    {
    public:
        constexpr_if_test (
        ) : tester ("test_constexpr_if",
                    "Runs tests on the C++14 approximation of C++17 if constexpr() statements but better.")
        {}

        void perform_test (
        )
        {
            test_switch_type();
            test_try_invoke();
            test_set_i();
        }
    } a;

}
