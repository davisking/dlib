// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <string>
#include <dlib/constexpr_if.h>
#include <dlib/invoke.h>
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

    template<typename T>
    auto handle_type_and_return1(T obj)
    {
        return switch_type_<T>(
            case_type_<A>([&](auto _) {
                return _(obj).i;
            }),
            case_type_<B>([&](auto _) {
                return _(obj).f;
            }),
            case_type_<C>([&](auto _) {
                return _(obj).str;
            }),
            default_([&](auto _) {
                printf("Don't know what this type is\n");
            })
        );
    }

    template<typename T>
    auto handle_type_and_return2(T obj)
    {
        return switch_(
            case_<std::is_same<T,A>::value>([&](auto _) {
                return _(obj).i;
            }),
            case_<std::is_same<T,B>::value>([&](auto _) {
                return _(obj).f;
            }),
            case_<std::is_same<T,C>::value>([&](auto _) {
                return _(obj).str;
            }),
            default_([&](auto _) {
                printf("Don't know what this type is\n");
            })
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
    bool try_calling(Func&& f, Args&&... args)
    {
        return switch_(
            case_<is_invocable<Func, Args...>::value>([&](auto _) {
                _(std::forward<Func>(f))(std::forward<Args>(args)...);
                return true;
            }),
            default_([](auto) {
                return false;
            })
        );
    }

    void test_try_calling_example()
    {
        int value = 0;

        auto foo = [&](int a, int b) { value += a + b; };
        auto bar = [&](std::string) { value++; };
        auto baz = [&]() { value++; };

        DLIB_TEST(try_calling(baz));
        DLIB_TEST(value == 1);
        DLIB_TEST(!try_calling(foo));
        DLIB_TEST(value == 1);
        DLIB_TEST(!try_calling(bar));
        DLIB_TEST(value == 1);
        DLIB_TEST(try_calling(bar, "stuff"));
        DLIB_TEST(value == 2);
        DLIB_TEST(!try_calling(baz, "stuff"));
        DLIB_TEST(value == 2);
        DLIB_TEST(try_calling(foo, 3, 1));
        DLIB_TEST(value == 6);
        DLIB_TEST(!try_calling(bar, 3, 1));
        DLIB_TEST(value == 6);
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
        }
    } a;

}