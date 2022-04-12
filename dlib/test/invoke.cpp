// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include <string>
#include <memory>
#include <array>
#include <tuple>
#include <utility>
#include <dlib/invoke.h>
#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.invoke");

    // ----------------------------------------------------------------------------------------

    static const std::string run1_str1 = "hello there 1";
    static const std::string run1_str2 = "hello there 2";
    static const std::string run1_str3 = "hello there 3";
    static const std::string run1_str4 = "hello there 4";
    static const std::string run1_str5 = "hello there 5";

    void func_testargs(int i, std::string ref1, const std::string& ref2, const std::string& ref3, std::string& ref4)
    {
        DLIB_TEST(i > 0);
        DLIB_TEST(ref1 == run1_str1);
        DLIB_TEST(ref2 == run1_str2);
        DLIB_TEST(ref3 == run1_str3);
        DLIB_TEST(ref4 == run1_str4);
        ref4 = run1_str5;
    }

    int func_return_addition(int i, int j)
    {
        return i + j;
    }

    void test_functions()
    {
        static_assert(dlib::is_invocable<decltype(func_testargs), int, std::string, const std::string&, const std::string&, std::string&>::value, "should be invocable!");
        static_assert(dlib::is_invocable<decltype(func_testargs), int, std::string, std::string, const std::string&, std::string&>::value, "should be invocable!");
        static_assert(dlib::is_invocable<decltype(func_testargs), int, std::string, std::string, std::string, std::string&>::value, "should be invocable!");
        static_assert(dlib::is_invocable<decltype(func_testargs), int, std::string, std::string, std::string, std::reference_wrapper<std::string>>::value, "should be invocable!");

        {
            std::string str = run1_str4;
            dlib::invoke(func_testargs, 1, run1_str1, run1_str2, std::cref(run1_str3), std::ref(str));
            DLIB_TEST(str == run1_str5);
        }

        {
            std::string str = run1_str4;
            dlib::apply(func_testargs, std::make_tuple(1, run1_str1, run1_str2, std::cref(run1_str3), std::ref(str)));
            DLIB_TEST(str == run1_str5);
        }

        {
            for (int i = -10 ; i <= 10 ; i++)
            {
                for (int j = -10 ; j <= 10 ; j++)
                {
                    DLIB_TEST(dlib::invoke(func_return_addition, i, j) == (i+j));
                    DLIB_TEST(dlib::apply(func_return_addition, std::make_tuple(i, j)) == (i+j));
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------

    void test_lambdas()
    {
        {
            std::string str = run1_str4;
            dlib::invoke([](int i, std::string ref1, const std::string& ref2, const std::string& ref3, std::string& ref4) {
                DLIB_TEST(i > 0);
                DLIB_TEST(ref1 == run1_str1);
                DLIB_TEST(ref2 == run1_str2);
                DLIB_TEST(ref3 == run1_str3);
                DLIB_TEST(ref4 == run1_str4);
                ref4 = run1_str5;
            }, 1, run1_str1, run1_str2, std::cref(run1_str3), std::ref(str));
            DLIB_TEST(str == run1_str5);
        }

        {
            std::string str = run1_str4;
            dlib::apply([](int i, std::string ref1, const std::string& ref2, const std::string& ref3, std::string& ref4) {
                DLIB_TEST(i > 0);
                DLIB_TEST(ref1 == run1_str1);
                DLIB_TEST(ref2 == run1_str2);
                DLIB_TEST(ref3 == run1_str3);
                DLIB_TEST(ref4 == run1_str4);
                ref4 = run1_str5;
            }, std::make_tuple(1, run1_str1, run1_str2, std::cref(run1_str3), std::ref(str)));
            DLIB_TEST(str == run1_str5);
        }

        {
            for (int i = -10 ; i <= 10 ; i++)
            {
                for (int j = -10 ; j <= 10 ; j++)
                {
                    DLIB_TEST(dlib::invoke([](int i, int j) {return i + j;}, i, j) == (i+j));
                    DLIB_TEST(dlib::apply([](int i, int j) {return i + j;}, std::make_tuple(i,j)) == (i+j));
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------

    struct example_struct
    {
        example_struct(int i_ = 0) : i(i_) {}
        example_struct(const example_struct&) = delete;
        example_struct& operator=(const example_struct&) = delete;
        example_struct(example_struct&& other) : i(other.i) {other.i = 0;}
        example_struct& operator=(example_struct&& other) {i = other.i; other.i = 0; return *this;}

        int get_i() const {return i;}

        int i = 0;
    };

    void test_member_functions_and_data()
    {
        example_struct obj1(10);
        std::unique_ptr<example_struct> obj2(new example_struct(11));
        std::shared_ptr<example_struct> obj3(new example_struct(12));

        DLIB_TEST(dlib::invoke(&example_struct::get_i,    obj1) == 10);
        DLIB_TEST(dlib::invoke(&example_struct::i,        obj1) == 10);
        DLIB_TEST(dlib::invoke(&example_struct::get_i,    &obj1) == 10);
        DLIB_TEST(dlib::invoke(&example_struct::i,        &obj1) == 10);
        DLIB_TEST(dlib::invoke(&example_struct::get_i,    obj2) == 11);
        DLIB_TEST(dlib::invoke(&example_struct::i,        obj2) == 11);
        DLIB_TEST(dlib::invoke(&example_struct::get_i,    obj3) == 12);
        DLIB_TEST(dlib::invoke(&example_struct::i,        obj3) == 12);
    }

    // ----------------------------------------------------------------------------------------
    int return_int()
    {
        return 0;
    }

    int& return_int_ref()
    {
        static int i = 0;
        return i;
    }

    const int& return_int_const_ref()
    {
        static const int i = 0;
        return i;
    }

    int* return_int_pointer()
    {
        static int i = 0;
        return &i;
    }

    const int* return_int_const_pointer()
    {
        static const int i = 0;
        return &i;
    }

    void test_return_types()
    {
        static_assert(std::is_same<int,         dlib::invoke_result_t<decltype(return_int)>>::value, "bad type");
        static_assert(std::is_same<int&,        dlib::invoke_result_t<decltype(return_int_ref)>>::value, "bad type");
        static_assert(std::is_same<const int&,  dlib::invoke_result_t<decltype(return_int_const_ref)>>::value, "bad type");
        static_assert(std::is_same<int*,        dlib::invoke_result_t<decltype(return_int_pointer)>>::value, "bad type");
        static_assert(std::is_same<const int*,  dlib::invoke_result_t<decltype(return_int_const_pointer)>>::value, "bad type");

        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(&example_struct::get_i), const example_struct&>>::value, "bad type");
        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(&example_struct::get_i), example_struct&>>::value, "bad type");
        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(&example_struct::get_i), const example_struct*>>::value, "bad type");
        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(&example_struct::get_i), example_struct*>>::value, "bad type");
        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(&example_struct::get_i), std::unique_ptr<example_struct>>>::value, "bad type");
        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(&example_struct::get_i), std::shared_ptr<example_struct>>>::value, "bad type");

        static_assert(std::is_same<const int&,  dlib::invoke_result_t<decltype(&example_struct::i), const example_struct&>>::value, "bad type");
        static_assert(std::is_same<int&,        dlib::invoke_result_t<decltype(&example_struct::i), example_struct&>>::value, "bad type");
        static_assert(std::is_same<const int&,  dlib::invoke_result_t<decltype(&example_struct::i), const example_struct*>>::value, "bad type");
        static_assert(std::is_same<int&,        dlib::invoke_result_t<decltype(&example_struct::i), example_struct*>>::value, "bad type");
        static_assert(std::is_same<int&,        dlib::invoke_result_t<decltype(&example_struct::i), std::unique_ptr<example_struct>>>::value, "bad type");
        static_assert(std::is_same<int&,        dlib::invoke_result_t<decltype(&example_struct::i), std::shared_ptr<example_struct>>>::value, "bad type");

        auto lambda_func_return_int = []() -> int {return 0;};
        static_assert(std::is_same<int, dlib::invoke_result_t<decltype(lambda_func_return_int)>>::value, "bad type");
    }

    // ----------------------------------------------------------------------------------------

    void test_make_from_tuple()
    {
        struct multi_args_object
        {
            multi_args_object(int i_, int j_) : i(i_), j(j_) {}
            int i = 0;
            int j = 0;
        };

        {
            auto obj = dlib::make_from_tuple<multi_args_object>(std::make_tuple(1, 2));
            static_assert(std::is_same<decltype(obj), multi_args_object>::value, "bad type");
            DLIB_TEST(obj.i == 1);
            DLIB_TEST(obj.j == 2);
        }

        {
            std::array<int,2> a = {3, 4};
            auto obj = dlib::make_from_tuple<multi_args_object>(a);
            static_assert(std::is_same<decltype(obj), multi_args_object>::value, "bad type");
            DLIB_TEST(obj.i == 3);
            DLIB_TEST(obj.j == 4);
        }

        {
            auto obj = dlib::make_from_tuple<multi_args_object>(std::make_pair(5, 6));
            static_assert(std::is_same<decltype(obj), multi_args_object>::value, "bad type");
            DLIB_TEST(obj.i == 5);
            DLIB_TEST(obj.j == 6);
        }
    }

    // ----------------------------------------------------------------------------------------

    const char* func_return_c_string()
    {
        return "hello darkness my old friend";
    }

    struct obj_return_c_string
    {
        obj_return_c_string() = default;
        obj_return_c_string(const obj_return_c_string& rhs) = delete;
        obj_return_c_string(obj_return_c_string&& rhs)      = delete;

        const char* run()
        {
            return "i've come to talk with you again";
        }
    };

    void test_invoke_r()
    {
        {
            static_assert(dlib::is_invocable_r<std::string, decltype(func_return_c_string)>::value, "should be invocable");
            auto str = dlib::invoke_r<std::string>(func_return_c_string);
            static_assert(std::is_same<decltype(str), std::string>::value, "bad return type");
            DLIB_TEST(str == "hello darkness my old friend");
        }

        {
            obj_return_c_string obj;
            static_assert(dlib::is_invocable_r<std::string, decltype(&obj_return_c_string::run), decltype(obj)>::value, "should be invocable");
            auto str = dlib::invoke_r<std::string>(&obj_return_c_string::run, obj);
            static_assert(std::is_same<decltype(str), std::string>::value, "bad return type");
            DLIB_TEST(str == "i've come to talk with you again");
        }

        {
            obj_return_c_string obj;
            static_assert(dlib::is_invocable_r<std::string, decltype(&obj_return_c_string::run), decltype(&obj)>::value, "should be invocable");
            auto str = dlib::invoke_r<std::string>(&obj_return_c_string::run, &obj);
            static_assert(std::is_same<decltype(str), std::string>::value, "bad return type");
            DLIB_TEST(str == "i've come to talk with you again");
        }

        {
            auto obj = std::make_shared<obj_return_c_string>();
            static_assert(dlib::is_invocable_r<std::string, decltype(&obj_return_c_string::run), decltype(obj)>::value, "should be invocable");
            auto str = dlib::invoke_r<std::string>(&obj_return_c_string::run, obj);
            static_assert(std::is_same<decltype(str), std::string>::value, "bad return type");
            DLIB_TEST(str == "i've come to talk with you again");
        }

        {
            std::unique_ptr<obj_return_c_string> obj(new obj_return_c_string());
            static_assert(dlib::is_invocable_r<std::string, decltype(&obj_return_c_string::run), decltype(obj)>::value, "should be invocable");
            auto str = dlib::invoke_r<std::string>(&obj_return_c_string::run, obj);
            static_assert(std::is_same<decltype(str), std::string>::value, "bad return type");
            DLIB_TEST(str == "i've come to talk with you again");
        }

        {
            auto lambda_return_c_string = [] {
                return "because a vision softly creeping";
            };
            static_assert(dlib::is_invocable_r<std::string, decltype(lambda_return_c_string)>::value, "should be invocable");
            auto str = dlib::invoke_r<std::string>(lambda_return_c_string);
            static_assert(std::is_same<decltype(str), std::string>::value, "bad return type");
            DLIB_TEST(str == "because a vision softly creeping");
        }
    }

    // ----------------------------------------------------------------------------------------

    constexpr int multiply_ints(int i, int j)
    {
        return i*j;
    }

    struct constexpr_object
    {
        constexpr int multiply_ints(int i, int j) const
        {
            return i*j;
        }
    };

    void test_constexpr()
    {
        static_assert(dlib::invoke(multiply_ints, 2, 5) == 10, "this should be constexpr");
        static_assert(dlib::invoke_r<long>(multiply_ints, 2, 5) == 10, "this should be constexpr");
        constexpr constexpr_object constexpr_obj;
#if defined (_MSC_VER)
        constexpr_obj; // avoid warning C4101: 'constexpr_obj': unreferenced local variable
#endif
        static_assert(dlib::invoke(&constexpr_object::multiply_ints, constexpr_obj, 2, 5) == 10, "this should be constexpr");
        static_assert(dlib::invoke_r<long>(&constexpr_object::multiply_ints, constexpr_obj, 2, 5) == 10, "this should be constexpr");
    }

    // ----------------------------------------------------------------------------------------

    class invoke_tester : public tester
    {
    public:
        invoke_tester(
        ) : tester("test_invoke",
                   "Runs tests on dlib::invoke and dlib::apply")
        {}

        void perform_test(
        )
        {
            test_functions();
            test_lambdas();
            test_member_functions_and_data();
            test_return_types();
            test_make_from_tuple();
            test_invoke_r();
            test_constexpr();
        }
    } a;
}
