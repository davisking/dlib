// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include <string>
#include <memory>
#include <dlib/invoke.h>
#include "tester.h"

namespace
{
    using namespace test;
    using namespace dlib;
    using namespace std;

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
        {
            std::string str = run1_str4;
            invoke(func_testargs, 1, run1_str1, run1_str2, std::cref(run1_str3), std::ref(str));
            DLIB_TEST(str == run1_str5);
        }

        {
            std::string str = run1_str4;
            apply(func_testargs, std::make_tuple(1, run1_str1, run1_str2, std::cref(run1_str3), std::ref(str)));
            DLIB_TEST(str == run1_str5);
        }

        {
            for (int i = -10 ; i <= 10 ; i++)
            {
                for (int j = -10 ; j <= 10 ; j++)
                {
                    DLIB_TEST(invoke(func_return_addition, i, j) == (i+j));
                    DLIB_TEST(apply(func_return_addition, std::make_tuple(i, j)) == (i+j));
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------

    void test_lambdas()
    {
        {
            std::string str = run1_str4;
            invoke([](int i, std::string ref1, const std::string& ref2, const std::string& ref3, std::string& ref4) {
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
            apply([](int i, std::string ref1, const std::string& ref2, const std::string& ref3, std::string& ref4) {
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
                    DLIB_TEST(invoke([](int i, int j) {return i + j;}, i, j) == (i+j));
                    DLIB_TEST(apply([](int i, int j) {return i + j;}, std::make_tuple(i,j)) == (i+j));
                }
            }
        }
    }

    // ----------------------------------------------------------------------------------------

    void test_member_functions_and_data()
    {
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

        {
            example_struct obj1(10);
            std::unique_ptr<example_struct> obj2(new example_struct(11));
            std::shared_ptr<example_struct> obj3(new example_struct(12));

            DLIB_TEST(invoke(&example_struct::get_i,    obj1) == 10);
            DLIB_TEST(invoke(&example_struct::i,        obj1) == 10);
            DLIB_TEST(invoke(&example_struct::get_i,    &obj1) == 10);
            DLIB_TEST(invoke(&example_struct::i,        &obj1) == 10);
            DLIB_TEST(invoke(&example_struct::get_i,    obj2) == 11);
            DLIB_TEST(invoke(&example_struct::i,        obj2) == 11);
            DLIB_TEST(invoke(&example_struct::get_i,    obj3) == 12);
            DLIB_TEST(invoke(&example_struct::i,        obj3) == 12);
        }
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
        }
    } a;
}
