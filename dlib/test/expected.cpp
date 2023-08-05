// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <tuple>
#include <vector>
#include <dlib/expected.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.optional");

// ---------------------------------------------------------------------------------------------------
    
    void test_unexpected_trivial()
    {
        using std::swap;

        unexpected<int> e1{1};
        DLIB_TEST(e1.error() == 1);

        unexpected<int> e2{2};
        e1.swap(e2);
        DLIB_TEST(e1.error() == 2);
        DLIB_TEST(e2.error() == 1);
        swap(e1, e2);
        DLIB_TEST(e1.error() == 1);
        DLIB_TEST(e2.error() == 2);

        unexpected<int> e3 = e2;
        DLIB_TEST(e3.error() == 2);
        DLIB_TEST(e3 == e2);
    }
    
// ---------------------------------------------------------------------------------------------------

    struct unexpected_error_type1
    {
        unexpected_error_type1(int i_, float f_, std::string str_) : i{i_}, f{f_}, str{str_} {};

        int         i{0};
        float       f{0.0f};
        std::string str;
    };

    bool operator==(const unexpected_error_type1& a, const unexpected_error_type1& b)
    {
        return std::tie(a.i, a.f, a.str) == std::tie(b.i, b.f, b.str);
    }

    void test_unexpected_nontrival1()
    {
        using std::swap;

        unexpected<unexpected_error_type1> e1{in_place, 1, 3.1415f, "hello there"};
        DLIB_TEST(e1.error() == unexpected_error_type1(1, 3.1415f, "hello there"));

        unexpected<unexpected_error_type1> e2{e1};
        DLIB_TEST(e1 == e2);

        unexpected<unexpected_error_type1> e3 = e2;
        DLIB_TEST(e1 == e3);

        unexpected<unexpected_error_type1> e4{in_place, 0, 0.0f, ""};
        swap(e1, e4);
        DLIB_TEST(e1.error() == unexpected_error_type1(0, 0.0f, ""));
        DLIB_TEST(e4.error() == unexpected_error_type1(1, 3.1415f, "hello there"));
    }

// ---------------------------------------------------------------------------------------------------

    struct unexpected_error_type2
    {
        unexpected_error_type2(std::initializer_list<int> l, int i_) : v{l}, i{i_} {}

        std::vector<int> v;
        int i{0};
    };

    bool operator==(const unexpected_error_type2& a, const unexpected_error_type2& b)
    {
        return std::tie(a.v, a.i) == std::tie(b.v, b.i);
    }

    void test_unexpected_nontrival2()
    {
        using std::swap;

        unexpected<unexpected_error_type2> e1{in_place, {0, 1, 2, 3}, 42};
        DLIB_TEST(e1.error() == unexpected_error_type2({0, 1, 2, 3}, 42));

        unexpected<unexpected_error_type2> e2{e1};
        DLIB_TEST(e1 == e2);

        unexpected<unexpected_error_type2> e3 = e2;
        DLIB_TEST(e1 == e3);

        unexpected<unexpected_error_type2> e4{in_place, {0}, 0};
        swap(e1, e4);
        DLIB_TEST(e1.error() == unexpected_error_type2({0}, 0));
        DLIB_TEST(e4.error() == unexpected_error_type2({0, 1, 2, 3}, 42));
    }
    
// ---------------------------------------------------------------------------------------------------

    void test_expected_int_int()
    {
        expected<int, int> e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());
    }

// ---------------------------------------------------------------------------------------------------

    class expected_tester : public tester
    {
    public:
        expected_tester (
        ) :
            tester ("test_expected",
                    "Runs tests on the expected object")
        {}

        void perform_test (
        )
        {
            test_unexpected_trivial();
            test_unexpected_nontrival1();
            test_unexpected_nontrival2();
            test_expected_int_int();
        }
    } a;
}