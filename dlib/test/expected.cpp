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

        dlib::unexpected<int> e1{1};
        DLIB_TEST(e1.error() == 1);

        dlib::unexpected<int> e2{2};
        e1.swap(e2);
        DLIB_TEST(e1.error() == 2);
        DLIB_TEST(e2.error() == 1);
        swap(e1, e2);
        DLIB_TEST(e1.error() == 1);
        DLIB_TEST(e2.error() == 2);

        dlib::unexpected<int> e3 = e2;
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

        dlib::unexpected<unexpected_error_type1> e1{in_place, 1, 3.1415f, "hello there"};
        DLIB_TEST(e1.error() == unexpected_error_type1(1, 3.1415f, "hello there"));

        dlib::unexpected<unexpected_error_type1> e2{e1};
        DLIB_TEST(e1 == e2);

        dlib::unexpected<unexpected_error_type1> e3 = e2;
        DLIB_TEST(e1 == e3);

        dlib::unexpected<unexpected_error_type1> e4{in_place, 0, 0.0f, ""};
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

        dlib::unexpected<unexpected_error_type2> e1{in_place, {0, 1, 2, 3}, 42};
        DLIB_TEST(e1.error() == unexpected_error_type2({0, 1, 2, 3}, 42));

        dlib::unexpected<unexpected_error_type2> e2{e1};
        DLIB_TEST(e1 == e2);

        dlib::unexpected<unexpected_error_type2> e3 = e2;
        DLIB_TEST(e1 == e3);

        dlib::unexpected<unexpected_error_type2> e4{in_place, {0}, 0};
        swap(e1, e4);
        DLIB_TEST(e1.error() == unexpected_error_type2({0}, 0));
        DLIB_TEST(e4.error() == unexpected_error_type2({0, 1, 2, 3}, 42));
    }
    
// ---------------------------------------------------------------------------------------------------

    void test_expected_int_int()
    {
        // Default construction
        dlib::expected<int, int> e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Construct from T
        dlib::expected<int, int> e2(1);
        DLIB_TEST(e2);
        DLIB_TEST(e2.has_value());
        DLIB_TEST(e2.value() == 1);
        DLIB_TEST(*e2 == 1);
        DLIB_TEST(e2.value_or(2) == 1);

        // Copy construction
        dlib::expected<int, int> e3{e2};
        DLIB_TEST(e3);
        DLIB_TEST(e3.has_value());
        DLIB_TEST(e3.value() == 1);
        DLIB_TEST(*e3 == 1);
        DLIB_TEST(e3.value_or(2) == 1);

        // Move construction
        dlib::expected<int, int> e4{std::move(e3)};
        DLIB_TEST(e4);
        DLIB_TEST(e4.has_value());
        DLIB_TEST(e4.value() == 1);
        DLIB_TEST(*e4 == 1);
        DLIB_TEST(e4.value_or(2) == 1);
    }

// ---------------------------------------------------------------------------------------------------

    void test_expected_void_int()
    {
        // Default construction
        dlib::expected<void, int> e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Copy construction
        dlib::expected<void, int> e2{e1};
        DLIB_TEST(e2);
        DLIB_TEST(e2.has_value());
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
            test_expected_void_int();
        }
    } a;
}