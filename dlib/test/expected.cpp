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

    struct nontrivial1
    {
        nontrivial1(int i_, float f_, std::string str_) : i{i_}, f{f_}, str{str_} {};

        int         i{0};
        float       f{0.0f};
        std::string str;
    };

    bool operator==(const nontrivial1& a, const nontrivial1& b)
    {
        return std::tie(a.i, a.f, a.str) == std::tie(b.i, b.f, b.str);
    }

    void test_unexpected_nontrival1()
    {
        using std::swap;

        dlib::unexpected<nontrivial1> e1{in_place, 1, 3.1415f, "hello there"};
        DLIB_TEST(e1.error() == nontrivial1(1, 3.1415f, "hello there"));

        dlib::unexpected<nontrivial1> e2{e1};
        DLIB_TEST(e1 == e2);

        dlib::unexpected<nontrivial1> e3 = e2;
        DLIB_TEST(e1 == e3);

        dlib::unexpected<nontrivial1> e4{in_place, 0, 0.0f, ""};
        swap(e1, e4);
        DLIB_TEST(e1.error() == nontrivial1(0, 0.0f, ""));
        DLIB_TEST(e4.error() == nontrivial1(1, 3.1415f, "hello there"));
    }

// ---------------------------------------------------------------------------------------------------

    struct nontrivial2
    {
        nontrivial2(std::initializer_list<int> l, int i_) : v{l}, i{i_} {}

        std::vector<int> v;
        int i{0};
    };

    bool operator==(const nontrivial2& a, const nontrivial2& b)
    {
        return std::tie(a.v, a.i) == std::tie(b.v, b.i);
    }

    void test_unexpected_nontrival2()
    {
        using std::swap;

        dlib::unexpected<nontrivial2> e1{in_place, {0, 1, 2, 3}, 42};
        DLIB_TEST(e1.error() == nontrivial2({0, 1, 2, 3}, 42));

        dlib::unexpected<nontrivial2> e2{e1};
        DLIB_TEST(e1 == e2);

        dlib::unexpected<nontrivial2> e3 = e2;
        DLIB_TEST(e1 == e3);

        dlib::unexpected<nontrivial2> e4{in_place, {0}, 0};
        swap(e1, e4);
        DLIB_TEST(e1.error() == nontrivial2({0}, 0));
        DLIB_TEST(e4.error() == nontrivial2({0, 1, 2, 3}, 42));
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

        // Construct from error
        dlib::expected<int, int> e5{dlib::unexpected<int>(1)};
        DLIB_TEST(!e5);
        DLIB_TEST(!e5.has_value());
        DLIB_TEST(e5.error() == 1);
        int thrown{0};
        try {
            e5.value() = 0;
        } catch(const std::exception&) {
            thrown = 1;
        }
        DLIB_TEST(thrown == 1);
        DLIB_TEST(e5.value_or(2) == 2);
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

    void test_expected_int_nontrivial1()
    {
        // Default construction
        dlib::expected<int, nontrivial1> e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Construct from T
        dlib::expected<int, nontrivial1> e2(1);
        DLIB_TEST(e2);
        DLIB_TEST(e2.has_value());
        DLIB_TEST(e2.value() == 1);
        DLIB_TEST(*e2 == 1);
        DLIB_TEST(e2.value_or(2) == 1);

        // Copy construction
        dlib::expected<int, nontrivial1> e3{e2};
        DLIB_TEST(e3);
        DLIB_TEST(e3.has_value());
        DLIB_TEST(e3.value() == 1);
        DLIB_TEST(*e3 == 1);
        DLIB_TEST(e3.value_or(2) == 1);

        // Move construction
        dlib::expected<int, nontrivial1> e4{std::move(e3)};
        DLIB_TEST(e4);
        DLIB_TEST(e4.has_value());
        DLIB_TEST(e4.value() == 1);
        DLIB_TEST(*e4 == 1);
        DLIB_TEST(e4.value_or(2) == 1);
    }

    void test_expected_void_nontrivial1()
    {
        // Default construction
        dlib::expected<void, nontrivial1> e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Copy construction
        dlib::expected<void, nontrivial1> e2{e1};
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
            test_expected_int_nontrivial1();
            test_expected_void_nontrivial1();
        }
    } a;
}