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
        DLIB_TEST(noexcept(e1.error()));
        DLIB_TEST(noexcept(as_const(e1).error()));

        dlib::unexpected<int> e2{2};
        e1.swap(e2);
        DLIB_TEST(e1.error() == 2);
        DLIB_TEST(e2.error() == 1);
        swap(e1, e2);
        DLIB_TEST(e1.error() == 1);
        DLIB_TEST(e2.error() == 2);
        DLIB_TEST(noexcept(swap(e1,e2)));
        DLIB_TEST(noexcept(e1.swap(e2)));

        dlib::unexpected<int> e3 = e2;
        DLIB_TEST(e3.error() == 2);
        DLIB_TEST(e3 == e2);
        DLIB_TEST(noexcept(e3 == e2));

        constexpr dlib::unexpected<int> e4{1};
        static_assert(e4.error() == 1, "bad");

        constexpr dlib::unexpected<int> e5{e4};
        static_assert(e5.error() == 1, "bad");

        constexpr dlib::unexpected<int> e6{std::move(e5)};
        static_assert(e6.error() == 1, "bad");

        auto e7 = make_unexpected(1);
        static_assert(std::is_same<decltype(e7), dlib::unexpected<int>>::value, "bad");
        DLIB_TEST(e7.error() == 1);

#ifdef __cpp_deduction_guides
        dlib::unexpected e8{2};
        static_assert(std::is_same<decltype(e8), dlib::unexpected<int>>::value, "bad");
        DLIB_TEST(e8.error() == 2);
#endif

        constexpr dlib::unexpected<int> e9 = make_unexpected(3);
        static_assert(e9.error() == 3);
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
        using Expected = dlib::expected<int, int>;
        static_assert(std::is_trivially_copy_constructible<Expected>::value,    "bad");
        static_assert(std::is_trivially_move_constructible<Expected>::value,    "bad");
        static_assert(std::is_trivially_destructible<Expected>::value,          "bad");
        static_assert(std::is_nothrow_default_constructible<Expected>::value,   "bad");
        static_assert(std::is_nothrow_copy_constructible<Expected>::value,      "bad");
        static_assert(std::is_nothrow_move_constructible<Expected>::value,      "bad");
        static_assert(std::is_nothrow_destructible<Expected>::value,            "bad");

        // Default construction
        Expected e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());
        DLIB_TEST(noexcept(e1.has_value()));

        // Construct from T
        Expected e2(1);
        DLIB_TEST(e2);
        DLIB_TEST(e2.has_value());
        DLIB_TEST(e2.value() == 1);
        DLIB_TEST(*e2 == 1);
        DLIB_TEST(e2.value_or(2) == 1);

        // Copy construction
        Expected e3{e2};
        DLIB_TEST(e3);
        DLIB_TEST(e3.has_value());
        DLIB_TEST(e3.value() == 1);
        DLIB_TEST(*e3 == 1);
        DLIB_TEST(e3.value_or(2) == 1);

        // Move construction
        Expected e4{std::move(e3)};
        DLIB_TEST(e4);
        DLIB_TEST(e4.has_value());
        DLIB_TEST(e4.value() == 1);
        DLIB_TEST(*e4 == 1);
        DLIB_TEST(e4.value_or(2) == 1);

        // Copy assign
        Expected e5;
        e5 = e4;
        DLIB_TEST(e5);
        DLIB_TEST(e5.has_value());
        DLIB_TEST(e4.value() == 1);
        DLIB_TEST(*e5 == 1);
        DLIB_TEST(e5.value_or(2) == 1);

        // Move assign
        Expected e6;
        e6 = std::move(e5);
        DLIB_TEST(e6);
        DLIB_TEST(e6.has_value());
        DLIB_TEST(e6.value() == 1);
        DLIB_TEST(*e6 == 1);
        DLIB_TEST(e6.value_or(2) == 1);

        // Construct from error
        Expected e7{dlib::unexpected<int>(1)};
        DLIB_TEST(!e7);
        DLIB_TEST(!e7.has_value());
        DLIB_TEST(e7.error() == 1);
        int thrown{0};
        try {
            e7.value() = 0;
        } catch(const bad_expected_access<int>& e) {
            thrown = 1;
            DLIB_TEST(e.error() == 1);
        }
        DLIB_TEST(thrown == 1);
        DLIB_TEST(e7.value_or(2) == 2);

        // Assign from error
        Expected e8;
        e8 = dlib::unexpected<int>(1);
        DLIB_TEST(!e8);
        DLIB_TEST(e8.error() == 1);
        try {
            e8.value() = 42;
        } catch(const bad_expected_access<int>& e) {
            thrown++;
            DLIB_TEST(e.error() == 1);
        }
        DLIB_TEST(thrown == 2);
        DLIB_TEST(e8.value_or(3) == 3);
    }

// ---------------------------------------------------------------------------------------------------

    void test_expected_void_int()
    {
        using Expected = dlib::expected<void, int>;
        static_assert(std::is_trivially_copy_constructible<Expected>::value,    "bad");
        static_assert(std::is_trivially_move_constructible<Expected>::value,    "bad");
        static_assert(std::is_trivially_destructible<Expected>::value,          "bad");
        static_assert(std::is_nothrow_default_constructible<Expected>::value,   "bad");
        static_assert(std::is_nothrow_copy_constructible<Expected>::value,      "bad");
        static_assert(std::is_nothrow_move_constructible<Expected>::value,      "bad");
        static_assert(std::is_nothrow_destructible<Expected>::value,            "bad");

        // Default construction
        Expected e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Copy construction
        Expected e2{e1};
        DLIB_TEST(e2);
        DLIB_TEST(e2.has_value());
    }

// ---------------------------------------------------------------------------------------------------

    void test_expected_int_nontrivial1()
    {
        using Expected = dlib::expected<int, nontrivial1>;
        // All verified on compiler explorer using std::expected https://godbolt.org/z/4GsM6h7Y5
        static_assert(!std::is_trivially_copy_constructible<Expected>::value,    "bad");
        static_assert(!std::is_trivially_move_constructible<Expected>::value,    "bad");
        static_assert(!std::is_trivially_destructible<Expected>::value,          "bad");
        static_assert(std::is_nothrow_default_constructible<Expected>::value,   "bad");
        static_assert(!std::is_nothrow_copy_constructible<Expected>::value,      "bad");
        static_assert(std::is_nothrow_move_constructible<Expected>::value,      "bad");
        static_assert(std::is_nothrow_destructible<Expected>::value,            "bad");

        // Default construction
        Expected e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Construct from T
        Expected e2(1);
        DLIB_TEST(e2);
        DLIB_TEST(e2.has_value());
        DLIB_TEST(e2.value() == 1);
        DLIB_TEST(*e2 == 1);
        DLIB_TEST(e2.value_or(2) == 1);

        // Copy construction
        Expected e3{e2};
        DLIB_TEST(e3);
        DLIB_TEST(e3.has_value());
        DLIB_TEST(e3.value() == 1);
        DLIB_TEST(*e3 == 1);
        DLIB_TEST(e3.value_or(2) == 1);

        // Move construction
        Expected e4{std::move(e3)};
        DLIB_TEST(e4);
        DLIB_TEST(e4.has_value());
        DLIB_TEST(e4.value() == 1);
        DLIB_TEST(*e4 == 1);
        DLIB_TEST(e4.value_or(2) == 1);
    }

    void test_expected_void_nontrivial1()
    {
        using Expected = dlib::expected<void, nontrivial1>;
        static_assert(!std::is_trivially_copy_constructible<Expected>::value,    "bad");
        static_assert(!std::is_trivially_move_constructible<Expected>::value,    "bad");
        static_assert(!std::is_trivially_destructible<Expected>::value,          "bad");
        static_assert(std::is_nothrow_default_constructible<Expected>::value,   "bad");
        static_assert(std::is_nothrow_destructible<Expected>::value,            "bad");

        // Default construction
        Expected e1;
        DLIB_TEST(e1);
        DLIB_TEST(e1.has_value());

        // Copy construction
        Expected e2{e1};
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