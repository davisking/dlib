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
        static_assert(e9.error() == 3, "bad");
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

        // Converting copy constructor (don't know what the technical term for this constructor is)
        const dlib::expected<long, int> tmp{2};
        Expected e9{tmp};
        DLIB_TEST(e9);
        DLIB_TEST(e9.has_value());
        DLIB_TEST(*e9 == 2);
        DLIB_TEST(e9.value() == 2);
        DLIB_TEST(e9.value_or(3) == 2);

        // Converting move constructor
        Expected e10{std::move(tmp)};
        DLIB_TEST(e10);
        DLIB_TEST(e10.has_value());
        DLIB_TEST(*e10 == 2);
        DLIB_TEST(e10.value() == 2);
        DLIB_TEST(e10.value_or(3) == 2);
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

        // Copy assign
        Expected e5;
        e5 = e4;
        DLIB_TEST(e5);
        DLIB_TEST(e5.has_value());
        DLIB_TEST(e5.value() == 1);
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
        Expected e7{dlib::unexpected<nontrivial1>{in_place, 1, 3.1415f, "hello there"}};
        DLIB_TEST(!e7);
        DLIB_TEST(!e7.has_value());
        DLIB_TEST(e7.error() == nontrivial1(1, 3.1415f, "hello there"));
        int thrown{0};
        try {
            e7.value() = 0;
        } catch(const bad_expected_access<nontrivial1>& e) {
            thrown = 1;
            DLIB_TEST(e.error() == nontrivial1(1, 3.1415f, "hello there"));
        }
        DLIB_TEST(thrown == 1);
        DLIB_TEST(e7.value_or(2) == 2);

        // Construct from error
        Expected e8{unexpect, 1, 3.1415f, "hello there"};
        DLIB_TEST(!e8);
        DLIB_TEST(e8.error() == nontrivial1(1, 3.1415f, "hello there"));
        try {
            e8.value() = 42;
        } catch(const bad_expected_access<nontrivial1>& e) {
            thrown++;
            DLIB_TEST(e.error() == nontrivial1(1, 3.1415f, "hello there"));
        }
        DLIB_TEST(thrown == 2);
        DLIB_TEST(e8.value_or(3) == 3);

        // Assign from error
        Expected e9;
        e9 = dlib::unexpected<nontrivial1>{in_place, 1, 3.1415f, "hello there"};
        DLIB_TEST(!e9);
        DLIB_TEST(!e9.has_value());
        DLIB_TEST(e9.error() == nontrivial1(1, 3.1415f, "hello there"));
        try {
            e9.value() = 0;
        } catch(const bad_expected_access<nontrivial1>& e) {
            thrown++;
            DLIB_TEST(e.error() == nontrivial1(1, 3.1415f, "hello there"));
        }
        DLIB_TEST(thrown == 3);
        DLIB_TEST(e9.value_or(2) == 2);
    }

    void test_expected_void_nontrivial2()
    {
        using Expected = dlib::expected<void, nontrivial2>;
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

        // Move construction
        Expected e3{std::move(e2)};
        DLIB_TEST(e3);
        DLIB_TEST(e3.has_value());

        // Copy assign
        Expected e4;
        e4 = e3;
        DLIB_TEST(e4);
        DLIB_TEST(e4.has_value());

        // Move assign
        Expected e5;
        e5 = std::move(e4);
        DLIB_TEST(e5);
        DLIB_TEST(e5.has_value());

        // Construct from error
        Expected e6{dlib::unexpected<nontrivial2>{in_place, {0, 1, 2, 3}, 42}};
        DLIB_TEST(!e6);
        DLIB_TEST(!e6.has_value());
        DLIB_TEST(e6.error() == nontrivial2({0, 1, 2, 3}, 42));
        int thrown{0};
        try {
            e6.value();
        } catch(const bad_expected_access<nontrivial2>& e) {
            thrown = 1;
            DLIB_TEST(e.error() == nontrivial2({0, 1, 2, 3}, 42));
        }
        DLIB_TEST(thrown == 1);

        // Construct from error
        Expected e7{unexpect, {0, 1, 2, 3}, 42};
        DLIB_TEST(!e7);
        DLIB_TEST(e7.error() == nontrivial2({0, 1, 2, 3}, 42));
        try {
            e7.value();
        } catch(const bad_expected_access<nontrivial2>& e) {
            thrown++;
            DLIB_TEST(e.error() == nontrivial2({0, 1, 2, 3}, 42));
        }
        DLIB_TEST(thrown == 2);

        // Assign from error
        Expected e8;
        e8 = dlib::unexpected<nontrivial2>{in_place, {0, 1, 2, 3}, 42};
        DLIB_TEST(!e8);
        DLIB_TEST(e8.error() == nontrivial2({0, 1, 2, 3}, 42));
        try {
            e8.value();
        } catch(const bad_expected_access<nontrivial2>& e) {
            thrown++;
            DLIB_TEST(e.error() == nontrivial2({0, 1, 2, 3}, 42));
        }
        DLIB_TEST(thrown == 3);
    }

// ---------------------------------------------------------------------------------------------------

    void test_expected_nontrivial1_int()
    {
        using std::swap;
        using Expected = dlib::expected<nontrivial1, int>;

        const nontrivial1 val1(1, 3.1415f, "hello there");
        const nontrivial1 val2(2, 2.72f, "general kenobi");

        const auto check_is_val = [](const Expected& e, const nontrivial1& val1, const nontrivial1& val2)
        {
            DLIB_TEST(e);
            DLIB_TEST(e.has_value());
            DLIB_TEST(*e == val1);
            DLIB_TEST(e.value() == val1);
            DLIB_TEST(e.value_or(val2) == val1);
        };

        const auto check_is_error = [](const Expected& e, const nontrivial1& val, int error)
        {
            DLIB_TEST(!e);
            DLIB_TEST(!e.has_value());
            DLIB_TEST(e.error() == error);
            int thrown{0};
            try {
                e.value() == val;
            } catch(const bad_expected_access<int>& err) {
                ++thrown;
                DLIB_TEST(err.error() == error);
            }
            DLIB_TEST(thrown == 1);
        };

        // Default construction
        static_assert(!std::is_default_constructible<Expected>::value, "bad");

        // In-place construction
        Expected e1{in_place, 1, 3.1415f, "hello there"};
        check_is_val(e1, val1, val2);

        // Copy constructor
        Expected e2{e1};
        check_is_val(e1, val1, val2);
        check_is_val(e2, val1, val2);

        // Move constructor
        Expected e3{std::move(e2)};
        check_is_val(e3, val1, val2);
        DLIB_TEST(e2->str == "");

        // Copy assign
        Expected e4{val2};
        e4 = e3;
        check_is_val(e3, val1, val2);
        check_is_val(e4, val1, val2);

        // Move assign
        Expected e5{val2};
        e5 = std::move(e4);
        check_is_val(e5, val1, val2);
        DLIB_TEST(e4->str == "");

        // Construct from error
        Expected e6{unexpect, 42};
        check_is_error(e6, val1, 42);

        // Construct from unexpected
        Expected e7{dlib::unexpected<int>{42}};
        check_is_error(e7, val1, 42);
        
        // Assign from error
        Expected e8{val1};
        e8 = dlib::unexpected<int>{42};
        check_is_error(e8, val1, 42);

        // Swap
        Expected e9{val1};
        Expected e10{val2};
        swap(e9, e10);
        check_is_val(e9, val2, val1);
        check_is_val(e10, val1, val2);
        e9.swap(e10);
        check_is_val(e9, val1, val2);
        check_is_val(e10, val2, val1);

        // Emplace - not defined because according to the standard, it is only available when the in-place construction is noexcept. 
        // Don't understand why. Maybe there are some incredibly subtle bugs relating to strong exception guarantees which are hard to handle...
        // I'm falling out of love with exceptions. They add way too many complexities when designing vocabulary types and libraries in general.

        // TODO more stuff here
    }

// ---------------------------------------------------------------------------------------------------

    void test_expected_nontrivial1_nontrivial2()
    {
        using std::swap;
        using Expected = dlib::expected<nontrivial1, nontrivial2>;

        const nontrivial1 val1(1, 3.1415f, "hello there");
        const nontrivial1 val2(2, 2.72f, "general kenobi");
        const nontrivial2 err1({0,1}, 1);

        const auto check_is_val = [](const Expected& e, const nontrivial1& val1, const nontrivial1& val2)
        {
            DLIB_TEST(e);
            DLIB_TEST(e.has_value());
            DLIB_TEST(*e == val1);
            DLIB_TEST(e.value() == val1);
            DLIB_TEST(e.value_or(val2) == val1);
        };

        const auto check_is_error = [](const Expected& e, const nontrivial1& val, const nontrivial2& err)
        {
            DLIB_TEST(!e);
            DLIB_TEST(!e.has_value());
            DLIB_TEST(e.error() == err);
            int thrown{0};
            try {
                e.value() == val;
            } catch(const bad_expected_access<nontrivial2>& e) {
                ++thrown;
                DLIB_TEST(e.error() == err);
            }
            DLIB_TEST(thrown == 1);
        };

        // Default construction
        static_assert(!std::is_default_constructible<Expected>::value, "bad");

        // In-place construction
        Expected e1{in_place, 1, 3.1415f, "hello there"};
        check_is_val(e1, val1, val2);

        // Copy constructor
        Expected e2{e1};
        check_is_val(e1, val1, val2);
        check_is_val(e2, val1, val2);

        // Move constructor
        Expected e3{std::move(e2)};
        check_is_val(e3, val1, val2);
        DLIB_TEST(e2->str.empty());

        // Copy assign
        Expected e4{val2};
        e4 = e3;
        check_is_val(e3, val1, val2);
        check_is_val(e4, val1, val2);

        // Move assign
        Expected e5{val2};
        e5 = std::move(e4);
        check_is_val(e5, val1, val2);
        DLIB_TEST(e4->str == "");

        // Construct from error
        Expected e6{unexpect, {0,1}, 1};
        check_is_error(e6, val1, err1);

        // Construct from unexpected
        Expected e7{dlib::unexpected<nontrivial2>{in_place, {0,1}, 1}};
        check_is_error(e7, val1, err1);
        
        // Assign from error
        Expected e8{val1};
        e8 = dlib::unexpected<nontrivial2>{in_place, {0,1}, 1};
        check_is_error(e8, val1, err1);

        // Swap
        Expected e9{val1};
        Expected e10{val2};
        swap(e9, e10);
        check_is_val(e9, val2, val1);
        check_is_val(e10, val1, val2);
        e9.swap(e10);
        check_is_val(e9, val1, val2);
        check_is_val(e10, val2, val1);
        e10.swap(e8);
        check_is_val(e8, val2, val1);
        check_is_error(e10, val1, err1);
    }

// ---------------------------------------------------------------------------------------------------

    void test_monads()
    {
        // TODO
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
            test_expected_void_nontrivial2();
            test_expected_nontrivial1_int();
            test_expected_nontrivial1_nontrivial2();
        }
    } a;
}