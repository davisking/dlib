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
        static_assert(noexcept(e1.error()), "bad");
        static_assert(noexcept(as_const(e1).error()), "bad");

        dlib::unexpected<int> e2{2};
        e1.swap(e2);
        DLIB_TEST(e1.error() == 2);
        DLIB_TEST(e2.error() == 1);
        swap(e1, e2);
        DLIB_TEST(e1.error() == 1);
        DLIB_TEST(e2.error() == 2);
        DLIB_TEST(noexcept(swap(e1,e2)));
        DLIB_TEST(noexcept(e1.swap(e2)));
        static_assert(noexcept(swap(e1,e2)), "bad");
        static_assert(noexcept(e1.swap(e2)), "bad");

        dlib::unexpected<int> e3 = e2;
        DLIB_TEST(e3.error() == 2);
        DLIB_TEST(e3 == e2);
        DLIB_TEST(noexcept(e3 == e2));
        static_assert(noexcept(e3 == e2), "bad");
        
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
        DLIB_TEST(e2 == 1);
        DLIB_TEST(1 == e2);
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
        DLIB_TEST(e7 == dlib::unexpected<int>(1));
        DLIB_TEST(dlib::unexpected<int>(1) == e7);
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
                bool dummy = e.value() == val;
                (void)dummy;
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
                bool dummy = e.value() == val;
                (void)dummy;
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
        auto ret = dlib::expected<int,int>{1}
            .and_then([](int i)  { return dlib::expected<long,int>(i+1); })
            .and_then([](long l) { return dlib::expected<std::string,int>(std::to_string(l)); })
            .and_then([](const std::string& str) { return dlib::expected<float,int>(std::stof(str)); });
        
        static_assert(std::is_same<decltype(ret), dlib::expected<float,int>>::value, "bad");
        DLIB_TEST(ret);
        DLIB_TEST(ret.has_value());
        DLIB_TEST(*ret == 2.0f);

        auto ret2 = ret
            .and_then([](float) -> dlib::expected<void,int> { return {}; })
            .and_then([] { return dlib::expected<double,int>(2.0); });

        static_assert(std::is_same<decltype(ret2), dlib::expected<double,int>>::value, "bad");
        DLIB_TEST(ret2);
        DLIB_TEST(ret2.has_value());
        DLIB_TEST(*ret2 == 2.0);

        auto ret3 = dlib::expected<int, int>{1}
            .transform([](int i)  -> long  { return i + 1; })
            .transform([](long j) -> float { return j * 2;  });

        static_assert(std::is_same<decltype(ret3), dlib::expected<float,int>>::value, "bad");
        DLIB_TEST(ret3);
        DLIB_TEST(ret3.has_value());
        DLIB_TEST(*ret3 == 4);

        auto ret5 = dlib::expected<int, int>{1}
            .and_then([](int i)     { return dlib::expected<int,int>{unexpect, 2}; })
            .transform([](int i)    { DLIB_TEST_MSG(false, "This shouldn't get called"); return i; })
            .or_else([](int e)      { DLIB_TEST(e == 2); return dlib::expected<int,int>(unexpect, e); } );
        
        static_assert(std::is_same<decltype(ret5), dlib::expected<int,int>>::value, "bad");
        DLIB_TEST(!ret5);
        DLIB_TEST(ret5.error() == 2);

        auto ret6 = dlib::expected<int, int>{1}
            .and_then([](int i)     { return expected<int,int>{unexpect, 2}; })
            .transform([](int i)    { DLIB_TEST_MSG(false, "This shouldn't get called"); })
            .or_else([](int e)      { DLIB_TEST(e == 2); return expected<void,int>(unexpect, e); } );
        
        static_assert(std::is_same<decltype(ret6), dlib::expected<void,int>>::value, "bad");
        DLIB_TEST(!ret6);
        DLIB_TEST(ret6.error() == 2);

        auto ret7 = dlib::expected<int, int>{1}
            .and_then([](int i)         { return expected<int,int>{unexpect, 2}; })
            .transform([](int i)        { DLIB_TEST_MSG(false, "This shouldn't get called"); })
            .transform_error([](int e)  { DLIB_TEST(e == 2); return e;} );
        
        static_assert(std::is_same<decltype(ret7), dlib::expected<void,int>>::value, "bad");
        DLIB_TEST(!ret7);
        DLIB_TEST(ret7.error() == 2);
    }

// ---------------------------------------------------------------------------------------------------

    void test_semantics()
    {
        // Checked against std::expected https://godbolt.org/z/61rTGbPrh

        static int i{0};
        static int j{0};
        static int k{0};
        static int l{0};
        static int m{0};
        static int n{0};

        const auto reset = [&] {
            i = j = k = l = m = n = 0;
        };

        const auto check = [&](int i_, int j_, int k_, int l_, int m_, int n_) {
            DLIB_TEST(i == i_);
            DLIB_TEST(j == j_);
            DLIB_TEST(k == k_);
            DLIB_TEST(l == l_);
            DLIB_TEST(m == m_);
            DLIB_TEST(n == n_);
        };

        struct dummy
        {
            dummy()                        noexcept {i++;}
            dummy(const dummy&)            noexcept {j++;}
            dummy(dummy&&)                 noexcept {k++;}
            dummy& operator=(const dummy&) noexcept {l++; return *this;}
            dummy& operator=(dummy&&)      noexcept {m++; return *this;}
            ~dummy()                       noexcept {n++;}
        };

        using Expected1 = dlib::expected<dummy, int>;

        Expected1 a{};
        check(1,0,0,0,0,0); reset();

        Expected1 b{dlib::in_place};
        check(1,0,0,0,0,0); reset();

        Expected1 c{b};
        check(0,1,0,0,0,0); reset();

        Expected1 d{std::move(c)};
        check(0,0,1,0,0,0); reset();

        Expected1 e;
        dummy tmp;
        reset();
        e = tmp;
        check(0,0,0,1,0,0); reset();

        Expected1 f;
        reset();
        e = dummy{};
        check(1,0,0,0,1,1); reset();

        Expected1 g;
        reset();
        g = f;
        check(0,0,0,1,0,0); reset();

        Expected1 h;
        reset();
        h = std::move(g);
        check(0,0,0,0,1,0); reset();

        Expected1 o{dlib::unexpect, 0};
        check(0,0,0,0,0,0); reset();

        Expected1 p{dlib::unexpected<int>(1)};
        check(0,0,0,0,0,0); reset();

        Expected1 q{dlib::unexpected<long>(2)};
        check(0,0,0,0,0,0); reset();

        dlib::unexpected<long> tmp2{2};
        Expected1 r{tmp2};
        check(0,0,0,0,0,0); reset();

        dlib::expected<dummy, long> tmp3;
        reset();
        Expected1 s{tmp3};
        check(0,1,0,0,0,0); reset();

        Expected1 t{std::move(tmp3)};
        check(0,0,1,0,0,0); reset();

        Expected1 u;
        reset();
        u = tmp3;
        check(0,1,0,0,1,1); reset();

        Expected1 v;
        reset();
        u = std::move(tmp3);
        check(0,0,1,0,1,1); reset();

        Expected1 w;
        reset();
        w.emplace();
        check(1,0,0,0,0,1); reset();

        using Expected2 = dlib::expected<int, dummy>;

        Expected2 aa;
        check(0,0,0,0,0,0); reset();

        Expected2 ab{aa};
        check(0,0,0,0,0,0); reset();

        Expected2 ac{std::move(ab)};
        check(0,0,0,0,0,0); reset();

        Expected2 ad;
        reset();
        ad = ac;
        check(0,0,0,0,0,0); reset();

        Expected2 ae;
        reset();
        ae = std::move(ad);
        check(0,0,0,0,0,0); reset();

        Expected2 af{1};
        check(0,0,0,0,0,0); reset();

        Expected2 ag;
        ag = 1;
        check(0,0,0,0,0,0); reset();

        Expected2 ah{dlib::expected<long, dummy>{2}};
        check(0,0,0,0,0,0); reset();

        Expected2 ai{dlib::unexpect};
        check(1,0,0,0,0,0); reset();

        Expected2 aj{dlib::unexpect, dummy{}};
        check(1,0,1,0,0,1); reset();

        Expected2 ak{dlib::unexpected<dummy>{dummy{}}};
        check(1,0,2,0,0,2); reset();
    }

// ---------------------------------------------------------------------------------------------------

    void test_strong_exception_guarantees()
    {
        struct can_throw
        {
            can_throw(int i_) : i{i_} {}
            can_throw(const can_throw&) {throw std::runtime_error("");}
            can_throw(can_throw&&) {throw std::runtime_error("");}
            can_throw& operator=(const can_throw&)  = default;
            can_throw& operator=(can_throw&&)       = default;
            ~can_throw()                            = default;
            int i{0};
        };

        using Expected = dlib::expected<can_throw, int>;

        Expected a{dlib::unexpect, 1};
        DLIB_TEST(!a);
        DLIB_TEST(a.error() == 1);

        int thrown{0};
        try {
            a = can_throw{2};
        } catch(...) {
            ++thrown;
        }
        DLIB_TEST(thrown==1);
        DLIB_TEST(!a);
        DLIB_TEST(a.error() == 1);

        Expected b{2};
        DLIB_TEST(b);
        DLIB_TEST(b.value().i == 2);
        try {
            a = b;
        } catch(...) {
            ++thrown;
        }
        DLIB_TEST(thrown==2);
        DLIB_TEST(!a);
        DLIB_TEST(a.error() == 1);

        Expected c{3};
        DLIB_TEST(c);
        DLIB_TEST(c.value().i == 3);
        try {
            a = std::move(c);
        } catch(...) {
            ++thrown;
        }
        DLIB_TEST(thrown==3);
        DLIB_TEST(!a);
        DLIB_TEST(a.error() == 1);
    }

// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------
// MSVC tests taken from https://github.com/microsoft/STL/blob/main/tests/std/tests/P0323R12_expected/test.cpp
// ---------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------

    enum class IsDefaultConstructible : bool { Not, Yes };
    enum class IsTriviallyCopyConstructible : bool { Not, Yes };
    enum class IsTriviallyMoveConstructible : bool { Not, Yes };
    enum class IsTriviallyDestructible : bool { Not, Yes };

    enum class IsNothrowConstructible : bool { Not, Yes };
    enum class IsNothrowCopyConstructible : bool { Not, Yes };
    enum class IsNothrowMoveConstructible : bool { Not, Yes };
    enum class IsNothrowCopyAssignable : bool { Not, Yes };
    enum class IsNothrowMoveAssignable : bool { Not, Yes };
    enum class IsNothrowConvertible : bool { Not, Yes };
    enum class IsNothrowComparable : bool { Not, Yes };
    enum class IsNothrowSwappable : bool { Not, Yes };

    enum class IsExplicitConstructible : bool { Not, Yes };

    template <typename E>
    constexpr bool IsYes(const E e) noexcept {
        return e == E::Yes;
    }

    namespace test_unexpected 
    {
        struct convertible 
        {
            constexpr convertible() = default;
            constexpr convertible(const int val) noexcept : _val(val) {}
            constexpr bool operator==(const int other) const noexcept {return other == _val;}
            int _val = 0;
        };

        template <
          bool nothrowCopyConstructible, 
          bool nothrowMoveConstructible,
          bool nothrowComparable
        >
        struct test
        {
            static constexpr bool copy_construction_is_noexcept = nothrowCopyConstructible;
            static constexpr bool move_construction_is_noexcept = nothrowMoveConstructible;
            static constexpr bool compare_is_noexcept           = nothrowComparable;

            struct test_error
            {
                constexpr test_error(const int& val)                                noexcept(copy_construction_is_noexcept) : _val(val) {}
                constexpr test_error(int&& val)                                     noexcept(move_construction_is_noexcept) : _val(val) {}
                constexpr test_error(std::initializer_list<int>, const int& val)    noexcept(copy_construction_is_noexcept) : _val(val) {}
                constexpr test_error(std::initializer_list<int>, int&& val)         noexcept(move_construction_is_noexcept) : _val(val) {}
                constexpr test_error(const convertible& other)                      noexcept(copy_construction_is_noexcept) : _val(other._val) {}
                constexpr test_error(convertible&& other)                           noexcept(move_construction_is_noexcept) : _val(other._val) {}
                constexpr bool operator==(const test_error& right) const            noexcept(compare_is_noexcept) { return _val == right._val;}
                constexpr bool operator==(const convertible& right) const           noexcept(compare_is_noexcept) { return _val == right._val; }
                int _val = 0;
            };

            using Unexpect = dlib::unexpected<test_error>;

            static void run()
            {
                using std::swap;

                // [expected.un.ctor]
                const int& input = 1;
                Unexpect in_place_lvalue_constructed{in_place, input};
                static_assert(noexcept(Unexpect{in_place, input}) == copy_construction_is_noexcept, "bad");
                DLIB_TEST(in_place_lvalue_constructed == Unexpect{test_error{1}});

                Unexpect in_place_rvalue_constructed{in_place, 42};
                static_assert(noexcept(Unexpect{in_place, 42}) == move_construction_is_noexcept, "bad");
                DLIB_TEST(in_place_rvalue_constructed == Unexpect{test_error{42}});

                Unexpect in_place_ilist_lvalue_constructed{in_place, {2}, input};
                static_assert(noexcept(Unexpect{in_place, {2}, input}) == copy_construction_is_noexcept, "bad");
                DLIB_TEST(in_place_ilist_lvalue_constructed == Unexpect{test_error{1}});

                Unexpect in_place_ilist_rvalue_constructed{in_place, {2}, 1337};
                static_assert(noexcept(Unexpect{in_place, {2}, 1337}) == move_construction_is_noexcept, "bad");
                DLIB_TEST(in_place_ilist_rvalue_constructed == Unexpect{test_error{1337}});

                Unexpect base_error_constructed{test_error{3}};
                static_assert(noexcept(Unexpect{test_error{3}}) == move_construction_is_noexcept, "bad");
                DLIB_TEST(base_error_constructed.error()._val == 3);

                Unexpect conversion_error_constructed{convertible{4}};
                static_assert(noexcept(Unexpect{convertible{4}}) == move_construction_is_noexcept, "bad");
                DLIB_TEST(conversion_error_constructed.error()._val == 4);

                Unexpect brace_error_constructed{{5}};
                static_assert(noexcept(Unexpect{{5}}) == move_construction_is_noexcept, "bad");
                DLIB_TEST(brace_error_constructed.error()._val == 5);

                // [expected.un.eq]
                DLIB_TEST(in_place_lvalue_constructed == in_place_lvalue_constructed);
                DLIB_TEST(in_place_lvalue_constructed != in_place_rvalue_constructed);
                static_assert(noexcept(in_place_lvalue_constructed == in_place_lvalue_constructed) == compare_is_noexcept, "bad");
                static_assert(noexcept(in_place_lvalue_constructed != in_place_lvalue_constructed) == compare_is_noexcept, "bad");

                const auto converted = unexpected<convertible>{convertible{3}};
                DLIB_TEST(base_error_constructed == converted);
                DLIB_TEST(conversion_error_constructed != converted);
                static_assert(noexcept(base_error_constructed == converted) == compare_is_noexcept, "bad");
                static_assert(noexcept(conversion_error_constructed != converted) == compare_is_noexcept, "bad");

                // [expected.un.swap]
                in_place_lvalue_constructed.swap(in_place_rvalue_constructed);
                DLIB_TEST(in_place_lvalue_constructed == Unexpect{test_error{42}});
                DLIB_TEST(in_place_rvalue_constructed == Unexpect{test_error{1}});
                static_assert(noexcept(in_place_lvalue_constructed.swap(in_place_rvalue_constructed)), "bad");

                swap(base_error_constructed, conversion_error_constructed);
                DLIB_TEST(base_error_constructed == Unexpect{test_error{4}});
                DLIB_TEST(conversion_error_constructed == Unexpect{test_error{3}});
                static_assert(noexcept(swap(base_error_constructed, conversion_error_constructed)), "bad");

                // [expected.un.obs]
                auto&& lvalue_error = base_error_constructed.error();
                DLIB_TEST(lvalue_error == test_error{4});
                static_assert(std::is_same<decltype(lvalue_error), test_error&>::value, "bad");

                auto&& rvalue_error = std::move(conversion_error_constructed).error();
                DLIB_TEST(rvalue_error == test_error{3});
                static_assert(std::is_same<decltype(rvalue_error), test_error&&>::value, "bad");

                auto&& const_lvalue_error = as_const(in_place_lvalue_constructed).error();
                DLIB_TEST(const_lvalue_error == test_error{42});
                static_assert(std::is_same<decltype(const_lvalue_error), const test_error&>::value, "bad");

                auto&& const_rvalue_error = std::move(as_const(in_place_ilist_lvalue_constructed)).error();
                DLIB_TEST(const_rvalue_error == test_error{1});
                static_assert(std::is_same<decltype(const_rvalue_error), const test_error&&>::value, "bad");

                // deduction guide
#ifdef __cpp_deduction_guides
                unexpected deduced(test_error{42});
                static_assert(std::is_same<decltype(deduced), Unexpect>::value, "bad");
#endif
            }
        };

        void test_all()
        {
            test<false, false, false>::run();
            test<false, false, true>::run();
            test<false, true, false>::run();
            test<false, true, true>::run();
            test<true, false, false>::run();
            test<true, false, true>::run();
            test<true, true, false>::run();
            test<true, true, true>::run();
        }
    } // namespace test_unexpected

    namespace test_unexpect 
    {
        auto copy = unexpect;
        static_assert(std::is_same<decltype(copy), unexpect_t>::value, "bad");
        static_assert(std::is_trivial<unexpect_t>::value, "bad");
        static_assert(std::is_empty<unexpect_t>::value, "bad");
    } // namespace test_unexpect

    namespace test_expected 
    {
        constexpr void test_aliases()
        {
            struct value_tag {};
            struct error_tag {};

            {
                using Expected = expected<value_tag, error_tag>;
                static_assert(std::is_same<typename Expected::value_type, value_tag>::value, "bad");
                static_assert(std::is_same<typename Expected::error_type, error_tag>::value, "bad");
                static_assert(std::is_same<typename Expected::unexpected_type, dlib::unexpected<error_tag>>::value, "bad");
                static_assert(std::is_same<typename Expected::rebind<int>, expected<int, error_tag>>::value, "bad");
            }

            {
                using Expected = expected<void, error_tag>;
                static_assert(std::is_same<typename Expected::value_type, void>::value, "bad");
                static_assert(std::is_same<typename Expected::error_type, error_tag>::value, "bad");
                static_assert(std::is_same<typename Expected::unexpected_type, dlib::unexpected<error_tag>>::value, "bad");
                static_assert(std::is_same<typename Expected::rebind<int>, expected<int, error_tag>>::value, "bad");
            }
        }

        struct payload_default_constructor 
        {
            constexpr payload_default_constructor() : _val(42) {}
            constexpr bool operator==(const int val) const noexcept {return _val == val;}

            int _val = 0;
        };

        struct payload_no_default_constructor
        {
            constexpr payload_no_default_constructor() = delete;
            constexpr bool operator==(const int val) const noexcept {return _val == val;}
            int _val = 0;
        };

        void test_default_constructors() 
        {
            static_assert(std::is_default_constructible<dlib::expected<payload_default_constructor, int>>::value, "bad");
            static_assert(!std::is_default_constructible<dlib::expected<payload_no_default_constructor, int>>::value, "bad");
            // we only care about payload type
            static_assert(std::is_default_constructible<dlib::expected<int, payload_default_constructor>>::value, "bad");
            static_assert(std::is_default_constructible<dlib::expected<void, payload_default_constructor>>::value, "bad");
            static_assert(std::is_default_constructible<dlib::expected<int, payload_no_default_constructor>>::value, "bad");
            static_assert(std::is_default_constructible<dlib::expected<void, payload_no_default_constructor>>::value, "bad");

            const dlib::expected<payload_default_constructor, int> defaulted;
            DLIB_TEST(defaulted);
            DLIB_TEST(defaulted.value() == 42);
        }

        template <
          bool triviallyCopyConstructible,
          bool nothrowCopyConstructible,
          bool copyConstructible = triviallyCopyConstructible || nothrowCopyConstructible
        >
        struct payload_copy_constructor 
        {
            constexpr payload_copy_constructor()                                           = default;
            constexpr payload_copy_constructor& operator=(const payload_copy_constructor&) = delete;
            constexpr payload_copy_constructor(const payload_copy_constructor&) noexcept(copyConstructible) : _val(42) {}
            constexpr bool operator==(const int val) const noexcept { return _val == val; }
            int _val = 0;
        };

        template <
          bool nothrowCopyConstructible
        >
        struct payload_copy_constructor<true,nothrowCopyConstructible,true>
        {
            constexpr payload_copy_constructor()                                           = default;
            constexpr payload_copy_constructor& operator=(const payload_copy_constructor&) = delete;
            constexpr payload_copy_constructor(const payload_copy_constructor&)            = default;
            constexpr bool operator==(const int val) const noexcept { return _val == val; }
            int _val = 0;
        };

        template <
          bool triviallyCopyConstructible,
          bool nothrowCopyConstructible
        >
        void test_copy_constructors() 
        {
            constexpr bool should_be_noexcept = triviallyCopyConstructible || nothrowCopyConstructible;

            using payload = payload_copy_constructor<triviallyCopyConstructible,nothrowCopyConstructible>;

            { // Check payload type
                using Expected = expected<payload, int>;
                static_assert(std::is_trivially_copy_constructible<Expected>::value == triviallyCopyConstructible, "bad");
                static_assert(std::is_copy_constructible<Expected>::value, "bad");

                const Expected with_value{in_place};
                const Expected from_value{with_value};
                DLIB_TEST(from_value);
                DLIB_TEST(from_value.value() == (triviallyCopyConstructible ? 0 : 42));
                static_assert(noexcept(Expected{with_value}) == should_be_noexcept, "bad");

                const Expected with_error{unexpect};
                const Expected from_error{with_error};
                DLIB_TEST(!from_error);
                DLIB_TEST(from_error.error() == 0);
                static_assert(noexcept(Expected{with_error}) == should_be_noexcept, "bad");
            }

            { // Check error type
                using Expected = dlib::expected<int, payload>;
                static_assert(std::is_trivially_copy_constructible<Expected>::value == triviallyCopyConstructible, "bad");
                static_assert(std::is_copy_constructible<Expected>::value, "bad");

                const Expected with_value{in_place};
                const Expected from_value{with_value};
                DLIB_TEST(from_value);
                DLIB_TEST(from_value.value() == 0);
                static_assert(noexcept(Expected{with_value}) == should_be_noexcept, "bad");

                const Expected with_error{unexpect};
                const Expected from_error{with_error};
                DLIB_TEST(!from_error);
                DLIB_TEST(from_error.error() == (triviallyCopyConstructible ? 0 : 42));
                static_assert(noexcept(Expected{with_error}) == should_be_noexcept, "bad");
            }

            { // Check void payload
                using Expected = expected<void, payload>;
                static_assert(std::is_trivially_copy_constructible<Expected>::value == triviallyCopyConstructible, "bad");
                static_assert(std::is_copy_constructible<Expected>::value, "bad");

                const Expected with_value{in_place};
                const Expected from_value{with_value};
                DLIB_TEST(from_value);
                // TODO: fix me
                //static_assert(noexcept(Expected{with_value}) == should_be_noexcept, "bad");

                const Expected with_error{unexpect};
                const Expected from_error{with_error};
                DLIB_TEST(!from_error);
                DLIB_TEST(from_error.error() == (triviallyCopyConstructible ? 0 : 42));
                // static_assert(noexcept(Expected{with_error}) == should_be_noexcept, "bad");
            }

            { // ensure we are not copy constructible if either the payload or the error are not
                struct not_copy_constructible {
                    not_copy_constructible(const not_copy_constructible&) = delete;
                };

                static_assert(!std::is_copy_constructible<expected<not_copy_constructible, int>>::value, "bad");
                static_assert(!std::is_copy_constructible<expected<int, not_copy_constructible>>::value, "bad");
                static_assert(!std::is_copy_constructible<expected<void, not_copy_constructible>>::value, "bad");
            }
        }

        template <
          bool triviallyMoveConstructible,
          bool nothrowMoveConstructible,
          bool nothrow = triviallyMoveConstructible || nothrowMoveConstructible
        >
        struct payload_move_constructor
        {
            constexpr payload_move_constructor()                                      = default;
            constexpr payload_move_constructor(const payload_move_constructor&)       = default;
            constexpr payload_move_constructor& operator=(payload_move_constructor&&) = delete;
            constexpr payload_move_constructor(payload_move_constructor&&) noexcept(nothrow) : _val(42) {}
            constexpr bool operator==(const int val) const noexcept {return _val == val;}
            int _val = 0;
        };

        template <
          bool nothrowMoveConstructible
        >
        struct payload_move_constructor<true,nothrowMoveConstructible,true>
        {
            constexpr payload_move_constructor()                                      = default;
            constexpr payload_move_constructor(const payload_move_constructor&)       = default;
            constexpr payload_move_constructor& operator=(payload_move_constructor&&) = delete;
            constexpr payload_move_constructor(payload_move_constructor&&)            = default;
            constexpr bool operator==(const int val) const noexcept {return _val == val;}
            int _val = 0;
        };

        template <
          bool triviallyMoveConstructible,
          bool nothrowMoveConstructible
        >
        constexpr void test_move_constructors() 
        {
            constexpr bool should_be_noexcept = triviallyMoveConstructible || nothrowMoveConstructible;

            using payload = payload_move_constructor<triviallyMoveConstructible,nothrowMoveConstructible>;

            { // Check payload type
                using Expected = expected<payload, int>;
                static_assert(std::is_trivially_move_constructible<Expected>::value == triviallyMoveConstructible, "bad");
                static_assert(std::is_move_constructible<Expected>::value, "bad");

                Expected value_input{in_place};
                const Expected from_value{std::move(value_input)};
                DLIB_TEST(from_value);
                DLIB_TEST(from_value.value() == (triviallyMoveConstructible ? 0 : 42));
                static_assert(noexcept(Expected{std::move(value_input)}) == should_be_noexcept, "bad");

                Expected error_input{unexpect};
                const Expected from_error{std::move(error_input)};
                DLIB_TEST(!from_error);
                DLIB_TEST(from_error.error() == 0);
                static_assert(noexcept(Expected{std::move(error_input)}) == should_be_noexcept, "bad");
            }

            { // Check error type
                using Expected = expected<int, payload>;
                static_assert(std::is_trivially_move_constructible<Expected>::value == triviallyMoveConstructible, "bad");
                static_assert(std::is_move_constructible<Expected>::value, "bad");

                Expected value_input{in_place};
                const Expected from_value{std::move(value_input)};
                DLIB_TEST(from_value);
                DLIB_TEST(from_value.value() == 0);
                static_assert(noexcept(Expected{std::move(value_input)}) == should_be_noexcept, "bad");

                Expected error_input{unexpect};
                const Expected from_error{std::move(error_input)};
                DLIB_TEST(!from_error);
                DLIB_TEST(from_error.error() == (triviallyMoveConstructible ? 0 : 42));
                static_assert(noexcept(Expected{std::move(error_input)}) == should_be_noexcept, "bad");
            }

            { // Check void payload
                using Expected = expected<void, payload>;
                static_assert(std::is_trivially_move_constructible<Expected>::value == triviallyMoveConstructible, "bad");
                static_assert(std::is_move_constructible<Expected>::value, "bad");

                Expected value_input{in_place};
                const Expected from_value{std::move(value_input)};
                DLIB_TEST(from_value);
                // static_assert(noexcept(Expected{std::move(value_input)}) == should_be_noexcept, "bad");

                Expected error_input{unexpect};
                const Expected from_error{std::move(error_input)};
                DLIB_TEST(!from_error);
                DLIB_TEST(from_error.error() == (triviallyMoveConstructible ? 0 : 42));
                // static_assert(noexcept(Expected{std::move(error_input)}) == should_be_noexcept, "bad");
            }

            { // ensure we are not move constructible if either the payload or the error are not
                struct not_move_constructible {
                    not_move_constructible(not_move_constructible&&) = delete;
                };

                static_assert(!std::is_move_constructible<expected<not_move_constructible, int>>::value, "bad");
                static_assert(!std::is_move_constructible<expected<int, not_move_constructible>>::value, "bad");
                static_assert(!std::is_move_constructible<expected<void, not_move_constructible>>::value, "bad");
            }
        }

        template <bool triviallyDestructible>
        struct payload_destructor 
        {
            constexpr payload_destructor(bool& destructor_called) : _destructor_called(destructor_called) {}
            ~payload_destructor() = default;
            bool& _destructor_called;
        };

        template <>
        struct payload_destructor<false>
        {
            constexpr payload_destructor(bool& destructor_called) : _destructor_called(destructor_called) {}
            ~payload_destructor() { _destructor_called = true; }
            bool& _destructor_called;
        };

        template <bool triviallyDestructible>
        constexpr void test_destructors()
        {
            constexpr bool is_trivial = triviallyDestructible;
            bool destructor_called    = false;

            { // Check payload
                using Expected = expected<payload_destructor<triviallyDestructible>, int>;
                static_assert(std::is_trivially_destructible<Expected>::value == is_trivial, "bad");
                Expected val{in_place, destructor_called};
            }

            DLIB_TEST(destructor_called == !is_trivial);
            destructor_called = false;

            { // Check error
                using Expected = expected<int, payload_destructor<triviallyDestructible>>;
                static_assert(std::is_trivially_destructible<Expected>::value == is_trivial, "bad");
                Expected err{unexpect, destructor_called};
            }

            DLIB_TEST(destructor_called == !is_trivial);
            destructor_called = false;

            { // Check void error
                using Expected = expected<void, payload_destructor<triviallyDestructible>>;
                static_assert(std::is_trivially_destructible<Expected>::value == is_trivial, "bad");
                Expected err{unexpect, destructor_called};
            }
            DLIB_TEST(destructor_called == !is_trivial);
        }

        void test_special_members() 
        {
            test_default_constructors();

            test_copy_constructors<false, false>();
            test_copy_constructors<false, true>();
            test_copy_constructors<true, false>();
            test_copy_constructors<true, true>();

            test_move_constructors<false, false>();
            test_move_constructors<false, true>();
            test_move_constructors<true, false>();
            test_move_constructors<true, true>();

            test_destructors<false>();
            test_destructors<true>();
        }

        using test_unexpected::convertible;

        template <
          bool nothrowConstructible, 
          bool explicitConstructible
        >
        struct payload_constructors 
        {
            payload_constructors() = default;
            // Note clang does not accept local variables in explicit
            constexpr payload_constructors(const convertible&) noexcept(nothrowConstructible) : _val(3) {}
            constexpr payload_constructors(convertible&&) noexcept(nothrowConstructible) : _val(42) {}
            constexpr payload_constructors(std::initializer_list<int>&, convertible) noexcept(nothrowConstructible) : _val(1337) {}
            constexpr bool operator==(const int val) const noexcept { return _val == val; }
            int _val = 0;
        };

        template <
          bool nothrowConstructible
        >
        struct payload_constructors<nothrowConstructible,true>
        {
            payload_constructors() = default;
            // Note clang does not accept local variables in explicit
            constexpr explicit payload_constructors(const convertible&) noexcept(nothrowConstructible) : _val(3) {}
            constexpr explicit payload_constructors(convertible&&) noexcept(nothrowConstructible) : _val(42) {}
            constexpr explicit payload_constructors(std::initializer_list<int>&, convertible) noexcept(nothrowConstructible) : _val(1337) {}
            constexpr bool operator==(const int val) const noexcept { return _val == val; }
            int _val = 0;
        };

        template <
          bool nothrowConstructible, 
          bool explicitConstructible
        >
        void test_constructors() 
        {
            constexpr bool should_be_noexcept = nothrowConstructible;
            constexpr bool should_be_explicit = explicitConstructible;

            using payload = payload_constructors<nothrowConstructible,explicitConstructible>;

            { // constructing from convertible payload
                using Input    = convertible;
                using Expected = expected<payload, payload>;
                static_assert(std::is_convertible<const Input&, Expected>::value != should_be_explicit, "bad");
                static_assert(std::is_convertible<Input, Expected>::value != should_be_explicit, "bad");

                const Input const_input_value{};
                const Expected copy_constructed_value{const_input_value};
                DLIB_TEST(copy_constructed_value);
                DLIB_TEST(copy_constructed_value.value() == 3);
                // static_assert(noexcept(Expected{const_input_value}) == should_be_noexcept, "bad");

                const Expected move_constructed_value{Input{}};
                DLIB_TEST(move_constructed_value);
                DLIB_TEST(move_constructed_value.value() == 42);
                // static_assert(noexcept(Expected{Input{}}) == should_be_noexcept, "bad");

                // TODO: FIX
                // const Expected brace_constructed_value{{}};
                // DLIB_TEST(brace_constructed_value);
                // DLIB_TEST(brace_constructed_value.value() == 0);
                // static_assert(noexcept(Expected{{}}), "bad");
            }

            { // converting from different expected
                using Input    = expected<convertible, convertible>;
                using Expected = expected<payload, payload>;
                static_assert(std::is_convertible<const Input&, Expected>::value != should_be_explicit, "bad");
                static_assert(std::is_convertible<Input, Expected>::value != should_be_explicit, "bad");

                const Input const_input_value{};
                const Expected copy_constructed_value{const_input_value};
                DLIB_TEST(copy_constructed_value);
                DLIB_TEST(copy_constructed_value.value() == 3);
                // TODO: FIX
                // static_assert(noexcept(Expected{const_input_value}) == should_be_noexcept, "bad");

                const Expected move_constructed_value{Input{in_place}};
                DLIB_TEST(move_constructed_value);
                DLIB_TEST(move_constructed_value.value() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{Input{in_place}}) == should_be_noexcept, "bad");

                const Input const_input_error{unexpect};
                const Expected copy_constructed_error{const_input_error};
                DLIB_TEST(!copy_constructed_error);
                DLIB_TEST(copy_constructed_error.error() == 3);
                // TODO: FIX
                // static_assert(noexcept(Expected{const_input_error}) == should_be_noexcept, "bad");

                const Expected move_constructed_error{Input{unexpect}};
                DLIB_TEST(!move_constructed_error);
                DLIB_TEST(move_constructed_error.error() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{Input{unexpect}}) == should_be_noexcept, "bad");
            }

            { // converting from unexpected
                using Input    = dlib::unexpected<convertible>;
                using Expected = dlib::expected<int, payload>;

                const Input const_input{in_place};
                const Expected copy_constructed{const_input};
                DLIB_TEST(!copy_constructed);
                DLIB_TEST(copy_constructed.error() == 3);
                // TODO: FIX
                // static_assert(noexcept(Expected{const_input}) == should_be_noexcept, "bad");

                const Expected move_constructed{Input{in_place}};
                DLIB_TEST(!move_constructed);
                DLIB_TEST(move_constructed.error() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{Input{in_place}}) == should_be_noexcept, "bad");
            }

            { // in place payload
                using Expected = expected<payload, int>;
                const Expected default_constructed{in_place};
                DLIB_TEST(default_constructed);
                DLIB_TEST(default_constructed.value() == 0);
                static_assert(noexcept(Expected{in_place}), "bad");

                const Expected value_constructed{in_place, convertible{}};
                DLIB_TEST(value_constructed);
                DLIB_TEST(value_constructed.value() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{in_place, convertible{}}) == should_be_noexcept, "bad");

                const Expected ilist_value_constructed{in_place, {1}, convertible{}};
                DLIB_TEST(ilist_value_constructed);
                DLIB_TEST(ilist_value_constructed.value() == 1337);
                // TODO: FIX
                // static_assert(noexcept(Expected{in_place, {1}, convertible{}}) == should_be_noexcept, "bad");
            }

            { // in place error
                using Expected = expected<int, payload>;
                const Expected default_constructed{unexpect};
                DLIB_TEST(!default_constructed);
                DLIB_TEST(default_constructed.error() == 0);
                static_assert(noexcept(Expected{unexpect}), "bad");

                const Expected value_constructed{unexpect, convertible{}};
                DLIB_TEST(!value_constructed);
                DLIB_TEST(value_constructed.error() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{unexpect, convertible{}}) == should_be_noexcept, "bad");

                const Expected ilist_value_constructed{unexpect, {1}, convertible{}};
                DLIB_TEST(!ilist_value_constructed);
                DLIB_TEST(ilist_value_constructed.error() == 1337);
                // TODO: FIX
                // static_assert(noexcept(Expected{unexpect, {1}, convertible{}}) == should_be_noexcept, "bad");
            }

            { // expected<void, E>: converting from different expected
                using Input    = expected<void, convertible>;
                using Expected = expected<void, payload>;
                static_assert(std::is_convertible<const Input&, Expected>::value != should_be_explicit, "bad");
                static_assert(std::is_convertible<Input, Expected>::value != should_be_explicit, "bad");

                const Input const_input_value{};
                const Expected copy_constructed_value{const_input_value};
                DLIB_TEST(copy_constructed_value);
                copy_constructed_value.value();
                // TODO: FIX
                // static_assert(noexcept(Expected{const_input_value}) == should_be_noexcept, "bad");

                const Expected move_constructed_value{Input{in_place}};
                DLIB_TEST(move_constructed_value);
                move_constructed_value.value();
                // TODO: FIX
                // static_assert(noexcept(Expected{Input{in_place}}) == should_be_noexcept, "bad");

                const Input const_input_error{unexpect};
                const Expected copy_constructed_error{const_input_error};
                DLIB_TEST(!copy_constructed_error);
                DLIB_TEST(copy_constructed_error.error() == 3);
                // TODO: FIX
                // static_assert(noexcept(Expected{const_input_error}) == should_be_noexcept, "bad");

                const Expected move_constructed_error{Input{unexpect}};
                DLIB_TEST(!move_constructed_error);
                DLIB_TEST(move_constructed_error.error() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{Input{unexpect}}) == should_be_noexcept, "bad");
            }

            { // expected<void, E>: converting from unexpected
                using Input    = dlib::unexpected<convertible>;
                using Expected = dlib::expected<void, payload>;

                const Input const_input{in_place};
                const Expected copy_constructed{const_input};
                DLIB_TEST(!copy_constructed);
                DLIB_TEST(copy_constructed.error() == 3);
                // TODO: FIX
                // static_assert(noexcept(Expected{const_input}) == should_be_noexcept, "bad");

                const Expected move_constructed{Input{in_place}};
                DLIB_TEST(!move_constructed);
                DLIB_TEST(move_constructed.error() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{Input{in_place}}) == should_be_noexcept, "bad");
            }

            { // expected<void, E>: in place payload
                using Expected = expected<void, int>;
                const Expected default_constructed{in_place};
                DLIB_TEST(default_constructed);
                default_constructed.value();
                static_assert(noexcept(Expected{in_place}), "bad");
            }

            { // expected<void, E>: in place error
                using Expected = expected<void, payload>;
                const Expected default_constructed{unexpect};
                DLIB_TEST(!default_constructed);
                DLIB_TEST(default_constructed.error() == 0);
                static_assert(noexcept(Expected{unexpect}), "bad");

                const Expected value_constructed{unexpect, convertible{}};
                DLIB_TEST(!value_constructed);
                DLIB_TEST(value_constructed.error() == 42);
                // TODO: FIX
                // static_assert(noexcept(Expected{unexpect, convertible{}}) == should_be_noexcept, "bad");

                const Expected ilist_value_constructed{unexpect, {1}, convertible{}};
                DLIB_TEST(!ilist_value_constructed);
                DLIB_TEST(ilist_value_constructed.error() == 1337);
                // TODO: FIX
                // static_assert(noexcept(Expected{unexpect, {1}, convertible{}}) == should_be_noexcept, "bad");
            }
        }

        void test_constructors_all() 
        {
            test_constructors<false, false>();
            test_constructors<false, true>();
            test_constructors<true, false>();
            test_constructors<true, true>();

            // LWG-3836
            struct BaseError {};
            struct DerivedError : BaseError {};

            dlib::expected<bool, DerivedError> e1(false);
            dlib::expected<bool, BaseError> e2(e1);
            // TODO: FIX
            // DLIB_TEST(!e2.value());
        }
    }


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
            test_monads();
            test_semantics();
            test_strong_exception_guarantees();

            //From MSVC's unit test https://github.com/microsoft/STL/blob/main/tests/std/tests/P0323R12_expected/test.cpp
            test_unexpected::test_all();
            test_expected::test_aliases();
            test_expected::test_special_members();
            test_expected::test_constructors_all();
            // test_expected::test_assignment();
            // test_expected::test_emplace();
            // test_expected::test_swap();
            // test_expected::test_access();
            // test_expected::test_monadic();
            // test_expected::test_equality();
        }
    } a;
}