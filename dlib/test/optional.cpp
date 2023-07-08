// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/optional.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.optional");

// ---------------------------------------------------------------------------------------------------

    static_assert(std::is_copy_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_copy_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_move_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_move_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_destructible<dlib::optional<int>>::value,       "bad");

    static_assert(std::is_trivially_copy_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_trivially_copy_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_trivially_move_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_trivially_move_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_trivially_destructible<dlib::optional<int>>::value,       "bad");

    static_assert(std::is_nothrow_copy_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_nothrow_copy_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_nothrow_move_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_nothrow_move_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_nothrow_destructible<dlib::optional<int>>::value,       "bad");

    struct trivial_type 
    {
        trivial_type(const trivial_type&)             = default;
        trivial_type(trivial_type&&)                  = default;
        trivial_type& operator=(const trivial_type&)  = default;
        trivial_type& operator=(trivial_type&&)       = default;
        ~trivial_type() = default;
    };

    static_assert(std::is_copy_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_copy_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_move_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_move_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_destructible<dlib::optional<trivial_type>>::value,       "bad");

    static_assert(std::is_trivially_copy_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_trivially_copy_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_trivially_move_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_trivially_move_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_trivially_destructible<dlib::optional<trivial_type>>::value,       "bad");

    static_assert(std::is_nothrow_copy_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_nothrow_copy_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_nothrow_move_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_nothrow_move_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_nothrow_destructible<dlib::optional<trivial_type>>::value,       "bad");

    struct non_trivial_type
    {
        non_trivial_type(const non_trivial_type&)               {}
        non_trivial_type(non_trivial_type&&)                    {};
        non_trivial_type& operator=(const non_trivial_type&)    { return *this; }
        non_trivial_type& operator=(non_trivial_type&&)         { return *this; };
        ~non_trivial_type()                                     {}
    };

    static_assert(std::is_copy_constructible<dlib::optional<non_trivial_type>>::value, "bad");
    static_assert(std::is_copy_assignable<dlib::optional<non_trivial_type>>::value,    "bad");
    static_assert(std::is_move_constructible<dlib::optional<non_trivial_type>>::value, "bad");
    static_assert(std::is_move_assignable<dlib::optional<non_trivial_type>>::value,    "bad");
    static_assert(std::is_destructible<dlib::optional<non_trivial_type>>::value,       "bad");

    static_assert(!std::is_trivially_copy_constructible<dlib::optional<non_trivial_type>>::value, "bad");
    static_assert(!std::is_trivially_copy_assignable<dlib::optional<non_trivial_type>>::value,    "bad");
    static_assert(!std::is_trivially_move_constructible<dlib::optional<non_trivial_type>>::value, "bad");
    static_assert(!std::is_trivially_move_assignable<dlib::optional<non_trivial_type>>::value,    "bad");
    static_assert(!std::is_trivially_destructible<dlib::optional<non_trivial_type>>::value,       "bad");

    static_assert(!std::is_nothrow_copy_constructible<dlib::optional<non_trivial_type>>::value, "bad");
    static_assert(!std::is_nothrow_copy_assignable<dlib::optional<non_trivial_type>>::value,    "bad");
    static_assert(!std::is_nothrow_move_constructible<dlib::optional<non_trivial_type>>::value, "bad");
    static_assert(!std::is_nothrow_move_assignable<dlib::optional<non_trivial_type>>::value,    "bad");

    struct nothing_works
    {
        nothing_works(const nothing_works&)             = delete;
        nothing_works(nothing_works&&)                  = delete;
        nothing_works& operator=(const nothing_works&)  = delete;
        nothing_works& operator=(nothing_works&&)       = delete;
    };

    static_assert(!std::is_copy_constructible<dlib::optional<nothing_works>>::value,    "bad");
    static_assert(!std::is_copy_assignable<dlib::optional<nothing_works>>::value,       "bad");
    static_assert(!std::is_move_constructible<dlib::optional<nothing_works>>::value,    "bad");
    static_assert(!std::is_move_assignable<dlib::optional<nothing_works>>::value,       "bad");

    struct copyable_type 
    {
        copyable_type(const copyable_type&)             = default;
        copyable_type(copyable_type&&)                  = delete;
        copyable_type& operator=(const copyable_type&)  = default;
        copyable_type& operator=(copyable_type&&)       = delete;
    };

    static_assert(std::is_copy_constructible<dlib::optional<copyable_type>>::value,     "bad");
    static_assert(std::is_copy_assignable<dlib::optional<copyable_type>>::value,        "bad");
    //copyable_type can still be moved, but it will just copy.

    struct moveable_type 
    {
        moveable_type(const moveable_type&)             = delete;
        moveable_type(moveable_type&&)                  = default;
        moveable_type& operator=(const moveable_type&)  = delete;
        moveable_type& operator=(moveable_type&&)       = default;
    };

    static_assert(!std::is_copy_constructible<dlib::optional<moveable_type>>::value,    "bad");
    static_assert(!std::is_copy_assignable<dlib::optional<moveable_type>>::value,       "bad");
    static_assert(std::is_move_constructible<dlib::optional<moveable_type>>::value,     "bad");
    static_assert(std::is_move_assignable<dlib::optional<moveable_type>>::value,        "bad");

// ---------------------------------------------------------------------------------------------------

    void test_optional_int()
    {
        dlib::optional<int> o1;
        DLIB_TEST(!o1);
        DLIB_TEST(!o1.has_value());
        DLIB_TEST(o1.value_or(2) == 2);
        int throw_counter{0};
        try  {
            o1.value();
        } catch(const std::exception& e) {
            throw_counter++;
        }
        DLIB_TEST(throw_counter == 1);
        DLIB_TEST(o1 == dlib::nullopt);

        dlib::optional<int> o2 = dlib::nullopt;
        static_assert(noexcept(o2 = dlib::nullopt), "bad");
        DLIB_TEST(!o2);
        DLIB_TEST(!o2.has_value());
        DLIB_TEST(o2.value_or(3) == 3);

        dlib::optional<int> o3 = 42;
        DLIB_TEST(*o3 == 42);
        DLIB_TEST(o3.value() == 42);
        DLIB_TEST(o3.value_or(3) == 42);
        DLIB_TEST(o3 == 42);
        DLIB_TEST(42 == o3);

        dlib::optional<int> o4 = o3;
        static_assert(noexcept(o4 = o3), "bad");
        DLIB_TEST(*o3 == 42);
        DLIB_TEST(*o4 == 42);
        DLIB_TEST(o4 == 42);
        DLIB_TEST(42 == o4);
        DLIB_TEST(o4.value() == 42);

        dlib::optional<int> o5 = o1;
        static_assert(noexcept(o5 = o1), "bad");
        DLIB_TEST(!o1);
        DLIB_TEST(!o5);
        DLIB_TEST(!o5.has_value());

        dlib::optional<int> o6 = std::move(o3);
        static_assert(noexcept(o6 = std::move(o3)), "bad");
        DLIB_TEST(*o6 == 42);
        DLIB_TEST(o6.value() == 42);

        dlib::optional<short> o7 = (short)42;
        static_assert(noexcept(o7 = (short)42), "bad");
        DLIB_TEST(*o7 == 42);

        dlib::optional<int> o8 = o7;
        DLIB_TEST(*o7 == 42);
        DLIB_TEST(*o8 == 42);

        dlib::optional<int> o9 = std::move(o7);
        DLIB_TEST(*o9 == 42);

        dlib::optional<int> o10;
        DLIB_TEST(!o10);
        o10 = o3;
        DLIB_TEST(*o10 == 42);
        o10 = dlib::nullopt;
        DLIB_TEST(!o10);

        dlib::optional<int> o11;
        o11 = std::move(o3);
        DLIB_TEST(*o11 == 42);
        o11 = (short)12;
        DLIB_TEST(*o11 == 12);

        dlib::optional<int> o12;
        swap(o12, o4);
        DLIB_TEST(o12);
        DLIB_TEST(!o4);
        DLIB_TEST(*o12 == 42);
        static_assert(noexcept(swap(o12, o4)), "bad");

        o4.reset();
        DLIB_TEST(!o4);
    }

// ---------------------------------------------------------------------------------------------------

    void test_optional_int_constexpr()
    {
        {
            constexpr dlib::optional<int> o2{};
            constexpr dlib::optional<int> o3 = {};
            constexpr dlib::optional<int> o4 = dlib::nullopt;
            constexpr dlib::optional<int> o5 = {dlib::nullopt};
            constexpr dlib::optional<int> o6(dlib::nullopt);

            static_assert(!o2, "bad");
            static_assert(!o3, "bad");
            static_assert(!o4, "bad");
            static_assert(!o5, "bad");
            static_assert(!o6, "bad");

            static_assert(o2.value_or(1) == 1, "bad");
            static_assert(o3.value_or(1) == 1, "bad");
            static_assert(o4.value_or(1) == 1, "bad");
            static_assert(o5.value_or(1) == 1, "bad");
            static_assert(o6.value_or(1) == 1, "bad");

            static_assert(o2 != 1, "bad");
            static_assert(1 != o2, "bad");
        }
    }

// ---------------------------------------------------------------------------------------------------

    void test_optional_int_monads()
    {
        dlib::optional<int> o1{42};

        {
            auto res = o1.and_then([](int i) { return dlib::optional<long>(i); });

            static_assert(std::is_same<decltype(res), dlib::optional<long>>::value, "bad map");
            DLIB_TEST(*res == 42);
        }

        {
            auto res = o1.transform([](int i) { return (long)i; });

            static_assert(std::is_same<decltype(res), dlib::optional<long>>::value, "bad map");
            DLIB_TEST(*res == 42);
        }

        {
            auto res = o1.and_then([](int i) { return dlib::optional<std::string>(std::to_string(i)); });

            static_assert(std::is_same<decltype(res), dlib::optional<std::string>>::value, "bad map");
            DLIB_TEST(*res == "42");
        }

        {
            auto res = o1.transform([](int i) { return std::to_string(i); });

            static_assert(std::is_same<decltype(res), dlib::optional<std::string>>::value, "bad map");
            DLIB_TEST(*res == "42");
        }

        {
            auto res = o1.transform([](int i)                   {return i+1;})
                         .transform([](int i)                   {return i*2;})
                         .transform([](int i)                   {return std::to_string(i);})
                         .transform([](const std::string& str)  {return std::stoi(str);})
                         .transform([](int i)                   {return i - 2;})
                         .or_else([]                            {return dlib::make_optional<int>(0);});

            static_assert(std::is_same<decltype(res), dlib::optional<int>>::value, "bad map");
            DLIB_TEST(*res == 84);
        }

        dlib::optional<int> o2;

        {
            auto res = o2.transform([](int i)   {return i+1;})
                         .or_else([]            {return dlib::make_optional<int>(0);});

            static_assert(std::is_same<decltype(res), dlib::optional<int>>::value, "bad map");
            DLIB_TEST(*res == 0);
        }
    }

// ---------------------------------------------------------------------------------------------------

    void test_optional_int_constexpr_monads()
    {
        constexpr dlib::optional<int> o1{42};

        {
            struct callback
            {
                constexpr auto operator()(int i) {return dlib::optional<long>{i}; };
            };

            constexpr auto res = o1.and_then(callback{});

            static_assert(std::is_same<std::decay_t<decltype(res)>, dlib::optional<long>>::value, "bad map");
            static_assert(*res == 42, "bad");
            DLIB_TEST(*res == 42);
        }
    }

// ---------------------------------------------------------------------------------------------------

    static int constructor_count{0};
    static int copy_constructor_count{0};
    static int move_constructor_count{0};
    static int copy_assign_count{0};
    static int move_assign_count{0};

    static void reset_counters()
    {
        constructor_count           = 0;
        copy_constructor_count      = 0;
        move_constructor_count      = 0;
        copy_assign_count           = 0;
        move_assign_count           = 0;
    }

    struct optional_dummy
    {
        optional_dummy() {++constructor_count;}
        optional_dummy(const optional_dummy&) {++copy_constructor_count;}
        optional_dummy(optional_dummy&&) {++move_constructor_count;}
        optional_dummy& operator=(const optional_dummy&) {++copy_assign_count; return *this;}
        optional_dummy& operator=(optional_dummy&&) {++move_assign_count; return *this;}
    };

    void test_constructors()
    {
        dlib::optional<optional_dummy> val;
        DLIB_TEST(constructor_count == 0);
        DLIB_TEST(copy_constructor_count == 0);
        DLIB_TEST(move_constructor_count == 0);
        DLIB_TEST(copy_assign_count == 0);
        DLIB_TEST(move_assign_count == 0);

        val = optional_dummy{};
        DLIB_TEST(constructor_count == 1);
        DLIB_TEST(copy_constructor_count == 0);
        DLIB_TEST(move_constructor_count == 1);
        DLIB_TEST(copy_assign_count == 0);
        DLIB_TEST(move_assign_count == 0);
        reset_counters();

        dlib::optional<optional_dummy> val2{val};
        DLIB_TEST(constructor_count == 0);
        DLIB_TEST(copy_constructor_count == 1);
        DLIB_TEST(move_constructor_count == 0);
        DLIB_TEST(copy_assign_count == 0);
        DLIB_TEST(move_assign_count == 0);
        reset_counters();

        dlib::optional<optional_dummy> val3{std::move(val)};
        DLIB_TEST(constructor_count == 0);
        DLIB_TEST(copy_constructor_count == 0);
        DLIB_TEST(move_constructor_count == 1);
        DLIB_TEST(copy_assign_count == 0);
        DLIB_TEST(move_assign_count == 0);
        reset_counters();

        val2 = val;
        DLIB_TEST(constructor_count == 0);
        DLIB_TEST(copy_constructor_count == 0);
        DLIB_TEST(move_constructor_count == 0);
        DLIB_TEST(copy_assign_count == 1);
        DLIB_TEST(move_assign_count == 0);
        reset_counters();

        val3 = std::move(val);
        DLIB_TEST(constructor_count == 0);
        DLIB_TEST(copy_constructor_count == 0);
        DLIB_TEST(move_constructor_count == 0);
        DLIB_TEST(copy_assign_count == 0);
        DLIB_TEST(move_assign_count == 1);
        reset_counters();
    }

// ---------------------------------------------------------------------------------------------------

    void test_emplace()
    {
        struct A
        {
            A() = default;
            explicit A(int i, float f, const std::string& str) : i_{i}, f_{f}, str_{str} {}

            int         i_{0};
            float       f_{0.0f};
            std::string str_;
        };

        dlib::optional<A> o1(dlib::in_place, 1, 2.5f, "hello there");
        DLIB_TEST(o1);
        DLIB_TEST(o1->i_ == 1);
        DLIB_TEST(o1->f_ == 2.5f);
        DLIB_TEST(o1->str_ == "hello there");

        dlib::optional<A> o2;
        o2.emplace(2, 5.1f, "general kenobi");
        DLIB_TEST(o2);
        DLIB_TEST(o2->i_ == 2);
        DLIB_TEST(o2->f_ == 5.1f);
        DLIB_TEST(o2->str_ == "general kenobi");

        auto o3 = dlib::make_optional<A>(3, 3.141592f, "from a certain point of view");
        DLIB_TEST(o3);
        DLIB_TEST(o3->i_ == 3);
        DLIB_TEST(o3->f_ == 3.141592f);
        DLIB_TEST(o3->str_ == "from a certain point of view");

        dlib::optional<std::string> o4(dlib::in_place, {'a', 'b', 'c'});
        DLIB_TEST(o4);
        DLIB_TEST(*o4 == "abc");
        DLIB_TEST(o4 == "abc");

        dlib::optional<std::string> o5;
        o5.emplace({'a', 'b', 'c'});
        DLIB_TEST(o5);
        DLIB_TEST(*o5 == "abc");
    }

// ---------------------------------------------------------------------------------------------------

    void test_make_optional()
    {
        constexpr auto o1 = dlib::make_optional(1);
        static_assert(std::is_same<std::decay_t<decltype(o1)>, dlib::optional<int>>::value, "bad");
        static_assert(o1 == 1, "bad");
        static_assert(1 == o1, "bad");
    }

    class optional_tester : public tester
    {
    public:
        optional_tester (
        ) :
            tester ("test_optional",
                    "Runs tests on the optional object")
        {}

        void perform_test (
        )
        {
            test_optional_int();
            test_optional_int_constexpr();
            test_optional_int_monads();
            test_optional_int_constexpr_monads();
            test_constructors();
            test_emplace();
            test_make_optional();
        }
    } a;
}