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

    static_assert(std::is_trivially_copy_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_trivially_copy_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_trivially_move_constructible<dlib::optional<int>>::value, "bad");
    static_assert(std::is_trivially_move_assignable<dlib::optional<int>>::value,    "bad");
    static_assert(std::is_trivially_destructible<dlib::optional<int>>::value,       "bad");

    struct trivial_type 
    {
        trivial_type(const trivial_type&)             = default;
        trivial_type(trivial_type&&)                  = default;
        trivial_type& operator=(const trivial_type&)  = default;
        trivial_type& operator=(trivial_type&&)       = default;
        ~trivial_type() = default;
    };

    static_assert(std::is_trivially_copy_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_trivially_copy_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_trivially_move_constructible<dlib::optional<trivial_type>>::value, "bad");
    static_assert(std::is_trivially_move_assignable<dlib::optional<trivial_type>>::value,    "bad");
    static_assert(std::is_trivially_destructible<dlib::optional<trivial_type>>::value,       "bad");

// ---------------------------------------------------------------------------------------------------

    void test_optional_int()
    {
        dlib::optional<int> o1;
        DLIB_TEST(!o1);

        dlib::optional<int> o2 = dlib::nullopt;
        DLIB_TEST(!o2);

        dlib::optional<int> o3 = 42;
        DLIB_TEST(*o3 == 42);

        dlib::optional<int> o4 = o3;
        DLIB_TEST(*o3 == 42);
        DLIB_TEST(*o4 == 42);

        dlib::optional<int> o5 = o1;
        DLIB_TEST(!o1);
        DLIB_TEST(!o5);

        dlib::optional<int> o6 = std::move(o3);
        DLIB_TEST(*o6 == 42);

        dlib::optional<short> o7 = (short)42;
        DLIB_TEST(*o7 == 42);

        dlib::optional<int> o8 = o7;
        DLIB_TEST(*o7 == 42);
        DLIB_TEST(*o8 == 42);

        dlib::optional<int> o9 = std::move(o7);
        DLIB_TEST(*o9 == 42);
    }

// ---------------------------------------------------------------------------------------------------

    static int constructor_count{0};
    static int copy_constructor_count{0};
    static int move_constructor_count{0};
    static int copy_assign_count{0};
    static int move_assign_count{0};

    struct optional_dummy
    {
        optional_dummy() {++constructor_count;}
        ~optional_dummy() {--constructor_count;}
        optional_dummy(const optional_dummy&) {++constructor_count; ++copy_constructor_count;}
        optional_dummy(optional_dummy&&) {++constructor_count; ++move_constructor_count;}
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
    }

// ---------------------------------------------------------------------------------------------------

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
            test_constructors();
        }
    } a;
}