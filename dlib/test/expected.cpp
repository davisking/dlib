// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/expected.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.expected");

// ---------------------------------------------------------------------------------------------------

    static_assert(std::is_copy_constructible<dlib::expected<int, int>>::value, "bad");
    static_assert(std::is_copy_assignable<dlib::expected<int, int>>::value,    "bad");
    static_assert(std::is_move_constructible<dlib::expected<int, int>>::value, "bad");
    static_assert(std::is_move_assignable<dlib::expected<int, int>>::value,    "bad");
    static_assert(std::is_destructible<dlib::expected<int, int>>::value,       "bad");

    static_assert(std::is_trivially_copy_constructible<dlib::expected<int, int>>::value, "bad");
    static_assert(std::is_trivially_copy_assignable<dlib::expected<int, int>>::value,    "bad");
    static_assert(std::is_trivially_move_constructible<dlib::expected<int, int>>::value, "bad");
    static_assert(std::is_trivially_move_assignable<dlib::expected<int, int>>::value,    "bad");
    static_assert(std::is_trivially_destructible<dlib::expected<int, int>>::value,       "bad");


    static_assert(std::is_nothrow_copy_constructible<dlib::expected<int, int>>::value, "bad");
    static_assert(std::is_nothrow_copy_assignable<dlib::expected<int, int>>::value,    "bad");
    static_assert(std::is_nothrow_move_constructible<dlib::expected<int, int>>::value, "bad");
    static_assert(std::is_nothrow_move_assignable<dlib::expected<int, int>>::value,    "bad");
    static_assert(std::is_nothrow_destructible<dlib::expected<int, int>>::value,       "bad");


    struct trivial_type 
    {
        trivial_type(const trivial_type&)             = default;
        trivial_type(trivial_type&&)                  = default;
        trivial_type& operator=(const trivial_type&)  = default;
        trivial_type& operator=(trivial_type&&)       = default;
        ~trivial_type() = default;
    };

    static_assert(std::is_copy_constructible<dlib::expected<trivial_type>>::value, "bad");
    static_assert(std::is_copy_assignable<dlib::expected<trivial_type>>::value,    "bad");
    static_assert(std::is_move_constructible<dlib::expected<trivial_type>>::value, "bad");
    static_assert(std::is_move_assignable<dlib::expected<trivial_type>>::value,    "bad");
    static_assert(std::is_destructible<dlib::expected<trivial_type>>::value,       "bad");

    static_assert(std::is_trivially_copy_constructible<dlib::expected<trivial_type>>::value, "bad");
    static_assert(std::is_trivially_copy_assignable<dlib::expected<trivial_type>>::value,    "bad");
    static_assert(std::is_trivially_move_constructible<dlib::expected<trivial_type>>::value, "bad");
    static_assert(std::is_trivially_move_assignable<dlib::expected<trivial_type>>::value,    "bad");
    static_assert(std::is_trivially_destructible<dlib::expected<trivial_type>>::value,       "bad");

    static_assert(std::is_nothrow_copy_constructible<dlib::expected<trivial_type>>::value, "bad");
    static_assert(std::is_nothrow_copy_assignable<dlib::expected<trivial_type>>::value,    "bad");
    static_assert(std::is_nothrow_move_constructible<dlib::expected<trivial_type>>::value, "bad");
    static_assert(std::is_nothrow_move_assignable<dlib::expected<trivial_type>>::value,    "bad");
    static_assert(std::is_nothrow_destructible<dlib::expected<trivial_type>>::value,       "bad");

    struct non_trivial_type
    {
        non_trivial_type(const non_trivial_type&)               {}
        non_trivial_type(non_trivial_type&&)                    {};
        non_trivial_type& operator=(const non_trivial_type&)    { return *this; }
        non_trivial_type& operator=(non_trivial_type&&)         { return *this; };
        ~non_trivial_type()                                     {}
    };

    static_assert(std::is_copy_constructible<dlib::expected<non_trivial_type>>::value, "bad");
    static_assert(std::is_copy_assignable<dlib::expected<non_trivial_type>>::value,    "bad");
    static_assert(std::is_move_constructible<dlib::expected<non_trivial_type>>::value, "bad");
    static_assert(std::is_move_assignable<dlib::expected<non_trivial_type>>::value,    "bad");
    static_assert(std::is_destructible<dlib::expected<non_trivial_type>>::value,       "bad");

    static_assert(!std::is_trivially_copy_constructible<dlib::expected<non_trivial_type>>::value, "bad");
    static_assert(!std::is_trivially_copy_assignable<dlib::expected<non_trivial_type>>::value,    "bad");
    static_assert(!std::is_trivially_move_constructible<dlib::expected<non_trivial_type>>::value, "bad");
    static_assert(!std::is_trivially_move_assignable<dlib::expected<non_trivial_type>>::value,    "bad");
    static_assert(!std::is_trivially_destructible<dlib::expected<non_trivial_type>>::value,       "bad");

    static_assert(!std::is_nothrow_copy_constructible<dlib::expected<non_trivial_type>>::value, "bad");
    static_assert(!std::is_nothrow_copy_assignable<dlib::expected<non_trivial_type>>::value,    "bad");
    static_assert(!std::is_nothrow_move_constructible<dlib::expected<non_trivial_type>>::value, "bad");
    static_assert(!std::is_nothrow_move_assignable<dlib::expected<non_trivial_type>>::value,    "bad");

    struct nothing_works
    {
        nothing_works(const nothing_works&)             = delete;
        nothing_works(nothing_works&&)                  = delete;
        nothing_works& operator=(const nothing_works&)  = delete;
        nothing_works& operator=(nothing_works&&)       = delete;
    };

    static_assert(!std::is_copy_constructible<dlib::expected<nothing_works>>::value,    "bad");
    static_assert(!std::is_copy_assignable<dlib::expected<nothing_works>>::value,       "bad");
    static_assert(!std::is_move_constructible<dlib::expected<nothing_works>>::value,    "bad");
    static_assert(!std::is_move_assignable<dlib::expected<nothing_works>>::value,       "bad");

    struct copyable_type 
    {
        copyable_type(const copyable_type&)             = default;
        copyable_type(copyable_type&&)                  = delete;
        copyable_type& operator=(const copyable_type&)  = default;
        copyable_type& operator=(copyable_type&&)       = delete;
    };

    static_assert(std::is_copy_constructible<dlib::expected<copyable_type>>::value,     "bad");
    static_assert(std::is_copy_assignable<dlib::expected<copyable_type>>::value,        "bad");
    //copyable_type can still be moved, but it will just copy.

    struct moveable_type 
    {
        moveable_type(const moveable_type&)             = delete;
        moveable_type(moveable_type&&)                  = default;
        moveable_type& operator=(const moveable_type&)  = delete;
        moveable_type& operator=(moveable_type&&)       = default;
    };

    static_assert(!std::is_copy_constructible<dlib::expected<moveable_type>>::value,    "bad");
    static_assert(!std::is_copy_assignable<dlib::expected<moveable_type>>::value,       "bad");
    static_assert(std::is_move_constructible<dlib::expected<moveable_type>>::value,     "bad");
    static_assert(std::is_move_assignable<dlib::expected<moveable_type>>::value,        "bad");

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
        }
        
    } a;
}