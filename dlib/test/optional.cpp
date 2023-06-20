// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/optional.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.optional");

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
            test_constructors();
        }
    } a;
}