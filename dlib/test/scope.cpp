// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <cstdlib>
#include <cstring>
#include <dlib/scope.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.scope");

// ---------------------------------------------------------------------------------------------------

    void test_scope_exit()
    {
        int counter{0};

        {
            auto s1 = make_scope_exit([&]{++counter;});
            static_assert(!std::is_copy_constructible<decltype(s1)>::value, "bad");
            static_assert(!std::is_copy_assignable<decltype(s1)>::value, "bad");
            static_assert(!std::is_move_assignable<decltype(s1)>::value, "bad");
            static_assert(std::is_move_constructible<decltype(s1)>::value, "bad");
            auto s2 = std::move(s1);
            auto s3 = std::move(s2);
            auto s4 = std::move(s3);
        }

        DLIB_TEST(counter == 1);

        const auto fn_inner = [&]
        {
            auto s = make_scope_exit([&]{++counter;});
            return s;
        };

        const auto fn_outer = [&]
        {
            auto s = fn_inner();
            return s;
        };

        {
            auto s = fn_outer();
        }

        DLIB_TEST(counter == 2);

#ifdef __cpp_deduction_guides

        {
            scope_exit s{[&]{++counter;}};
        }

        DLIB_TEST(counter == 3);
#endif

    }

// ---------------------------------------------------------------------------------------------------


    void test_scope_exit_erased()
    {
        int counter{0};

        {
            scope_exit_erased s1([&]{++counter;});
            static_assert(!std::is_copy_constructible<decltype(s1)>::value, "bad");
            static_assert(!std::is_copy_assignable<decltype(s1)>::value, "bad");
            static_assert(!std::is_move_assignable<decltype(s1)>::value, "bad");
            static_assert(std::is_move_constructible<decltype(s1)>::value, "bad");
            auto s2 = std::move(s1);
            auto s3 = std::move(s2);
            auto s4 = std::move(s3);
        }

        DLIB_TEST(counter == 1);

        const auto fn_inner = [&]
        {
            scope_exit_erased s([&]{++counter;});
            return s;
        };

        const auto fn_outer = [&]
        {
            auto s = fn_inner();
            return s;
        };

        {
            auto s = fn_outer();
        }

        DLIB_TEST(counter == 2);
    }

    struct results_with_delayed_C_library_resource_management
    {
        int                 ndata{0};
        char*               data{nullptr};
        scope_exit_erased   s;
    };

    void test_composition()
    {
        int counter{0};

        const auto fn = [&]
        {
            // Pretend you're in a cpp file using a C library which isn't exposed to the API via the header.
            // You want to return some results, but those results are only valid so long as something returned by the C library is still alive
            // You want to delay releasing any resources allocated by the C library until after you've returned your results and the caller is done using them.
            // You could return a std::unique_ptr<results> object with a custom deleter which deletes that resource but because all types in std::unique_ptr
            // must be complete types, you would have to pollute the header. You can use std::shared_ptr with a custom deleter, defined at runtime,
            // but this is less efficient.
            // You can use a scope_exit_erased object to wrap the resouce management function from the C library and delay the call further up the stack,
            // all behind a type erased callback.

            // pretend malloc() is a fancy function from some exotic C library.
            // pretend free() is another fancy function which you don't want users to have to manually call, and you want to delay calling it until after the results are used
            // pretend cstdlib is a fancy header you don't want to expose in your own header file.
            char* data = (char*)std::malloc(100);
            std::memset(data, 0, 100);
            std::snprintf(data, 100, "hello there!");
            scope_exit_erased s{[=, &counter] {free(data); ++counter;}};

            results_with_delayed_C_library_resource_management results{100, data, std::move(s)};

            return results;
        };

        {
            // Oh, look at me. I'm using these results, blissfully unaware that some super complicated function in a C library will get called when i'm done using results.
            const auto results = fn();
            DLIB_TEST(results.ndata == 100);
            DLIB_TEST(std::strcmp(results.data, "hello there!") == 0);
            DLIB_TEST(counter == 0);
        }

        DLIB_TEST(counter == 1);   
    }

// ---------------------------------------------------------------------------------------------------

    class scope_tester : public tester
    {
    public:
        scope_tester (
        ) :
            tester ("test_scope",
                    "Runs tests on the scope_exit and related objects")
        {}

        void perform_test (
        )
        {
            test_scope_exit();
            test_scope_exit_erased();
            test_composition();
        }
    } a;

// ---------------------------------------------------------------------------------------------------

}