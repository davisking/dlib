// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/algs.h>
#include <dlib/metaprogramming.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.metaprogramming");


    void metaprogramming_test (
    )
    /*!
        ensures
            - runs tests on template metaprogramming objects and functions for compliance with the specs 
    !*/
    {        

        print_spinner();

        DLIB_TEST(is_signed_type<signed char>::value == true);
        DLIB_TEST(is_signed_type<signed short>::value == true);
        DLIB_TEST(is_signed_type<signed int>::value == true);
        DLIB_TEST(is_signed_type<signed long>::value == true);
        DLIB_TEST(is_unsigned_type<signed char>::value == false);
        DLIB_TEST(is_unsigned_type<signed short>::value == false);
        DLIB_TEST(is_unsigned_type<signed int>::value == false);
        DLIB_TEST(is_unsigned_type<signed long>::value == false);

        DLIB_TEST(is_unsigned_type<unsigned char>::value == true);
        DLIB_TEST(is_unsigned_type<unsigned short>::value == true);
        DLIB_TEST(is_unsigned_type<unsigned int>::value == true);
        DLIB_TEST(is_unsigned_type<unsigned long>::value == true);
        DLIB_TEST(is_signed_type<unsigned char>::value == false);
        DLIB_TEST(is_signed_type<unsigned short>::value == false);
        DLIB_TEST(is_signed_type<unsigned int>::value == false);
        DLIB_TEST(is_signed_type<unsigned long>::value == false);


        COMPILE_TIME_ASSERT(is_signed_type<signed char>::value == true);
        COMPILE_TIME_ASSERT(is_signed_type<signed short>::value == true);
        COMPILE_TIME_ASSERT(is_signed_type<signed int>::value == true);
        COMPILE_TIME_ASSERT(is_signed_type<signed long>::value == true);
        COMPILE_TIME_ASSERT(is_unsigned_type<signed char>::value == false);
        COMPILE_TIME_ASSERT(is_unsigned_type<signed short>::value == false);
        COMPILE_TIME_ASSERT(is_unsigned_type<signed int>::value == false);
        COMPILE_TIME_ASSERT(is_unsigned_type<signed long>::value == false);

        COMPILE_TIME_ASSERT(is_unsigned_type<unsigned char>::value == true);
        COMPILE_TIME_ASSERT(is_unsigned_type<unsigned short>::value == true);
        COMPILE_TIME_ASSERT(is_unsigned_type<unsigned int>::value == true);
        COMPILE_TIME_ASSERT(is_unsigned_type<unsigned long>::value == true);
        COMPILE_TIME_ASSERT(is_signed_type<unsigned char>::value == false);
        COMPILE_TIME_ASSERT(is_signed_type<unsigned short>::value == false);
        COMPILE_TIME_ASSERT(is_signed_type<unsigned int>::value == false);
        COMPILE_TIME_ASSERT(is_signed_type<unsigned long>::value == false);


    }


    void test_call_if_valid() 
    {
        int value = 0;

        auto foo = [&](int a, int b) { value += a + b; };
        auto bar = [&](std::string) { value++; };
        auto baz = [&]() { value++; };

        DLIB_TEST(value == 0);
        DLIB_TEST(call_if_valid(baz));
        DLIB_TEST(value == 1);
        DLIB_TEST(!call_if_valid(foo));
        DLIB_TEST(value == 1);
        DLIB_TEST(!call_if_valid(bar));
        DLIB_TEST(value == 1);
        DLIB_TEST(call_if_valid(bar, "stuff"));
        DLIB_TEST(value == 2);
        DLIB_TEST(!call_if_valid(baz, "stuff"));
        DLIB_TEST(value == 2);
        DLIB_TEST(call_if_valid(foo, 3, 1));
        DLIB_TEST(value == 6);
        DLIB_TEST(!call_if_valid(bar, 3, 1));
        DLIB_TEST(value == 6);


        // make sure stateful lambdas are modified when called
        value = 0;
        int i = 0;
        auto stateful = [&value, i]() mutable { ++i; value = i; };
        DLIB_TEST(call_if_valid(stateful));
        DLIB_TEST(value == 1);
        DLIB_TEST(call_if_valid(stateful));
        DLIB_TEST(value == 2);
        DLIB_TEST(call_if_valid(stateful));
        DLIB_TEST(value == 3);
    }


    class metaprogramming_tester : public tester
    {
    public:
        metaprogramming_tester (
        ) :
            tester ("test_metaprogramming",
                    "Runs tests on the metaprogramming objects and functions.")
        {}

        void perform_test (
        )
        {
            metaprogramming_test();
            test_call_if_valid();
        }
    } a;

}



