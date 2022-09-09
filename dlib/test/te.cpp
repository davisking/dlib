// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include "../te.h"
#include "../function.h"

namespace
{

    using namespace test;
    using namespace dlib;
    using namespace te;

    logger dlog("test.te");

    struct A
    {
        A(
            int& copy_counter_,
            int& move_counter_,
            int& delete_counter_
        ) : copy_counter{copy_counter_},
            move_counter{move_counter_},
            delete_counter{delete_counter_} 
        {}

        A(const A& other)
        :   copy_counter{other.copy_counter},
            move_counter{other.move_counter},
            delete_counter{other.delete_counter} 
        {
            ++copy_counter;
        }

        A(A&& other)
        :   copy_counter{other.copy_counter},
            move_counter{other.move_counter},
            delete_counter{other.delete_counter} 
        {
            ++move_counter;
        }

        ~A()
        {
            ++delete_counter;
        }

        int& copy_counter;
        int& move_counter;
        int& delete_counter;
    };

    void test_type_erasure()
    {
        int copy_counter = 0;
        int move_counter = 0;
        int delete_counter = 0;

        {
            storage_heap str1{A{copy_counter, move_counter, delete_counter}};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_heap str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_heap str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1); //pointer was moved with storage_heap so move constructor not called
            DLIB_TEST(delete_counter == 1);
        }

        DLIB_TEST(copy_counter == 1);
        DLIB_TEST(move_counter == 1);
        DLIB_TEST(delete_counter == 3); //one of the pointers was moved so one of the destructors was not called
        
        copy_counter = move_counter = delete_counter = 0;

        {
            storage_stack<sizeof(A)> str1{A{copy_counter, move_counter, delete_counter}};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_stack<sizeof(A)> str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_stack<sizeof(A)> str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 2);
            DLIB_TEST(delete_counter == 1);
        }

        DLIB_TEST(copy_counter == 1);
        DLIB_TEST(move_counter == 2);
        DLIB_TEST(delete_counter == 4);

        copy_counter = move_counter = delete_counter = 0;

        {
            storage_sbo<4> str1{A{copy_counter, move_counter, delete_counter}};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_sbo<4> str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_sbo<4> str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);   // SBO 4 isn't big enough, so heap is used, so pointers are moved
            DLIB_TEST(delete_counter == 1);
        }

        DLIB_TEST(copy_counter == 1);
        DLIB_TEST(move_counter == 1);
        DLIB_TEST(delete_counter == 3); //one of the pointers was moved so one of the destructors was not called

        copy_counter = move_counter = delete_counter = 0;

        {
            storage_sbo<sizeof(A)> str1{A{copy_counter, move_counter, delete_counter}};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_sbo<sizeof(A)> str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);

            storage_sbo<sizeof(A)> str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 2);   // SBO is big enough, so stack is used, so move constructor is used
            DLIB_TEST(delete_counter == 1);
        }

        DLIB_TEST(copy_counter == 1);
        DLIB_TEST(move_counter == 2);
        DLIB_TEST(delete_counter == 4);

        copy_counter = move_counter = delete_counter = 0;

        {
            A a{copy_counter, move_counter, delete_counter};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 0);
            DLIB_TEST(delete_counter == 0);

            storage_view str1{a};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 0);
            DLIB_TEST(delete_counter == 0);

            storage_view str2 = str1;
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 0);
            DLIB_TEST(delete_counter == 0);

            storage_view str3 = std::move(str2);
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 0);
            DLIB_TEST(delete_counter == 0);
        }

        DLIB_TEST(copy_counter == 0);
        DLIB_TEST(move_counter == 0);
        DLIB_TEST(delete_counter == 1);
    }

    template<typename Function>
    void test_function()
    {
        Function f1;
        DLIB_TEST(!f1);

        int a = 42;
        Function f2{[a](int b) {return a + b;}};
        DLIB_TEST(f2);
        DLIB_TEST(f2(1) == 43);

        Function f3{f2};
        DLIB_TEST(f2);
        DLIB_TEST(f3);
        DLIB_TEST(f2(2) == 44);
        DLIB_TEST(f3(2) == 44);

        f1 = f2;
        DLIB_TEST(f1);
        DLIB_TEST(f2);
        DLIB_TEST(f3);
        DLIB_TEST(f1(2) == 44);
        DLIB_TEST(f2(2) == 44);
        DLIB_TEST(f3(2) == 44);

        Function f4{std::move(f2)};
        DLIB_TEST(f1);
        DLIB_TEST(!f2);
        DLIB_TEST(f3);
        DLIB_TEST(f4);
        DLIB_TEST(f1(2) == 44);
        DLIB_TEST(f3(2) == 44);
        DLIB_TEST(f4(2) == 44);

        f2 = std::move(f1);
        DLIB_TEST(!f1);
        DLIB_TEST(f2);
        DLIB_TEST(f3);
        DLIB_TEST(f4);
        DLIB_TEST(f2(2) == 44);
        DLIB_TEST(f3(2) == 44);
        DLIB_TEST(f4(2) == 44);
    }

    void test_function_view()
    {
        auto f = [a = 42](int i) mutable {a += i; return a;};
        dlib::function_view<int(int)> g(f);

        DLIB_TEST(f(1) == 43);
        DLIB_TEST(g(1) == 44);
        DLIB_TEST(f(1) == 45);
        DLIB_TEST(g(1) == 46);
    }



    class dnn2_tester : public tester
    {
    public:
        dnn2_tester (
        ) : tester ("test_te",
                    "Runs tests on type erasure tools")
        {}

        void perform_test ()
        {
            test_type_erasure();
            
            test_function<dlib::function_heap<int(int)>>();
            test_function<dlib::function_stack<int(int), 32>>();
            test_function<dlib::function_sbo<int(int), 32>>();
            test_function_view();
        }
    } a;
}