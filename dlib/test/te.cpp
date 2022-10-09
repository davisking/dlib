// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include "../any/storage.h"
#include "../any/any_function.h"

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

    template <typename Storage>
    void test_storage_basic() 
    {
        Storage a;
        DLIB_TEST(a.get_ptr() == nullptr);
        DLIB_TEST(a.is_empty());
        DLIB_TEST(a.template contains<int>() == false);
        int value = 5;
        a = value;
        DLIB_TEST(a.get_ptr() != nullptr);
        DLIB_TEST(!a.is_empty());
        DLIB_TEST(a.template contains<int>());
        DLIB_TEST(!a.template contains<std::string>());
        DLIB_TEST(a.template cast_to<int>() == 5);
        DLIB_TEST(a.template get<int>() == 5);

        Storage b = a;
        DLIB_TEST(b.get_ptr() != nullptr);
        DLIB_TEST(!b.is_empty());
        DLIB_TEST(b.template contains<int>());
        DLIB_TEST(b.template cast_to<int>() == 5);
        DLIB_TEST(b.template get<int>() == 5);
        DLIB_TEST(a.get_ptr() != nullptr);
        DLIB_TEST(!a.is_empty());
        DLIB_TEST(a.template contains<int>());
        DLIB_TEST(a.template cast_to<int>() == 5);
        DLIB_TEST(a.template get<int>() == 5);

        DLIB_TEST(*static_cast<int*>(a.get_ptr()) == 5);
        DLIB_TEST(*static_cast<int*>(b.get_ptr()) == 5);


        a.clear();
        DLIB_TEST(a.get_ptr() == nullptr);
        DLIB_TEST(a.is_empty());
        DLIB_TEST(!a.template contains<int>());

    }

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
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());

            storage_heap str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());

            DLIB_TEST(!str2.is_empty());
            storage_heap str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1); //pointer was moved with storage_heap so move constructor not called
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(str2.is_empty());

            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(!str2.contains<A>());
            DLIB_TEST(!str3.contains<int>());
            DLIB_TEST(str3.contains<A>());
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
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());

            storage_stack<sizeof(A)> str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());

            DLIB_TEST(!str2.is_empty());
            storage_stack<sizeof(A)> str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 2);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str2.is_empty());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());
            DLIB_TEST(!str3.contains<int>());
            DLIB_TEST(str3.contains<A>());
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
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());

            storage_sbo<4> str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());

            DLIB_TEST(!str2.is_empty());
            storage_sbo<4> str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);   // SBO 4 isn't big enough, so heap is used, so pointers are moved
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(str2.is_empty());

            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(!str2.contains<A>());
            DLIB_TEST(!str3.contains<int>());
            DLIB_TEST(str3.contains<A>());
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
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());

            storage_sbo<sizeof(A)> str2 = str1;
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());

            DLIB_TEST(!str2.is_empty());
            storage_sbo<sizeof(A)> str3 = std::move(str2);
            DLIB_TEST(copy_counter == 1);
            DLIB_TEST(move_counter == 2);   // SBO is big enough, so stack is used, so move constructor is used
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str2.is_empty());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());
            DLIB_TEST(!str3.contains<int>());
            DLIB_TEST(str3.contains<A>());
        }

        DLIB_TEST(copy_counter == 1);
        DLIB_TEST(move_counter == 2);
        DLIB_TEST(delete_counter == 4);

        copy_counter = move_counter = delete_counter = 0;

        {
            storage_shared str1{A{copy_counter, move_counter, delete_counter}};
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());

            storage_shared str2 = str1;
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());

            DLIB_TEST(!str2.is_empty());
            storage_shared str3 = std::move(str2);
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 1);
            DLIB_TEST(delete_counter == 1);
            DLIB_TEST(str2.is_empty());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(!str2.contains<A>());
            DLIB_TEST(!str3.contains<int>());
            DLIB_TEST(str3.contains<A>());
        }

        DLIB_TEST(copy_counter == 0);
        DLIB_TEST(move_counter == 1);
        DLIB_TEST(delete_counter == 2);

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
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());

            storage_view str2 = str1;
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 0);
            DLIB_TEST(delete_counter == 0);
            DLIB_TEST(!str1.contains<int>());
            DLIB_TEST(str1.contains<A>());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());

            DLIB_TEST(!str2.is_empty());
            storage_view str3 = std::move(str2);
            DLIB_TEST(copy_counter == 0);
            DLIB_TEST(move_counter == 0);
            DLIB_TEST(delete_counter == 0);
            DLIB_TEST(!str2.is_empty());
            DLIB_TEST(!str2.contains<int>());
            DLIB_TEST(str2.contains<A>());
            DLIB_TEST(!str3.contains<int>());
            DLIB_TEST(str3.contains<A>());
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
        dlib::any_function_view<int(int)> g(f);

        DLIB_TEST(f(1) == 43);
        DLIB_TEST(g(1) == 44);
        DLIB_TEST(f(1) == 45);
        DLIB_TEST(g(1) == 46);
    }

    void global_function1(int& a) {a += 1;}

    template<template<class...> class Function, class Storage>
    void test_function_pointer()
    {
        {
            Function<Storage, void(int&)> f{global_function1};
            DLIB_TEST(f);
            int a = 0;
            f(a);
            DLIB_TEST(a == 1);

            f = nullptr;
            DLIB_TEST(!f);
            f = global_function1;
            DLIB_TEST(f);
            f(a);
            DLIB_TEST(a == 2);
        }

        /*! Use address !*/
        {
            Function<Storage, void(int&)> f{&global_function1};
            DLIB_TEST(f);
            int a = 0;
            f(a);
            DLIB_TEST(a == 1);

            f = nullptr;
            DLIB_TEST(!f);
            f = &global_function1;
            DLIB_TEST(f);
            f(a);
            DLIB_TEST(a == 2);
        }
    }

    struct member_function
    {
        void increment(int& a) const {a += 1;}
    };

    template<template<class...> class Function, class Storage>
    void test_member_pointer()
    {
        {
            member_function obj;

            Function<Storage, void(member_function&, int&)> f{&member_function::increment};
            DLIB_TEST(f);
            int a = 0;
            f(obj, a);
            DLIB_TEST(a == 1);

            f = nullptr;
            DLIB_TEST(!f);
            f = &member_function::increment;
            DLIB_TEST(f);
            f(obj, a);
            DLIB_TEST(a == 2);
        }

        /*! Use std::mem_fn !*/
        {
            member_function obj;

            Function<Storage, void(member_function&, int&)> f{std::mem_fn(&member_function::increment)};
            DLIB_TEST(f);
            int a = 0;
            f(obj, a);
            DLIB_TEST(a == 1);

            f = nullptr;
            DLIB_TEST(!f);
            f = std::mem_fn(&member_function::increment);
            DLIB_TEST(f);
            f(obj, a);
            DLIB_TEST(a == 2);
        }

        /*! Use std::bind !*/
        {
            using namespace std::placeholders;

            member_function obj;

            Function<Storage, void(int&)> f{std::bind(&member_function::increment, obj, _1)};
            DLIB_TEST(f);
            int a = 0;
            f(a);
            DLIB_TEST(a == 1);

            f = nullptr;
            DLIB_TEST(!f);
            f = std::bind(&member_function::increment, obj, _1);
            DLIB_TEST(f);
            f(a);
            DLIB_TEST(a == 2);
        }
    }

    class te_tester : public tester
    {
    public:
        te_tester (
        ) : tester ("test_te",
                    "Runs tests on type erasure tools")
        {}

        void perform_test ()
        {
            test_type_erasure();
            dlog << LINFO << "test_storage_basic<storage_heap>()";
            test_storage_basic<storage_heap>();
            dlog << LINFO << "test_storage_basic<storage_sbo<20>>()";
            test_storage_basic<storage_sbo<20>>();
            dlog << LINFO << "test_storage_basic<storage_stack<20>>()";
            test_storage_basic<storage_stack<20>>();
            dlog << LINFO << "test_storage_basic<storage_shared>()";
            test_storage_basic<storage_shared>();
            dlog << LINFO << "test_storage_basic<storage_view>()";
            test_storage_basic<storage_view>();
            
            test_function<dlib::any_function<int(int)>>();
            test_function_view();
            test_function_pointer<dlib::any_function_basic, storage_heap>();
            test_function_pointer<dlib::any_function_basic, storage_stack<32>>();
            test_function_pointer<dlib::any_function_basic, storage_sbo<32>>();
            test_function_pointer<dlib::any_function_basic, storage_shared>();
            test_member_pointer<dlib::any_function_basic, storage_heap>();
            test_member_pointer<dlib::any_function_basic, storage_stack<32>>();
            test_member_pointer<dlib::any_function_basic, storage_sbo<32>>();
            test_member_pointer<dlib::any_function_basic, storage_shared>();
        }
    } a;
}
