// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/misc_api.h>
#include <dlib/threads.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.thread_pool");


    struct some_struct : noncopyable
    {
        float val;
    };

    int global_var = 0;

    struct add_functor
    {
        template <typename T, typename U, typename V>
        void operator()(T a, U b, V& res)
        {
            dlib::sleep(20);
            res = a + b;
        }

        void set_global_var() { global_var = 9; }
        void set_global_var_const() const { global_var = 9; }

        void operator()()
        {
            global_var = 9;
        }
    };


    void set_global_var() {  global_var = 9; }

    void gset_struct_to_zero (some_struct& a) { a.val = 0; }
    void gset_to_zero (int& a) { a = 0; }
    void gincrement (int& a) { ++a; }
    void gadd (int a, const int& b, int& res) { dlib::sleep(20); res = a + b; }
    void gadd1(int& a, int& res) { res += a; }
    void gadd2 (int c, int a, const int& b, int& res) { dlib::sleep(20); res = a + b + c; }

    class thread_pool_tester : public tester
    {
    public:
        thread_pool_tester (
        ) :
            tester ("test_thread_pool",
                    "Runs tests on the thread_pool component.")
        {}

        void perform_test (
        )
        {
            add_functor f;
            for (int num_threads= 0; num_threads < 4; ++num_threads)
            {
                future<int> a, b, c, res;
                thread_pool tp(num_threads);
                print_spinner();

                future<some_struct> obj;


                for (int i = 0; i < 4; ++i)
                {
                    a = 1;
                    b = 2;
                    c = 3;
                    res = 4;


                    DLIB_TEST(a==a);
                    DLIB_TEST(a!=b);
                    DLIB_TEST(a==1);

                    tp.add_task(gset_to_zero, a);
                    tp.add_task(gset_to_zero, b);
                    tp.add_task(*this, &thread_pool_tester::set_to_zero, c);
                    tp.add_task(gset_to_zero, res);
                    DLIB_TEST(a == 0);
                    DLIB_TEST(b == 0);
                    DLIB_TEST(c == 0);
                    DLIB_TEST(res == 0);


                    tp.add_task(gincrement, a);
                    tp.add_task(*this, &thread_pool_tester::increment, b);
                    tp.add_task(*this, &thread_pool_tester::increment, c);
                    tp.add_task(gincrement, res);

                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 1);
                    DLIB_TEST(c == 1);
                    DLIB_TEST(res == 1);

                    tp.add_task(&gincrement, a);
                    tp.add_task(*this, &thread_pool_tester::increment, b);
                    tp.add_task(*this, &thread_pool_tester::increment, c);
                    tp.add_task(&gincrement, res);
                    tp.add_task(gincrement, a);
                    tp.add_task(*this, &thread_pool_tester::increment, b);
                    tp.add_task(*this, &thread_pool_tester::increment, c);
                    tp.add_task(gincrement, res);

                    DLIB_TEST(a == 3);
                    DLIB_TEST(b == 3);
                    DLIB_TEST(c == 3);
                    DLIB_TEST(res == 3);

                    tp.add_task(*this, &thread_pool_tester::increment, c);
                    tp.add_task(gincrement, res);
                    DLIB_TEST(c == 4);
                    DLIB_TEST(res == 4);


                    tp.add_task(gadd, a, b, res);
                    DLIB_TEST(res == a+b);
                    DLIB_TEST(res == 6);
                    a = 3;
                    b = 4;
                    res = 99;
                    DLIB_TEST(res == 99);
                    tp.add_task(*this, &thread_pool_tester::add, a, b, res);
                    DLIB_TEST(res == a+b);
                    DLIB_TEST(res == 7);

                    a = 1;
                    b = 2;
                    c = 3;
                    res = 88;
                    DLIB_TEST(res == 88);
                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 2);
                    DLIB_TEST(c == 3);

                    tp.add_task(gadd2, a, b, c, res);
                    DLIB_TEST(res == 6);
                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 2);
                    DLIB_TEST(c == 3);

                    a = 1;
                    b = 2;
                    c = 3;
                    res = 88;
                    DLIB_TEST(res == 88);
                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 2);
                    DLIB_TEST(c == 3);
                    tp.add_task(*this, &thread_pool_tester::add2, a, b, c, res);
                    DLIB_TEST(res == 6);
                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 2);
                    DLIB_TEST(c == 3);

                    a = 1;
                    b = 2;
                    c = 3;
                    res = 88;
                    tp.add_task(gadd1, a, b);
                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 3);
                    a = 2;
                    tp.add_task(*this, &thread_pool_tester::add1, a, b);
                    DLIB_TEST(a == 2);
                    DLIB_TEST(b == 5);


                    val = 4;
                    uint64 id = tp.add_task(*this, &thread_pool_tester::zero_val);
                    tp.wait_for_task(id);
                    DLIB_TEST(val == 0);
                    id = tp.add_task(*this, &thread_pool_tester::accum2, 1,2);
                    tp.wait_for_all_tasks();
                    DLIB_TEST(val == 3);
                    id = tp.add_task(*this, &thread_pool_tester::accum1, 3);
                    tp.wait_for_task(id);
                    DLIB_TEST(val == 6);


                    obj.get().val = 8;
                    DLIB_TEST(obj.get().val == 8);
                    tp.add_task(gset_struct_to_zero, obj);
                    DLIB_TEST(obj.get().val == 0);
                    obj.get().val = 8;
                    DLIB_TEST(obj.get().val == 8);
                    tp.add_task(*this,&thread_pool_tester::set_struct_to_zero, obj);
                    DLIB_TEST(obj.get().val == 0);

                    a = 1;
                    b = 2;
                    res = 0;
                    tp.add_task(f, a, b, res);
                    DLIB_TEST(a == 1);
                    DLIB_TEST(b == 2);
                    DLIB_TEST(res == 3);


                    global_var = 0;
                    DLIB_TEST(global_var == 0);
                    id = tp.add_task(&set_global_var);
                    tp.wait_for_task(id);
                    DLIB_TEST(global_var == 9);

                    global_var = 0;
                    DLIB_TEST(global_var == 0);
                    id = tp.add_task(f);
                    tp.wait_for_task(id);
                    DLIB_TEST(global_var == 9);

                    global_var = 0;
                    DLIB_TEST(global_var == 0);
                    id = tp.add_task(f, &add_functor::set_global_var);
                    tp.wait_for_task(id);
                    DLIB_TEST(global_var == 9);


                    global_var = 0;
                    DLIB_TEST(global_var == 0);
                    id = tp.add_task(f, &add_functor::set_global_var_const);
                    tp.wait_for_task(id);
                    DLIB_TEST(global_var == 9);



                }

                // add this task just to to perterb the thread pool before it goes out of scope
                tp.add_task(f, a, b, res);
            }
        }

        long val;
        void accum1(long a) { val += a; }
        void accum2(long a, long b) { val += a + b; }
        void zero_val() { dlib::sleep(20); val = 0; }


        void set_struct_to_zero (some_struct& a) { a.val = 0; }
        void set_to_zero (int& a) { dlib::sleep(20); a = 0; }
        void increment (int& a) const { dlib::sleep(20); ++a; }
        void add (int a, const int& b, int& res) { dlib::sleep(20); res = a + b; }
        void add1(int& a, int& res) const { res += a; }
        void add2 (int c, int a, const int& b, int& res) { res = a + b + c; }


    } a;


}



