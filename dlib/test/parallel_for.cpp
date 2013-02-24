// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/threads.h>
#include <vector>
#include <sstream>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.parallel_for");

    class assign_element
    {
    public:

        assign_element(
            std::vector<int>& vect_
        ) : vect(vect_){}

        std::vector<int>& vect;

        void go (long i ) 
        {
            DLIB_TEST( 0 <= i && i < vect.size());
            vect[i] = i;
        }

        void operator() (long i ) const
        {
            DLIB_TEST( 0 <= i && i < vect.size());
            vect[i] = i;
        }

    };

    void test_parallel_for(long start)
    {
        std::vector<int> vect(200,0);

        parallel_for(4, start, vect.size(), assign_element(vect));

        for (unsigned long i = 0; i < start;  ++i)
        {
            DLIB_TEST(vect[i] == 0);
        }
        for (unsigned long i = start; i < vect.size(); ++i)
        {
            DLIB_TEST(vect[i] == i);
        }
    }

    void test_parallel_for2(long start)
    {
        std::vector<int> vect(200,0);

        assign_element temp(vect);
        parallel_for(4, start, vect.size(), temp, &assign_element::go);

        for (unsigned long i = 0; i < start;  ++i)
        {
            DLIB_TEST(vect[i] == 0);
        }
        for (unsigned long i = start; i < vect.size(); ++i)
        {
            DLIB_TEST(vect[i] == i);
        }
    }


    class test_parallel_for_routines : public tester
    {
    public:
        test_parallel_for_routines (
        ) :
            tester (
                "test_parallel_for",       // the command line argument name for this test
                "Run tests on the parallel_for routines.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }

        void perform_test (
        )
        {
            test_parallel_for(0);
            test_parallel_for(30);
            test_parallel_for(50);
            test_parallel_for2(0);
            test_parallel_for2(30);
            test_parallel_for2(50);
        }
    };

    test_parallel_for_routines a;

}




