// Copyright (C) 2012  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/svm.h>
#include <dlib/rand.h>
#include <dlib/string.h>
#include <vector>
#include <sstream>
#include <ctime>

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;
    dlib::logger dlog("test.sparse_vector");


    class sparse_vector_tester : public tester
    {
    public:
        sparse_vector_tester (
        ) :
            tester (
                "test_sparse_vector",       // the command line argument name for this test
                "Run tests on the sparse_vector routines.", // the command line argument description
                0                     // the number of command line arguments for this test
            )
        {
        }


        void perform_test (
        )
        {
            std::map<unsigned int, double> v;
            v[4] = 8;
            v[2] = -4;
            v[9] = 10;

            DLIB_TEST(max(v) == 10);
            DLIB_TEST(min(v) == -4);

            v.clear();
            v[4] = 8;
            v[9] = 10;
            DLIB_TEST(max(v) == 10);
            DLIB_TEST(min(v) == 0);


            v.clear();
            v[4] = -9;
            v[9] = -4;
            DLIB_TEST(max(v) == 0);
            DLIB_TEST(min(v) == -9);

        }
    };

    sparse_vector_tester a;

}



