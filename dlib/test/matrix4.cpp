// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/matrix.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../stl_checked.h"
#include "../array.h"
#include "../rand.h"

#include "tester.h"
#include <dlib/memory_manager_stateless.h>
#include <dlib/array2d.h>

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.matrix4");

    void matrix_test (
    )
    /*!
        ensures
            - runs tests on the matrix stuff compliance with the specs
    !*/
    {        
        typedef memory_manager_stateless<char>::kernel_2_2a MM;
        print_spinner();

        {
            matrix<double,3,3> m = round(10*randm(3,3));
            matrix<double,3,1> v = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }

        {
            matrix<double,3,3> m = round(10*randm(3,3));
            matrix<double,1,3> v = round(10*randm(1,3));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }

        {
            matrix<double> m = round(10*randm(3,3));
            matrix<double,1,3> v = round(10*randm(1,3));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }

        {
            matrix<double> m = round(10*randm(3,3));
            matrix<double,0,3> v = round(10*randm(1,3));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }


        {
            matrix<double> m = round(10*randm(3,3));
            matrix<double,1,0> v = round(10*randm(1,3));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }

        {
            matrix<double> m = round(10*randm(3,3));
            matrix<double,3,0> v = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }


        {
            matrix<double> m = round(10*randm(3,3));
            matrix<double,0,1> v = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }

        {
            matrix<double,3,3> m = round(10*randm(3,3));
            matrix<double,3,0> v = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }


        {
            matrix<double,3,3> m = round(10*randm(3,3));
            matrix<double,0,1> v = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v) , m*tmp(diagm(v)) ));
            DLIB_TEST(equal( scale_columns(m,v) , m*tmp(diagm(v)) ));

            DLIB_TEST(equal( diagm(v)*m , tmp(diagm(v))*m )); 
            DLIB_TEST(equal( scale_rows(m,v) , tmp(diagm(v))*m )); 
        }

        {
            matrix<double,3,5> m = round(10*randm(3,5));
            matrix<double,0,1> v1 = round(10*randm(5,1));
            matrix<double,0,1> v2 = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v1) , m*tmp(diagm(v1)) ));
            DLIB_TEST(equal( scale_columns(m,v1) , m*tmp(diagm(v1)) ));

            DLIB_TEST(equal( diagm(v2)*m , tmp(diagm(v2))*m )); 
            DLIB_TEST(equal( scale_rows(m,v2) , tmp(diagm(v2))*m )); 
        }

        {
            matrix<double,3,5> m = round(10*randm(3,5));
            matrix<double,5,1> v1 = round(10*randm(5,1));
            matrix<double,3,1> v2 = round(10*randm(3,1));

            DLIB_TEST(equal( m*diagm(v1) , m*tmp(diagm(v1)) ));
            DLIB_TEST(equal( scale_columns(m,v1) , m*tmp(diagm(v1)) ));

            DLIB_TEST(equal( diagm(v2)*m , tmp(diagm(v2))*m )); 
            DLIB_TEST(equal( scale_rows(m,v2) , tmp(diagm(v2))*m )); 
        }

    }

    void test_stuff()
    {
        print_spinner();

        {
            matrix<double> m(3,3), lr(3,3), ud(3,3);

            m = 1,2,3,
                4,5,6,
                7,8,9;


            lr = 3,2,1,
                 6,5,4,
                 9,8,7;

            ud = 7,8,9,
                 4,5,6,
                 1,2,3;

            DLIB_TEST(lr == fliplr(m));
            DLIB_TEST(ud == flipud(m));
        }
        {
            matrix<double> m(3,2), lr(3,2), ud(3,2);

            m = 1,2,
                3,4,
                5,6;

            lr = 2,1,
                 4,3,
                 6,5;

            ud = 5,6,
                 3,4,
                 1,2;

            DLIB_TEST(lr == fliplr(m));
            DLIB_TEST(ud == flipud(m));
        }

        {
            matrix<int> a, b;

            a = matrix_cast<int>(round(10*randm(3,3)));
            b = a;

            b *= b;
            DLIB_TEST(b == a*a);
        }
    }





    class matrix_tester : public tester
    {
    public:
        matrix_tester (
        ) :
            tester ("test_matrix4",
                    "Runs tests on the scale_rows and scale_columns functions.")
        {}

        void perform_test (
        )
        {
            test_stuff();
            for (int i = 0; i < 10; ++i)
                matrix_test();
        }
    } a;

}



