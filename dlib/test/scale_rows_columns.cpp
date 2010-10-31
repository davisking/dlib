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

    logger dlog("test.scale_rows_columns");

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




    class matrix_tester : public tester
    {
    public:
        matrix_tester (
        ) :
            tester ("test_scale_rows_columns",
                    "Runs tests on the scale_rows and scale_columns functions.")
        {}

        void perform_test (
        )
        {
            for (int i = 0; i < 10; ++i)
                matrix_test();
        }
    } a;

}



