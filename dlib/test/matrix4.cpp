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

        {
            matrix<double> m(2,3), m2(2,3);

            m = 1,2,3,
                4,5,6;


            m2 = 3,4,5,
                 6,7,8;

            DLIB_TEST(m + 2 == m2);
            DLIB_TEST(2 + m == m2);

            m += 2;
            DLIB_TEST(m == m2);
            m -= 2;

            m2 = 0,1,2,
                 3,4,5;

            DLIB_TEST(m - 1 == m2);

            m -= 1;
            DLIB_TEST(m == m2);
            m += 1;


            m2 = 5,4,3,
                 2,1,0;

            DLIB_TEST(6 - m == m2);
        }

        {
            matrix<float> m(2,3), m2(2,3);

            m = 1,2,3,
                4,5,6;


            m2 = 3,4,5,
                 6,7,8;

            DLIB_TEST(m + 2 == m2);
            DLIB_TEST(2 + m == m2);

            m += 2;
            DLIB_TEST(m == m2);
            m -= 2;

            m2 = 0,1,2,
                 3,4,5;

            DLIB_TEST(m - 1 == m2);

            m -= 1;
            DLIB_TEST(m == m2);
            m += 1;


            m2 = 5,4,3,
                 2,1,0;

            DLIB_TEST(6 - m == m2);
        }

        {
            matrix<int> m(2,3), m2(2,3);

            m = 1,2,3,
                4,5,6;


            m2 = 3,4,5,
                 6,7,8;

            DLIB_TEST(m + 2 == m2);
            DLIB_TEST(2 + m == m2);

            m += 2;
            DLIB_TEST(m == m2);
            m -= 2;

            m2 = 0,1,2,
                 3,4,5;

            DLIB_TEST(m - 1 == m2);

            m -= 1;
            DLIB_TEST(m == m2);
            m += 1;


            m2 = 5,4,3,
                 2,1,0;

            DLIB_TEST(6 - m == m2);
        }

        {
            matrix<int,2,3> m, m2;

            m = 1,2,3,
                4,5,6;


            m2 = 3,4,5,
                 6,7,8;

            DLIB_TEST(m + 2 == m2);
            DLIB_TEST(2 + m == m2);

            m += 2;
            DLIB_TEST(m == m2);
            m -= 2;

            m2 = 0,1,2,
                 3,4,5;

            DLIB_TEST(m - 1 == m2);

            m -= 1;
            DLIB_TEST(m == m2);
            m += 1;


            m2 = 5,4,3,
                 2,1,0;

            DLIB_TEST(6 - m == m2);
        }

        {
            matrix<double> m(2,3), m2(3,2);

            m = 1,2,3,
                4,5,6;

            m2 = 2,5,
                 3,6,
                 4,7;

            DLIB_TEST(trans(m+1) == m2);
            DLIB_TEST(trans(m)+1 == m2);
            DLIB_TEST(1+trans(m) == m2);
            DLIB_TEST(1+m-1 == m);

            m = trans(m+1);
            DLIB_TEST(m == m2);
            m = trans(m-1);
            DLIB_TEST(trans(m+1) == m2);
            m = trans(m)+1;
            DLIB_TEST(m == m2);
        }

        {
            matrix<double> d(3,1), di(3,1);
            matrix<double> m(3,3);
             
            m = 1,2,3,
                4,5,6,
                7,8,9;

            d = 1,2,3;

            di = 1, 1/2.0, 1/3.0;

            DLIB_TEST(inv(diagm(d)) == diagm(di));
            DLIB_TEST(pinv(diagm(d)) == diagm(di));
            DLIB_TEST(inv(diagm(d))*m == tmp(diagm(di))*m);
            DLIB_TEST(m*inv(diagm(d)) == m*tmp(diagm(di)));

            DLIB_TEST(equal(inv(diagm(d)) + m , tmp(diagm(di)) + m));
            DLIB_TEST(equal(m + inv(diagm(d)) , tmp(diagm(di)) + m));

            DLIB_TEST((m + identity_matrix<double>(3) == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>() == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + 2*identity_matrix<double>(3) == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + 2*identity_matrix<double,3>() == m + 2*tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + identity_matrix<double>(3)*2 == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>()*2 == m + 2*tmp(identity_matrix<double,3>())));

            DLIB_TEST((identity_matrix<double>(3) + m == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((identity_matrix<double,3>() + m == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((2*identity_matrix<double>(3) + m == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((2*identity_matrix<double,3>() + m == m + 2*tmp(identity_matrix<double,3>())));

        }
        {
            matrix<double,3,1> d(3,1), di(3,1);
            matrix<double,3,3> m(3,3);
             
            m = 1,2,3,
                4,5,6,
                7,8,9;

            d = 1,2,3;

            di = 1, 1/2.0, 1/3.0;

            DLIB_TEST(equal(inv(diagm(d)) , diagm(di)));
            DLIB_TEST(equal(inv(diagm(d)) , diagm(di)));
            DLIB_TEST(equal(inv(diagm(d))*m , tmp(diagm(di))*m));
            DLIB_TEST(equal(m*inv(diagm(d)) , m*tmp(diagm(di))));

            DLIB_TEST_MSG(equal(inv(diagm(d)) + m , tmp(diagm(di)) + m), 
                          (inv(diagm(d)) + m) - (tmp(diagm(di)) + m) );
            DLIB_TEST(equal(m + inv(diagm(d)) , tmp(diagm(di)) + m));


            DLIB_TEST((m + identity_matrix<double>(3) == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>() == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + 2*identity_matrix<double>(3) == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + 2*identity_matrix<double,3>() == m + 2*tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + identity_matrix<double>(3)*2 == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>()*2 == m + 2*tmp(identity_matrix<double,3>())));

            DLIB_TEST((identity_matrix<double>(3) + m == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((identity_matrix<double,3>() + m == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((2*identity_matrix<double>(3) + m == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((2*identity_matrix<double,3>() + m == m + 2*tmp(identity_matrix<double,3>())));
        }

        {
            matrix<double,1,3> d(1,3), di(1,3);
            matrix<double,3,3> m(3,3);
             
            m = 1,2,3,
                4,5,6,
                7,8,9;

            d = 1,2,3;

            di = 1, 1/2.0, 1/3.0;

            DLIB_TEST(equal(inv(diagm(d)) , diagm(di)));
            DLIB_TEST(equal(inv(diagm(d)) , diagm(di)));
            DLIB_TEST(equal(inv(diagm(d))*m , tmp(diagm(di))*m));
            DLIB_TEST(equal(m*inv(diagm(d)) , m*tmp(diagm(di))));

            DLIB_TEST(equal(inv(diagm(d)) + m , tmp(diagm(di)) + m));
            DLIB_TEST(equal(m + inv(diagm(d)) , tmp(diagm(di)) + m));


            DLIB_TEST((m + identity_matrix<double>(3) == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>() == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + 2*identity_matrix<double>(3) == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + 2*identity_matrix<double,3>() == m + 2*tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + identity_matrix<double>(3)*2 == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>()*2 == m + 2*tmp(identity_matrix<double,3>())));

            DLIB_TEST((identity_matrix<double>(3) + m == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((identity_matrix<double,3>() + m == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((2*identity_matrix<double>(3) + m == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((2*identity_matrix<double,3>() + m == m + 2*tmp(identity_matrix<double,3>())));
        }

        {
            matrix<double,1,0> d(1,3), di(1,3);
            matrix<double,0,3> m(3,3);
             
            m = 1,2,3,
                4,5,6,
                7,8,9;

            d = 1,2,3;

            di = 1, 1/2.0, 1/3.0;

            DLIB_TEST(equal(inv(diagm(d)) , diagm(di)));
            DLIB_TEST(equal(inv(diagm(d)) , diagm(di)));
            DLIB_TEST(equal(inv(diagm(d))*m , tmp(diagm(di))*m));
            DLIB_TEST(equal(m*inv(diagm(d)) , m*tmp(diagm(di))));

            DLIB_TEST(equal(inv(diagm(d)) + m , tmp(diagm(di)) + m));
            DLIB_TEST(equal(m + inv(diagm(d)) , tmp(diagm(di)) + m));


            DLIB_TEST((m + identity_matrix<double>(3) == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>() == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + 2*identity_matrix<double>(3) == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + 2*identity_matrix<double,3>() == m + 2*tmp(identity_matrix<double,3>())));
            DLIB_TEST((m + identity_matrix<double>(3)*2 == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((m + identity_matrix<double,3>()*2 == m + 2*tmp(identity_matrix<double,3>())));

            DLIB_TEST((identity_matrix<double>(3) + m == m + tmp(identity_matrix<double>(3))));
            DLIB_TEST((identity_matrix<double,3>() + m == m + tmp(identity_matrix<double,3>())));
            DLIB_TEST((2*identity_matrix<double>(3) + m == m + 2*tmp(identity_matrix<double>(3))));
            DLIB_TEST((2*identity_matrix<double,3>() + m == m + 2*tmp(identity_matrix<double,3>())));
        }


        {
            matrix<double,3,1> d1, d2;

            d1 = 1,2,3;

            d2 = 2,3,4;

            matrix<double,3,3> ans;
            ans = 2, 0, 0,
                  0, 6, 0,
                  0, 0, 12;

            DLIB_TEST(ans == diagm(d1)*diagm(d2));
        }


        dlib::rand::float_1a rnd;
        for (int i = 0; i < 1; ++i)
        {
            matrix<double> d1 = randm(4,1,rnd);
            matrix<double,5,1> d2 = randm(5,1,rnd);

            matrix<double,4,5> m = randm(4,5,rnd);

            DLIB_TEST_MSG(equal(pointwise_multiply(d1*trans(d2), m) , diagm(d1)*m*diagm(d2)),
                          pointwise_multiply(d1*trans(d2), m) - diagm(d1)*m*diagm(d2)
            );
            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , diagm(d1)*(m*diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , (diagm(d1)*m)*diagm(d2)));

            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , inv(diagm(d1))*m*inv(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , inv(diagm(d1))*(m*inv(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , (inv(diagm(d1))*m)*inv(diagm(d2))));

            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , inv(diagm(d1))*m*(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , inv(diagm(d1))*(m*(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , (inv(diagm(d1))*m)*(diagm(d2))));

            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , (diagm(d1))*m*inv(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , (diagm(d1))*(m*inv(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , ((diagm(d1))*m)*inv(diagm(d2))));
        }
        for (int i = 0; i < 1; ++i)
        {
            matrix<double,4,1> d1 = randm(4,1,rnd);
            matrix<double,5,1> d2 = randm(5,1,rnd);

            matrix<double,4,5> m = randm(4,5,rnd);

            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , diagm(d1)*m*diagm(d2)));
            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , diagm(d1)*(m*diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , (diagm(d1)*m)*diagm(d2)));

            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , inv(diagm(d1))*m*inv(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , inv(diagm(d1))*(m*inv(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , (inv(diagm(d1))*m)*inv(diagm(d2))));

            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , inv(diagm(d1))*m*(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , inv(diagm(d1))*(m*(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , (inv(diagm(d1))*m)*(diagm(d2))));

            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , (diagm(d1))*m*inv(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , (diagm(d1))*(m*inv(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , ((diagm(d1))*m)*inv(diagm(d2))));
        }
        for (int i = 0; i < 1; ++i)
        {
            matrix<double,4,1> d1 = randm(4,1,rnd);
            matrix<double,5,1> d2 = randm(5,1,rnd);

            matrix<double,0,0> m = randm(4,5,rnd);

            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , diagm(d1)*m*diagm(d2)));
            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , diagm(d1)*(m*diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(d1*trans(d2), m) , (diagm(d1)*m)*diagm(d2)));

            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , inv(diagm(d1))*m*inv(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , inv(diagm(d1))*(m*inv(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans(reciprocal(d2)), m) , (inv(diagm(d1))*m)*inv(diagm(d2))));

            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , inv(diagm(d1))*m*(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , inv(diagm(d1))*(m*(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply(reciprocal(d1)*trans((d2)), m) , (inv(diagm(d1))*m)*(diagm(d2))));

            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , (diagm(d1))*m*inv(diagm(d2))));
            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , (diagm(d1))*(m*inv(diagm(d2)))));
            DLIB_TEST(equal(pointwise_multiply((d1)*trans(reciprocal(d2)), m) , ((diagm(d1))*m)*inv(diagm(d2))));
        }



        {
            for (int i = 0; i < 5; ++i)
            {
                matrix<double> m = randm(3,4) + 1;

                DLIB_TEST(equal(1.0/m , reciprocal(m)));
                DLIB_TEST(equal(0.0/m , zeros_matrix<double>(3,4)));
            }
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



