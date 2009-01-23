// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
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

    logger dlog("test.matrix3");


    const double eps_mul = 500000;

    template <typename T, typename U>
    void check_equal (
        const T& a,
        const U& b
    )
    {
        DLIB_CASSERT(a.nr() == b.nr(),"");
        DLIB_CASSERT(a.nc() == b.nc(),"");
        typedef typename T::type type;
        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                type error = std::abs(a(r,c) - b(r,c));
                DLIB_CASSERT(error < std::numeric_limits<type>::epsilon()*eps_mul, "error: " << error <<
                             "    eps: " << std::numeric_limits<type>::epsilon()*eps_mul);
            }
        }
    }

    template <typename T, typename U>
    void c_check_equal (
        const T& a,
        const U& b
    )
    {
        DLIB_CASSERT(a.nr() == b.nr(),"");
        DLIB_CASSERT(a.nc() == b.nc(),"");
        typedef typename T::type type;
        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                typename type::value_type error = std::abs(a(r,c) - b(r,c));
                DLIB_CASSERT(error < std::numeric_limits<typename type::value_type>::epsilon()*eps_mul, "error: " << error <<
                             "    eps: " << std::numeric_limits<typename type::value_type>::epsilon()*eps_mul);
            }
        }
    }

    template <typename T, typename U>
    void assign_no_blas (
        const T& a_,
        const U& b
    )
    {
        T& a = const_cast<T&>(a_);
        DLIB_CASSERT(a.nr() == b.nr(),"");
        DLIB_CASSERT(a.nc() == b.nc(),"");
        for (long r = 0; r < a.nr(); ++r)
        {
            for (long c = 0; c < a.nc(); ++c)
            {
                a(r,c) = b(r,c);
            }
        }
    }

    template <typename type>
    type rnd_num (dlib::rand::float_1a& rnd)
    {
        return static_cast<type>(10*rnd.get_random_double());
    }

    template <typename type>
    void test_blas( long rows, long cols)
    {
        // The tests in this function exercise the BLAS bindings located in the matrix/matrix_blas_bindings.h file.
        // It does this by performing an assignment that is subject to BLAS bindings and comparing the
        // results directly to an unevaluated matrix_exp that should be equal.

        dlib::rand::float_1a rnd;

        matrix<type> a(rows,cols), temp, temp2, temp3;

        for (int i = 0; i < 6; ++i)
        {
            for (long r= 0; r < a.nr(); ++r)
            {
                for (long c = 0; c < a.nc(); ++c)
                {
                    a(r,c) = rnd_num<type>(rnd);
                }
            }
            matrix<type> at;
            at = trans(a);

            matrix<complex<type> > c_a(rows,cols), c_at;
            for (long r= 0; r < a.nr(); ++r)
            {
                for (long c = 0; c < a.nc(); ++c)
                {
                    c_a(r,c) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));
                }
            }
            c_at = trans(c_a);

            matrix<complex<type> > c_temp(cols,cols), c_temp2(cols,cols);
            const complex<type> i(0,1);

            const type one = 1;
            const type two = 1;
            const type num1 = static_cast<type>(3.6);
            const type num2 = static_cast<type>(6.6);
            const type num3 = static_cast<type>(8.6);

            matrix<complex<type>,0,1> c_cv4(cols), c_cv3(rows);
            matrix<complex<type>,1,0> c_rv4(cols), c_rv3(rows);

            matrix<type,0,1> cv4(cols);

            for (long idx = 0; idx < cv4.size(); ++idx)
                cv4(idx) = rnd_num<type>(rnd);

            for (long idx = 0; idx < c_cv4.size(); ++idx)
                c_cv4(idx) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));

            matrix<type,1,0> rv3(rows);

            for (long idx = 0; idx < rv3.size(); ++idx)
                rv3(idx) = rnd_num<type>(rnd);

            for (long idx = 0; idx < c_rv3.size(); ++idx)
                c_rv3(idx) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));

            matrix<type,0,1> cv3(rows);

            for (long idx = 0; idx < cv3.size(); ++idx)
                cv3(idx) = rnd_num<type>(rnd);

            for (long idx = 0; idx < c_cv3.size(); ++idx)
                c_cv3(idx) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));

            matrix<type,1,0> rv4(cols);
            for (long idx = 0; idx < rv4.size(); ++idx)
                rv4(idx) = rnd_num<type>(rnd);

            for (long idx = 0; idx < c_rv4.size(); ++idx)
                c_rv4(idx) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));



            // GEMM tests
            dlog << LTRACE << "1.1";
            check_equal(tmp(at*a),   at*a);
            check_equal(tmp(trans(at*a)),   trans(at*a));
            check_equal(tmp(2.4*trans(4*trans(at*a) + at*3*a)),   2.4*trans(4*trans(at*a) + at*3*a));
            dlog << LTRACE << "1.2";
            check_equal(tmp(trans(a)*a),   trans(a)*a);
            check_equal(tmp(trans(trans(a)*a)),   trans(trans(a)*a));
            dlog << LTRACE << "1.3";
            check_equal(tmp(at*trans(at)),   at*trans(at));
            check_equal(tmp(trans(at*trans(at))),   trans(at*trans(at)));
            dlog << LTRACE << "1.4";
            check_equal(tmp(trans(at)*trans(a)),   a*at);
            check_equal(tmp(trans(trans(at)*trans(a))),   trans(a*at));
            dlog << LTRACE << "1.5";

            print_spinner();
            c_check_equal(tmp(conj(trans(c_a))*c_a),   trans(conj(c_a))*c_a);
            c_check_equal(tmp(trans(conj(trans(c_a))*c_a)),   trans(trans(conj(c_a))*c_a));
            dlog << LTRACE << "1.6";
            c_check_equal(tmp(c_at*trans(conj(c_at))),   c_at*conj(trans(c_at)));
            c_check_equal(tmp(trans(c_at*trans(conj(c_at)))),   trans(c_at*conj(trans(c_at))));
            dlog << LTRACE << "1.7";
            c_check_equal(tmp(conj(trans(c_at))*trans(conj(c_a))),  conj(trans(c_at))*trans(conj(c_a)));
            c_check_equal(tmp(trans(conj(trans(c_at))*trans(conj(c_a)))), trans(conj(trans(c_at))*trans(conj(c_a))));
            dlog << LTRACE << "1.8";

            check_equal(tmp(a*trans(rowm(a,1))) ,  a*trans(rowm(a,1)));
            check_equal(tmp(a*colm(at,1)) ,  a*colm(at,1));
            check_equal(tmp(subm(a,1,1,2,2)*subm(a,1,2,2,2)), subm(a,1,1,2,2)*subm(a,1,2,2,2));

            dlog << LTRACE << "1.9";
            check_equal(tmp(trans(a*trans(rowm(a,1)))) ,  trans(a*trans(rowm(a,1))));
            dlog << LTRACE << "1.10";
            check_equal(tmp(trans(a*colm(at,1))) ,  trans(a*colm(at,1)));
            dlog << LTRACE << "1.11";
            check_equal(tmp(trans(subm(a,1,1,2,2)*subm(a,1,2,2,2))), trans(subm(a,1,1,2,2)*subm(a,1,2,2,2)));
            dlog << LTRACE << "1.12";

            {
                temp = at*a;
                temp2 = temp;

                temp += 3.5*at*a;
                assign_no_blas(temp2, temp2 + 3.5*at*a);
                check_equal(temp, temp2);

                temp -= at*3.5*a;
                assign_no_blas(temp2, temp2 - at*3.5*a);
                check_equal(temp, temp2);

                temp = temp + 4*at*a;
                assign_no_blas(temp2, temp2 + 4*at*a);
                check_equal(temp, temp2);

                temp = temp - 2.4*at*a;
                assign_no_blas(temp2, temp2 - 2.4*at*a);
                check_equal(temp, temp2);
            }
            dlog << LTRACE << "1.13";
            {
                temp = trans(at*a);
                temp2 = temp;
                temp3 = temp;

                dlog << LTRACE << "1.14";
                temp += trans(3.5*at*a);
                assign_no_blas(temp2, temp2 + trans(3.5*at*a));
                check_equal(temp, temp2);

                dlog << LTRACE << "1.15";
                temp -= trans(at*3.5*a);
                assign_no_blas(temp2, temp2 - trans(at*3.5*a));
                check_equal(temp, temp2);

                dlog << LTRACE << "1.16";
                temp = trans(temp + 4*at*a);
                assign_no_blas(temp3, trans(temp2 + 4*at*a));
                check_equal(temp, temp3);

                temp2 = temp;
                dlog << LTRACE << "1.17";
                temp = trans(temp - 2.4*at*a);
                assign_no_blas(temp3, trans(temp2 - 2.4*at*a));
                check_equal(temp, temp3);
            }

            dlog << LTRACE << "1.18";

            // GEMV tests
            check_equal(tmp(a*cv4),  a*cv4);
            check_equal(tmp(trans(a*cv4)),  trans(a*cv4));
            check_equal(tmp(rv3*a),  rv3*a);
            check_equal(tmp(trans(cv4)*at),  trans(cv4)*at);
            check_equal(tmp(a*trans(rv4)),  a*trans(rv4));
            check_equal(tmp(trans(a*trans(rv4))),  trans(a*trans(rv4)));

            check_equal(tmp(trans(a)*cv3),  trans(a)*cv3);
            check_equal(tmp(rv4*trans(a)),  rv4*trans(a));
            check_equal(tmp(trans(cv3)*trans(at)),  trans(cv3)*trans(at));
            check_equal(tmp(trans(cv3)*a),  trans(cv3)*a);
            check_equal(tmp(trans(a)*trans(rv3)),  trans(a)*trans(rv3));


            c_check_equal(tmp(trans(conj(c_a))*c_cv3),  trans(conj(c_a))*c_cv3);
            c_check_equal(tmp(c_rv4*trans(conj(c_a))),  c_rv4*trans(conj(c_a)));
            c_check_equal(tmp(trans(c_cv3)*trans(conj(c_at))),  trans(c_cv3)*trans(conj(c_at)));
            c_check_equal(tmp(conj(trans(c_a))*trans(c_rv3)),  trans(conj(c_a))*trans(c_rv3));
            c_check_equal(tmp(c_rv4*conj(c_at)),  c_rv4*conj(c_at));
            c_check_equal(tmp(trans(c_cv4)*conj(c_at)),  trans(c_cv4)*conj(c_at));



            dlog << LTRACE << "6";
            temp = a*at;
            check_equal(temp, a*at);
            temp = temp + a*at + trans(at)*at + trans(at)*sin(at);
            check_equal(temp, a*at + a*at+ trans(at)*at + trans(at)*sin(at));

            dlog << LTRACE << "6.1";
            temp = a*at;
            check_equal(temp, a*at);
            temp = a*at + temp;
            check_equal(temp, a*at + a*at);

            print_spinner();
            dlog << LTRACE << "6.2";
            temp = a*at;
            check_equal(temp, a*at);
            dlog << LTRACE << "6.2.3";
            temp = temp - a*at;
            dlog << LTRACE << "6.2.4";
            check_equal(temp, a*at-a*at);

            dlog << LTRACE << "6.3";
            temp = a*at;
            dlog << LTRACE << "6.3.5";
            check_equal(temp, a*at);
            dlog << LTRACE << "6.3.6";
            temp = a*at - temp;
            dlog << LTRACE << "6.4";
            check_equal(temp, a*at-a*at);



            const long d = min(rows,cols);
            rectangle rect(1,1,d,d);
            temp.set_size(max(rows,cols)+4,max(rows,cols)+4);
            set_all_elements(temp,4);
            temp2 = temp;

            dlog << LTRACE << "7";
            set_subm(temp,rect) = a*at;
            assign_no_blas( set_subm(temp2,rect) , a*at);
            check_equal(temp, temp2);

            temp = a;
            temp2 = a;

            set_colm(temp,1) = a*cv4;
            assign_no_blas( set_colm(temp2,1) , a*cv4);
            check_equal(temp, temp2);

            set_rowm(temp,1) = rv3*a;
            assign_no_blas( set_rowm(temp2,1) , rv3*a);
            check_equal(temp, temp2);


            // Test BLAS GER
            {
                temp.set_size(cols,cols);
                set_all_elements(temp,3);
                temp2 = temp;


                dlog << LTRACE << "8";
                temp += cv4*rv4;
                assign_no_blas(temp2, temp2 + cv4*rv4);
                check_equal(temp, temp2);

                dlog << LTRACE << "8.3";
                temp = temp + cv4*rv4;
                assign_no_blas(temp2, temp2 + cv4*rv4);
                check_equal(temp, temp2);
                dlog << LTRACE << "8.9";
            }
            {
                temp.set_size(cols,cols);
                set_all_elements(temp,3);
                temp2 = temp;
                temp3 = 0;

                dlog << LTRACE << "8.10";

                temp += trans(cv4*rv4);
                assign_no_blas(temp3, temp2 + trans(cv4*rv4));
                check_equal(temp, temp3);
                temp3 = 0;

                dlog << LTRACE << "8.11";
                temp2 = temp;
                temp = trans(temp + cv4*rv4);
                assign_no_blas(temp3, trans(temp2 + cv4*rv4));
                check_equal(temp, temp3);
                dlog << LTRACE << "8.12";
            }
            {
                matrix<complex<type> > temp, temp2, temp3;
                matrix<complex<type>,0,1 > cv4;
                matrix<complex<type>,1,0 > rv4;
                cv4.set_size(cols);
                rv4.set_size(cols);
                temp.set_size(cols,cols);
                set_all_elements(temp,complex<type>(3,5));
                temp(cols-1, cols-4) = 9;
                temp2 = temp;
                temp3.set_size(cols,cols);
                temp3 = 0;

                for (long i = 0; i < rv4.size(); ++i)
                {
                    rv4(i) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));
                    cv4(i) = complex<type>(rnd_num<type>(rnd),rnd_num<type>(rnd));
                }

                dlog << LTRACE << "8.13";

                temp += trans(cv4*rv4);
                assign_no_blas(temp3, temp2 + trans(cv4*rv4));
                c_check_equal(temp, temp3);
                temp3 = 0;

                dlog << LTRACE << "8.14";
                temp2 = temp;
                temp = trans(temp + cv4*rv4);
                assign_no_blas(temp3, trans(temp2 + cv4*rv4));
                c_check_equal(temp, temp3);
                dlog << LTRACE << "8.15";
            }




            set_all_elements(c_temp, one + num1*i);
            c_temp2 = c_temp;
            set_all_elements(c_rv4, one + num2*i);
            set_all_elements(c_cv4, two + num3*i);


            dlog << LTRACE << "9";
            c_temp += c_cv4*c_rv4;
            assign_no_blas(c_temp2, c_temp2 + c_cv4*c_rv4);
            c_check_equal(c_temp, c_temp2);
            dlog << LTRACE << "9.1";
            c_temp += c_cv4*conj(c_rv4);
            assign_no_blas(c_temp2, c_temp2 + c_cv4*conj(c_rv4));
            c_check_equal(c_temp, c_temp2);
            dlog << LTRACE << "9.2";
            c_temp = c_cv4*conj(c_rv4) + c_temp;
            assign_no_blas(c_temp2, c_temp2 + c_cv4*conj(c_rv4));
            c_check_equal(c_temp, c_temp2);
            dlog << LTRACE << "9.3";
            c_temp = trans(c_rv4)*trans(conj(c_cv4)) + c_temp;
            assign_no_blas(c_temp2, c_temp2 + trans(c_rv4)*trans(conj(c_cv4)));
            c_check_equal(c_temp, c_temp2);

            dlog << LTRACE << "10";


            print_spinner();

            // Test DOT
            check_equal( tmp(rv4*cv4), rv4*cv4);
            check_equal( tmp(trans(rv4*cv4)), trans(rv4*cv4));
            check_equal( tmp(trans(cv4)*trans(rv4)), trans(cv4)*trans(rv4));
            check_equal( tmp(rv4*3.9*cv4), rv4*3.9*cv4);
            check_equal( tmp(trans(cv4)*3.9*trans(rv4)), trans(cv4)*3.9*trans(rv4));
            check_equal( tmp(rv4*cv4*3.9), rv4*3.9*cv4);
            check_equal( tmp(trans(cv4)*trans(rv4)*3.9), trans(cv4)*3.9*trans(rv4));

            temp.set_size(1,1);
            temp = 4;
            check_equal( tmp(temp + rv4*cv4), temp + rv4*cv4);
            check_equal( tmp(temp + trans(cv4)*trans(rv4)), temp + trans(cv4)*trans(rv4));

            dlog << LTRACE << "11";



            c_check_equal( tmp(conj(c_rv4)*c_cv4), conj(c_rv4)*c_cv4);
            c_check_equal( tmp(conj(trans(c_cv4))*trans(c_rv4)), trans(conj(c_cv4))*trans(c_rv4));

            c_check_equal( tmp(conj(c_rv4)*i*c_cv4), conj(c_rv4)*i*c_cv4);
            c_check_equal( tmp(conj(trans(c_cv4))*i*trans(c_rv4)), trans(conj(c_cv4))*i*trans(c_rv4));

            c_temp.set_size(1,1);
            c_temp = 4;
            c_check_equal( tmp(c_temp + conj(c_rv4)*c_cv4), c_temp + conj(c_rv4)*c_cv4);
            c_check_equal( tmp(c_temp + trans(conj(c_cv4))*trans(c_rv4)), c_temp + trans(conj(c_cv4))*trans(c_rv4));

            DLIB_CASSERT(abs((static_cast<complex<type> >(c_rv4*c_cv4) + i) - ((c_rv4*c_cv4)(0) + i)) < std::numeric_limits<type>::epsilon()*eps_mul ,"");
            DLIB_CASSERT(abs((rv4*cv4 + 1.0) - ((rv4*cv4)(0) + 1.0)) < std::numeric_limits<type>::epsilon()*eps_mul,"");

        }
    }


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
            matrix<long> m1(2,2), m2(2,2);

            m1 = 1, 2,
            3, 4;

            m2 = 4, 5,
            6, 7;


            DLIB_CASSERT(subm(tensor_product(m1,m2),range(0,1), range(0,1)) == 1*m2,"");
            DLIB_CASSERT(subm(tensor_product(m1,m2),range(0,1), range(2,3)) == 2*m2,"");
            DLIB_CASSERT(subm(tensor_product(m1,m2),range(2,3), range(0,1)) == 3*m2,"");
            DLIB_CASSERT(subm(tensor_product(m1,m2),range(2,3), range(2,3)) == 4*m2,"");
        }

        {
            print_spinner();
            dlog << LTRACE << "testing blas stuff";
            dlog << LTRACE << " \nsmall double";
            test_blas<double>(3,4);
            print_spinner();
            dlog << LTRACE << " \nsmall float";
            test_blas<float>(3,4);
            print_spinner();
            dlog << LTRACE << " \nbig double";
            test_blas<double>(120,131);
            print_spinner();
            dlog << LTRACE << " \nbig float";
            test_blas<float>(120,131);
            print_spinner();
            dlog << LTRACE << "testing done";
        }


        {
            matrix<long> m(3,4), ml(3,4), mu(3,4);
            m = 1,2,3,4,
                4,5,6,7,
                7,8,9,0;

            ml = 1,0,0,0,
                 4,5,0,0,
                 7,8,9,0;

            mu = 1,2,3,4,
                 0,5,6,7,
                 0,0,9,0;


            DLIB_CASSERT(lowerm(m) == ml,"");
            DLIB_CASSERT(upperm(m) == mu,"");

            ml = 3,0,0,0,
                 4,3,0,0,
                 7,8,3,0;

            mu = 4,2,3,4,
                 0,4,6,7,
                 0,0,4,0;

            DLIB_CASSERT(lowerm(m,3) == ml,"");
            DLIB_CASSERT(upperm(m,4) == mu,"");

        }

        {
            matrix<long> m(3,4), row(1,3), col(2,1);
            m = 1,2,3,4,
                4,5,6,7,
                7,8,9,0;

            row = 4,5,6;
            col = 3,6;

            DLIB_CASSERT(rowm(m, 1, 3) == row,"");
            DLIB_CASSERT(colm(m, 2, 2) == col,"");

        }



    }






    class matrix_tester : public tester
    {
    public:
        matrix_tester (
        ) :
            tester ("test_matrix3",
                    "Runs tests on the matrix component.")
        {}

        void perform_test (
        )
        {
            matrix_test();
        }
    } a;

}


