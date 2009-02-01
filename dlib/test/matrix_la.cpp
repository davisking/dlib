// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
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
#include <dlib/string.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.matrix_la");

    dlib::rand::float_1a rnd;

// ----------------------------------------------------------------------------------------

    template <typename mat_type>
    const matrix<typename mat_type::type> symm(const mat_type& m) { return m*trans(m); }

// ----------------------------------------------------------------------------------------

    template <typename type>
    const matrix<type> randm(long r, long c)
    {
        matrix<type> m(r,c);
        for (long row = 0; row < m.nr(); ++row)
        {
            for (long col = 0; col < m.nc(); ++col)
            {
                m(row,col) = static_cast<type>(rnd.get_random_double()); 
            }
        }

        return m;
    }

    template <typename type, long NR, long NC>
    const matrix<type,NR,NC> randm()
    {
        matrix<type,NR,NC> m;
        for (long row = 0; row < m.nr(); ++row)
        {
            for (long col = 0; col < m.nc(); ++col)
            {
                m(row,col) = static_cast<type>(rnd.get_random_double()); 
            }
        }

        return m;
    }

// ----------------------------------------------------------------------------------------

    template <typename matrix_type>
    void test_qr ( const matrix_type& m)
    {
        typedef typename matrix_type::type type;
        const type eps = 10*max(abs(m))*sqrt(std::numeric_limits<type>::epsilon());
        dlog << LDEBUG << "test_qr():  " << m.nr() << " x " << m.nc() << "  eps: " << eps;
        print_spinner();


        qr_decomposition<matrix_type> test(m);


        DLIB_CASSERT(test.nr() == m.nr(),"");
        DLIB_CASSERT(test.nc() == m.nc(),"");


        type temp;
        DLIB_CASSERT( (temp= max(abs(test.get_q()*test.get_r() - m))) < eps,temp);

        // none of the matrices we should be passing in to test_qr() should be non-full rank.  
        DLIB_CASSERT(test.is_full_rank() == true,"");

        if (m.nr() == m.nc())
        {
            matrix<type> m2;
            matrix<type,0,1> col;

            m2 = identity_matrix<type>(m.nr());
            DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
            m2 = randm<type>(m.nr(),5);
            DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
            m2 = randm<type>(m.nr(),1);
            DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
            col = randm<type>(m.nr(),1);
            DLIB_CASSERT(equal(m*test.solve(col), col,eps),max(abs(m*test.solve(m2)- m2)));
        }
        else
        {
            DLIB_CASSERT(dlib::equal(pinv(m), test.solve(identity_matrix<type>(m.nr())), eps), 
                        max(abs(pinv(m) - test.solve(identity_matrix<type>(m.nr())))) );
        }

        // now make us a non-full rank matrix
        if (m.nc() > 1)
        {
            matrix<type> sm(m);
            set_colm(sm,0) = colm(sm,1);

            qr_decomposition<matrix_type> test2(sm);
            DLIB_CASSERT( (temp= max(abs(test.get_q()*test.get_r() - m))) < eps,temp);

            if (test2.nc() < 100)
            {
                DLIB_CASSERT(test2.is_full_rank() == false,"eps: " << eps);
            }

        }

    }

// ----------------------------------------------------------------------------------------

    template <typename matrix_type>
    void test_lu ( const matrix_type& m)
    {
        typedef typename matrix_type::type type;
        const type eps = 10*max(abs(m))*sqrt(std::numeric_limits<type>::epsilon());
        dlog << LDEBUG << "test_lu():  " << m.nr() << " x " << m.nc() << "  eps: " << eps;
        print_spinner();


        lu_decomposition<matrix_type> test(m);

        DLIB_CASSERT(test.is_square() == (m.nr() == m.nc()), "");

        DLIB_CASSERT(test.nr() == m.nr(),"");
        DLIB_CASSERT(test.nc() == m.nc(),"");

        type temp;
        DLIB_CASSERT( (temp= max(abs(test.get_l()*test.get_u() - rowm(m,test.get_pivot())))) < eps,temp);

        if (test.is_square())
        {
            // none of the matrices we should be passing in to test_lu() should be singular.  
            DLIB_CASSERT (abs(test.det()) > eps/100, "det: " << test.det() );
            dlog << LDEBUG << "big det: " << test.det();

            DLIB_CASSERT(test.is_singular() == false,"");

            matrix<type> m2;
            matrix<type,0,1> col;

            m2 = identity_matrix<type>(m.nr());
            DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
            m2 = randm<type>(m.nr(),5);
            DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
            m2 = randm<type>(m.nr(),1);
            DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
            col = randm<type>(m.nr(),1);
            DLIB_CASSERT(equal(m*test.solve(col), col,eps),max(abs(m*test.solve(m2)- m2)));

            // now make us a singular matrix
            if (m.nr() > 1)
            {
                matrix<type> sm(m);
                set_colm(sm,0) = colm(sm,1);

                lu_decomposition<matrix_type> test2(sm);
                DLIB_CASSERT( (temp= max(abs(test2.get_l()*test2.get_u() - rowm(sm,test2.get_pivot())))) < eps,temp);

                // these checks are only accurate for small matrices
                if (test2.nr() < 100)
                {
                    DLIB_CASSERT(test2.is_singular() == true,"det: " << test2.det());
                    DLIB_CASSERT(abs(test2.det()) < eps,"det: " << test2.det());
                }

            }
        }

    }

// ----------------------------------------------------------------------------------------

    template <typename matrix_type>
    void test_cholesky ( const matrix_type& m)
    {
        typedef typename matrix_type::type type;
        const type eps = 10*max(abs(m))*sqrt(std::numeric_limits<type>::epsilon());
        dlog << LDEBUG << "test_cholesky():  " << m.nr() << " x " << m.nc() << "  eps: " << eps;
        print_spinner();


        cholesky_decomposition<matrix_type> test(m);

        // none of the matrices we should be passing in to test_cholesky() should be non-spd.  
        DLIB_CASSERT(test.is_spd() == true,  "");

        type temp;
        DLIB_CASSERT( (temp= max(abs(test.get_l()*trans(test.get_l()) - m))) < eps,temp);


        matrix<type> m2;
        matrix<type,0,1> col;

        m2 = identity_matrix<type>(m.nr());
        DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
        m2 = randm<type>(m.nr(),5);
        DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
        m2 = randm<type>(m.nr(),1);
        DLIB_CASSERT(equal(m*test.solve(m2), m2,eps),max(abs(m*test.solve(m2)- m2)));
        col = randm<type>(m.nr(),1);
        DLIB_CASSERT(equal(m*test.solve(col), col,eps),max(abs(m*test.solve(m2)- m2)));

        // now make us a non-spd matrix
        if (m.nr() > 1)
        {
            matrix<type> sm(lowerm(m));
            sm(1,1) = 0;

            cholesky_decomposition<matrix_type> test2(sm);
            DLIB_CASSERT(test2.is_spd() == false,  test2.get_l());


            cholesky_decomposition<matrix_type> test3(sm*trans(sm));
            DLIB_CASSERT(test3.is_spd() == false,  test3.get_l());

            sm = sm*trans(sm);
            sm(1,1) = 5;
            sm(1,0) -= 1;
            cholesky_decomposition<matrix_type> test4(sm);
            DLIB_CASSERT(test4.is_spd() == false,  test4.get_l());
        }

    }

// ----------------------------------------------------------------------------------------

    template <typename matrix_type>
    void test_eigenvalue ( const matrix_type& m)
    {
        typedef typename matrix_type::type type;
        const type eps = max(abs(m))*sqrt(std::numeric_limits<type>::epsilon());
        dlog << LDEBUG << "test_eigenvalue():  " << m.nr() << " x " << m.nc() << "  eps: " << eps;
        print_spinner();


        eigenvalue_decomposition<matrix_type> test(m);

        DLIB_CASSERT(test.dim() == m.nr(), "");

        // make sure all the various ways of asking for the eigenvalues are actually returning a
        // consistent set of eigenvalues.
        DLIB_CASSERT(equal(real(test.get_eigenvalues()), test.get_real_eigenvalues(), eps), ""); 
        DLIB_CASSERT(equal(imag(test.get_eigenvalues()), test.get_imag_eigenvalues(), eps), ""); 
        DLIB_CASSERT(equal(real(diag(test.get_d())), test.get_real_eigenvalues(), eps), ""); 
        DLIB_CASSERT(equal(imag(diag(test.get_d())), test.get_imag_eigenvalues(), eps), ""); 

        const matrix<type> V = test.get_pseudo_v();
        const matrix<type> D = test.get_pseudo_d();
        const matrix<complex<type> > CV = test.get_v();
        const matrix<complex<type> > CD = test.get_d();
        const matrix<complex<type> > CM = complex_matrix(m, uniform_matrix<type>(m.nr(),m.nc(),0));

        DLIB_CASSERT(V.nr() == test.dim(),"");
        DLIB_CASSERT(V.nc() == test.dim(),"");
        DLIB_CASSERT(D.nr() == test.dim(),"");
        DLIB_CASSERT(D.nc() == test.dim(),"");

        // CD is a diagonal matrix
        DLIB_CASSERT(diagm(diag(CD)) == CD,"");

        // verify that these things are actually eigenvalues and eigenvectors of m
        DLIB_CASSERT(max(abs(m*V - V*D)) < eps, "");
        DLIB_CASSERT(max(norm(CM*CV - CV*CD)) < eps, "");

        // if m is a symmetric matrix
        if (max(abs(m-trans(m))) < 1e-5)
        {
            dlog << LTRACE << "m is symmetric";
            // there aren't any imaginary eigenvalues 
            DLIB_CASSERT(max(abs(test.get_imag_eigenvalues())) < eps, ""); 
            DLIB_CASSERT(diagm(diag(D)) == D,"");

            // V is orthogonal
            DLIB_CASSERT(equal(V*trans(V), identity_matrix<type>(test.dim()), eps), "");
            DLIB_CASSERT(equal(m , V*D*trans(V), eps), "");
        }
        else
        {
            dlog << LTRACE << "m is NOT symmetric";
            DLIB_CASSERT(equal(m , V*D*inv(V), eps), max(abs(m - V*D*inv(V))));
        }
    }

// ----------------------------------------------------------------------------------------

    void matrix_test_double()
    {

    // -------------------------------

        test_lu(10*randm<double>(1,1));
        test_lu(10*randm<double>(2,2));
        test_lu(10*symm(randm<double>(2,2)));
        test_lu(10*randm<double>(4,4));
        test_lu(10*randm<double>(9,4));
        test_lu(10*randm<double>(3,8));
        test_lu(10*randm<double>(15,15));
        test_lu(2*symm(randm<double>(15,15)));
        test_lu(10*randm<double>(100,100));
        test_lu(10*randm<double>(137,200));
        test_lu(10*randm<double>(200,101));

        test_lu(10*randm<double,1,1>());
        test_lu(10*randm<double,2,2>());
        test_lu(10*randm<double,4,3>());
        test_lu(10*randm<double,4,4>());
        test_lu(10*randm<double,9,4>());
        test_lu(10*randm<double,3,8>());
        test_lu(10*randm<double,15,15>());
        test_lu(10*randm<double,100,100>());
        test_lu(10*randm<double,137,200>());
        test_lu(10*randm<double,200,101>());

    // -------------------------------

        test_cholesky(uniform_matrix<double>(1,1,1) + 10*symm(randm<double>(1,1)));
        test_cholesky(uniform_matrix<double>(2,2,1) + 10*symm(randm<double>(2,2)));
        test_cholesky(uniform_matrix<double>(3,3,1) + 10*symm(randm<double>(3,3)));
        test_cholesky(uniform_matrix<double>(4,4,1) + 10*symm(randm<double>(4,4)));
        test_cholesky(uniform_matrix<double>(15,15,1) + 10*symm(randm<double>(15,15)));
        test_cholesky(uniform_matrix<double>(101,101,1) + 10*symm(randm<double>(101,101)));

    // -------------------------------

        test_qr(10*randm<double>(1,1));
        test_qr(10*randm<double>(2,2));
        test_qr(10*symm(randm<double>(2,2)));
        test_qr(10*randm<double>(4,4));
        test_qr(10*randm<double>(9,4));
        test_qr(10*randm<double>(15,15));
        test_qr(2*symm(randm<double>(15,15)));
        test_qr(10*randm<double>(100,100));
        test_qr(10*randm<double>(237,200));
        test_qr(10*randm<double>(200,101));

        test_qr(10*randm<double,1,1>());
        test_qr(10*randm<double,2,2>());
        test_qr(10*randm<double,4,3>());
        test_qr(10*randm<double,4,4>());
        test_qr(10*randm<double,9,4>());
        test_qr(10*randm<double,15,15>());
        test_qr(10*randm<double,100,100>());

    // -------------------------------

        test_eigenvalue(10*randm<double>(1,1));
        test_eigenvalue(10*randm<double>(2,2));
        test_eigenvalue(10*randm<double>(3,3));
        test_eigenvalue(10*randm<double>(4,4));
        test_eigenvalue(10*randm<double>(15,15));
        test_eigenvalue(10*randm<double>(150,150));

        test_eigenvalue(10*randm<double,1,1>());
        test_eigenvalue(10*randm<double,2,2>());
        test_eigenvalue(10*randm<double,3,3>());

        test_eigenvalue(10*symm(randm<double>(1,1)));
        test_eigenvalue(10*symm(randm<double>(2,2)));
        test_eigenvalue(10*symm(randm<double>(3,3)));
        test_eigenvalue(10*symm(randm<double>(4,4)));
        test_eigenvalue(10*symm(randm<double>(15,15)));
        test_eigenvalue(10*symm(randm<double>(150,150)));

    // -------------------------------
    }

// ----------------------------------------------------------------------------------------

    void matrix_test_float()
    {

    // -------------------------------

        test_lu(3*randm<float>(1,1));
        test_lu(3*randm<float>(2,2));
        test_lu(3*randm<float>(4,4));
        test_lu(3*randm<float>(9,4));
        test_lu(3*randm<float>(3,8));
        test_lu(3*randm<float>(137,200));
        test_lu(3*randm<float>(200,101));

        test_lu(3*randm<float,1,1>());
        test_lu(3*randm<float,2,2>());
        test_lu(3*randm<float,4,3>());
        test_lu(3*randm<float,4,4>());
        test_lu(3*randm<float,9,4>());
        test_lu(3*randm<float,3,8>());
        test_lu(3*randm<float,137,200>());
        test_lu(3*randm<float,200,101>());

    // -------------------------------

        test_cholesky(uniform_matrix<float>(1,1,1) + 2*symm(randm<float>(1,1)));
        test_cholesky(uniform_matrix<float>(2,2,1) + 2*symm(randm<float>(2,2)));
        test_cholesky(uniform_matrix<float>(3,3,1) + 2*symm(randm<float>(3,3)));

    // -------------------------------

        test_qr(3*randm<float>(1,1));
        test_qr(3*randm<float>(2,2));
        test_qr(3*randm<float>(4,4));
        test_qr(3*randm<float>(9,4));
        test_qr(3*randm<float>(237,200));

        test_qr(3*randm<float,1,1>());
        test_qr(3*randm<float,2,2>());
        test_qr(3*randm<float,4,3>());
        test_qr(3*randm<float,4,4>());
        test_qr(3*randm<float,9,4>());

    // -------------------------------


        test_eigenvalue(10*randm<float>(1,1));
        test_eigenvalue(10*randm<float>(2,2));
        test_eigenvalue(10*randm<float>(3,3));
        test_eigenvalue(10*randm<float>(4,4));
        test_eigenvalue(10*randm<float>(15,15));
        test_eigenvalue(10*randm<float>(150,150));

        test_eigenvalue(10*randm<float,1,1>());
        test_eigenvalue(10*randm<float,2,2>());
        test_eigenvalue(10*randm<float,3,3>());

        test_eigenvalue(10*symm(randm<float>(1,1)));
        test_eigenvalue(10*symm(randm<float>(2,2)));
        test_eigenvalue(10*symm(randm<float>(3,3)));
        test_eigenvalue(10*symm(randm<float>(4,4)));
        test_eigenvalue(10*symm(randm<float>(15,15)));
        test_eigenvalue(10*symm(randm<float>(150,150)));
    }

// ----------------------------------------------------------------------------------------

    class matrix_tester : public tester
    {
    public:
        matrix_tester (
        ) :
            tester ("test_matrix_la",
                    "Runs tests on the matrix component.")
        {
            rnd.set_seed(cast_to_string(time(0)));
        }

        void perform_test (
        )
        {
            dlog << LINFO << "seed string: " << rnd.get_seed();

            dlog << LINFO << "begin testing with double";
            matrix_test_double();
            dlog << LINFO << "begin testing with float";
            matrix_test_float();
        }
    } a;

}



