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

    logger dlog("test.matrix_eig");

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

        matrix<type> eig1 ( real_eigenvalues(m));
        matrix<type> eig2 ( test.get_real_eigenvalues());
        sort(&eig1(0), &eig1(0) + eig1.size());
        sort(&eig2(0), &eig2(0) + eig2.size());
        DLIB_CASSERT(max(abs(eig1 - eig2)) < eps, "");

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

    }

// ----------------------------------------------------------------------------------------

    void matrix_test_float()
    {

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
            tester ("test_matrix_eig",
                    "Runs tests on the matrix eigen decomp component.")
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



