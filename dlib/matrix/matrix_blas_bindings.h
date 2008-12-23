// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_BLAS_BINDINGS_
#define DLIB_MATRIx_BLAS_BINDINGS_

#include "matrix_assign.h"

#ifdef DLIB_FOUND_BLAS
#include "mkl_cblas.h"
#endif

namespace dlib
{


    namespace blas_bindings 
    {

    // ----------------------------------------------------------------------------------------

        typedef memory_manager<char>::kernel_1a mm;
        extern matrix<double,0,0,mm,row_major_layout> dm;
        extern matrix<float,0,0,mm,row_major_layout>  sm;

        extern matrix<double,1,0,mm,row_major_layout> drv;
        extern matrix<double,0,1,mm,row_major_layout> dcv;

        extern matrix<float,1,0,mm,row_major_layout>  srv;
        extern matrix<float,0,1,mm,row_major_layout>  scv;

        using namespace std;

#ifdef DLIB_FOUND_BLAS

        DLIB_ADD_BLAS_BINDING(double, row_major_layout, dm*dm)
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const double alpha = 1;
            const double* A = &src.lhs(0,0);
            const int lda = src.lhs.nc();
            const double* B = &src.rhs(0,0);
            const int ldb = src.rhs.nc();

            const double beta = 0;
            double* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }};


        DLIB_ADD_BLAS_BINDING(double, row_major_layout, trans(dm)*dm)
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const double alpha = 1;
            const double* A = &src.lhs.m(0,0);
            const int lda = src.lhs.m.nc();
            const double* B = &src.rhs(0,0);
            const int ldb = src.rhs.nc();

            const double beta = 0;
            double* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }};


// double overloads

        DLIB_ADD_BLAS_BINDING(float, row_major_layout, sm*sm)
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const float alpha = 1;
            const float* A = &src.lhs(0,0);
            const int lda = src.lhs.nc();
            const float* B = &src.rhs(0,0);
            const int ldb = src.rhs.nc();

            const float beta = 0;
            float* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }};


        DLIB_ADD_BLAS_BINDING(float, row_major_layout, trans(sm)*sm)
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const float alpha = 1;
            const float* A = &src.lhs.m(0,0);
            const int lda = src.lhs.m.nc();
            const float* B = &src.rhs(0,0);
            const int ldb = src.rhs.nc();

            const float beta = 0;
            float* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        }};


#endif // DLIB_FOUND_BLAS

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_BLAS_BINDINGS_

