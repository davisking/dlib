// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_BLAS_BINDINGS_
#define DLIB_MATRIx_BLAS_BINDINGS_

#include "matrix_assign.h"

#ifdef DLIB_FOUND_BLAS
#include "cblas.h"
#endif

namespace dlib
{


    namespace blas_bindings 
    {

    // ----------------------------------------------------------------------------------------

        // Here we declare some matrix objects for use in the DLIB_ADD_BLAS_BINDING macro.  These
        // extern declarations don't actually correspond to any real matrix objects.  They are
        // simply here so we can build matrix expressions with the DLIB_ADD_BLAS_BINDING marco.


        typedef memory_manager<char>::kernel_1a mm;
        // Note that the fact that these are double matrices isn't important.  The type
        // that matters is the one that is the first argument of the DLIB_ADD_BLAS_BINDING.
        // That type determines what the type of the elements of the matrices that we
        // are dealing with is.
        extern matrix<double,0,0,mm,row_major_layout> rm;     // general matrix with row major order
        extern matrix<double,0,0,mm,column_major_layout> cm;  // general matrix with column major order
        extern matrix<double,1,0> rv;  // general row vector
        extern matrix<double,0,1> cv;  // general column vector
        extern const double s;



#ifdef DLIB_FOUND_BLAS

        DLIB_ADD_BLAS_BINDING(double, row_major_layout, rm*rm)
        {
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
        } DLIB_END_BLAS_BINDING

        DLIB_ADD_BLAS_BINDING(double, row_major_layout,rm + rm*rm)
        {
            if (&src.lhs != &dest)
            {
                dest = src.lhs;
            }

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.nr());
            const int N = static_cast<int>(src.rhs.nc());
            const int K = static_cast<int>(src.rhs.lhs.nc());
            const double alpha = 1;
            const double* A = &src.rhs.lhs(0,0);
            const int lda = src.rhs.lhs.nc();
            const double* B = &src.rhs.rhs(0,0);
            const int ldb = src.rhs.rhs.nc();

            const double beta = 1;
            double* C = &dest(0,0);
            const int ldc = src.rhs.nc();

            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(double, row_major_layout, trans(rm)*rm)
        {
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
        } DLIB_END_BLAS_BINDING

        DLIB_ADD_BLAS_BINDING(double, row_major_layout, rm + s*trans(rm)*rm)
        {
            if (&src.lhs != &dest)
            {
                dest = src.lhs;
            }

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.m.nr());
            const int N = static_cast<int>(src.rhs.m.nc());
            const int K = static_cast<int>(src.rhs.m.lhs.nc());
            const double alpha = src.rhs.s;
            const double* A = &src.rhs.m.lhs.m(0,0);
            const int lda = src.rhs.m.lhs.m.nc();
            const double* B = &src.rhs.m.rhs(0,0);
            const int ldb = src.rhs.m.rhs.nc();

            const double beta = 1;
            double* C = &dest(0,0);
            const int ldc = dest.nc();

            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        DLIB_ADD_BLAS_BINDING(double, row_major_layout, s*trans(rm)*rm)
        {
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.m.nr());
            const int N = static_cast<int>(src.m.nc());
            const int K = static_cast<int>(src.m.lhs.nc());
            const double alpha = src.s;
            const double* A = &src.m.lhs.m(0,0);
            const int lda = src.m.lhs.m.nc();
            const double* B = &src.m.rhs(0,0);
            const int ldb = src.m.rhs.nc();

            const double beta = 0;
            double* C = &dest(0,0);
            const int ldc = dest.nc();

            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING


        DLIB_ADD_BLAS_BINDING(double, row_major_layout, rm + trans(rm)*rm)
        {
            if (&src.lhs != &dest)
            {
                dest = src.lhs;
            }

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.nr());
            const int N = static_cast<int>(src.rhs.nc());
            const int K = static_cast<int>(src.rhs.lhs.nc());
            const double alpha = 1;
            const double* A = &src.rhs.lhs.m(0,0);
            const int lda = src.rhs.lhs.m.nc();
            const double* B = &src.rhs.rhs(0,0);
            const int ldb = src.rhs.rhs.nc();

            const double beta = 1;
            double* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_dgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING


        // ---------------------------------------------------------------------
        // -------------------------- float overloads --------------------------
        // ---------------------------------------------------------------------

        DLIB_ADD_BLAS_BINDING(float, row_major_layout, rm*rm)
        {
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
        } DLIB_END_BLAS_BINDING

        DLIB_ADD_BLAS_BINDING(float, row_major_layout,rm + rm*rm)
        {
            if (&src.lhs != &dest)
            {
                dest = src.lhs;
            }

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.nr());
            const int N = static_cast<int>(src.rhs.nc());
            const int K = static_cast<int>(src.rhs.lhs.nc());
            const float alpha = 1;
            const float* A = &src.rhs.lhs(0,0);
            const int lda = src.rhs.lhs.nc();
            const float* B = &src.rhs.rhs(0,0);
            const int ldb = src.rhs.rhs.nc();

            const float beta = 1;
            float* C = &dest(0,0);
            const int ldc = src.rhs.nc();

            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(float, row_major_layout, trans(rm)*rm)
        {
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
        } DLIB_END_BLAS_BINDING

        DLIB_ADD_BLAS_BINDING(float, row_major_layout, rm + s*trans(rm)*rm)
        {
            if (&src.lhs != &dest)
            {
                dest = src.lhs;
            }

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.m.nr());
            const int N = static_cast<int>(src.rhs.m.nc());
            const int K = static_cast<int>(src.rhs.m.lhs.nc());
            const float alpha = src.rhs.s;
            const float* A = &src.rhs.m.lhs.m(0,0);
            const int lda = src.rhs.m.lhs.m.nc();
            const float* B = &src.rhs.m.rhs(0,0);
            const int ldb = src.rhs.m.rhs.nc();

            const float beta = 1;
            float* C = &dest(0,0);
            const int ldc = dest.nc();

            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        DLIB_ADD_BLAS_BINDING(float, row_major_layout, s*trans(rm)*rm)
        {
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.m.nr());
            const int N = static_cast<int>(src.m.nc());
            const int K = static_cast<int>(src.m.lhs.nc());
            const float alpha = src.s;
            const float* A = &src.m.lhs.m(0,0);
            const int lda = src.m.lhs.m.nc();
            const float* B = &src.m.rhs(0,0);
            const int ldb = src.m.rhs.nc();

            const float beta = 0;
            float* C = &dest(0,0);
            const int ldc = dest.nc();

            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING


        DLIB_ADD_BLAS_BINDING(float, row_major_layout, rm + trans(rm)*rm)
        {
            if (&src.lhs != &dest)
            {
                dest = src.lhs;
            }

            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.nr());
            const int N = static_cast<int>(src.rhs.nc());
            const int K = static_cast<int>(src.rhs.lhs.nc());
            const float alpha = 1;
            const float* A = &src.rhs.lhs.m(0,0);
            const int lda = src.rhs.lhs.m.nc();
            const float* B = &src.rhs.rhs(0,0);
            const int ldb = src.rhs.rhs.nc();

            const float beta = 1;
            float* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING


#endif // DLIB_FOUND_BLAS

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_BLAS_BINDINGS_

