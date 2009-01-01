// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_BLAS_BINDINGS_
#define DLIB_MATRIx_BLAS_BINDINGS_

#ifndef DLIB_USE_BLAS
#error "DLIB_USE_BLAS should be defined if you want to use the BLAS bindings"
#endif

#include "matrix_assign.h"

#include "cblas.h"

#include <iostream>

namespace dlib
{


    namespace blas_bindings 
    {

    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------

        inline void cblas_gemm( const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                                const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                                const int K, const float alpha, const float *A,
                                const int lda, const float *B, const int ldb,
                                const float beta, float *C, const int ldc)
        {
            cblas_sgemm( Order, TransA, TransB,  M,  N,
                          K,  alpha, A, lda, B,  ldb, beta, C,  ldc);
        }

        inline void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                         const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                         const int K, const double alpha, const double *A,
                         const int lda, const double *B, const int ldb,
                         const double beta, double *C, const int ldc)
        {
            cblas_dgemm( Order, TransA, TransB,  M,  N,
                          K,  alpha, A, lda, B,  ldb, beta, C,  ldc);
        }

        inline void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                         const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                         const int K, const std::complex<float> *alpha, const std::complex<float> *A,
                         const int lda, const std::complex<float> *B, const int ldb,
                         const std::complex<float> *beta, std::complex<float> *C, const int ldc)
        {
            cblas_cgemm( Order, TransA, TransB,  M,  N,
                          K,  alpha, A, lda, B,  ldb, beta, C,  ldc);
        }

        inline void cblas_gemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                         const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                         const int K, const std::complex<double> *alpha, const std::complex<double> *A,
                         const int lda, const std::complex<double> *B, const int ldb,
                         const std::complex<double> *beta, std::complex<double> *C, const int ldc)
        {
            cblas_zgemm( Order, TransA, TransB,  M,  N,
                          K,  alpha, A, lda, B,  ldb, beta, C,  ldc);
        }

    // ----------------------------------------------------------------------------------------

        inline void cblas_gemv(const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const float alpha, const float *A, const int lda,
                        const float *X, const int incX, const float beta,
                        float *Y, const int incY)
        {
            cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        }

        inline void cblas_gemv(const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const double alpha, const double *A, const int lda,
                        const double *X, const int incX, const double beta,
                        double *Y, const int incY)
        {
            cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        }

        inline void cblas_gemv(const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const std::complex<float> *alpha, const std::complex<float> *A, const int lda,
                        const std::complex<float> *X, const int incX, const std::complex<float> *beta,
                        std::complex<float> *Y, const int incY)
        {
            cblas_cgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        }

        inline void cblas_gemv(const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                        const std::complex<double> *alpha, const std::complex<double> *A, const int lda,
                        const std::complex<double> *X, const int incX, const std::complex<double> *beta,
                        std::complex<double> *Y, const int incY)
        {
            cblas_zgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        }

    // ----------------------------------------------------------------------------------------

        inline void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
                        const std::complex<float> *alpha, const std::complex<float> *X, const int incX,
                        const std::complex<float> *Y, const int incY, std::complex<float> *A, const int lda)
        {
            cblas_cgeru (order,  M, N, alpha, X, incX, Y, incY, A, lda);
        }

        inline void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
                        const std::complex<double> *alpha, const std::complex<double> *X, const int incX,
                        const std::complex<double> *Y, const int incY, std::complex<double> *A, const int lda)
        {
            cblas_zgeru (order,  M, N, alpha, X, incX, Y, incY, A, lda);
        }

        inline void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
                        const float alpha, const float *X, const int incX,
                        const float *Y, const int incY, float *A, const int lda)
        {
            cblas_sger (order,  M, N, alpha, X, incX, Y, incY, A, lda);
        }

        inline void cblas_ger(const enum CBLAS_ORDER order, const int M, const int N,
                        const double alpha, const double *X, const int incX,
                        const double *Y, const int incY, double *A, const int lda)
        {
            cblas_dger (order,  M, N, alpha, X, incX, Y, incY, A, lda);
        }

    // ----------------------------------------------------------------------------------------

        inline void cblas_gerc(const enum CBLAS_ORDER order, const int M, const int N,
                        const std::complex<float> *alpha, const std::complex<float> *X, const int incX,
                        const std::complex<float> *Y, const int incY, std::complex<float> *A, const int lda)
        {
            cblas_cgerc (order,  M, N, alpha, X, incX, Y, incY, A, lda);
        }

        inline void cblas_gerc(const enum CBLAS_ORDER order, const int M, const int N,
                        const std::complex<double> *alpha, const std::complex<double> *X, const int incX,
                        const std::complex<double> *Y, const int incY, std::complex<double> *A, const int lda)
        {
            cblas_zgerc (order,  M, N, alpha, X, incX, Y, incY, A, lda);
        }

    // ----------------------------------------------------------------------------------------

        inline float cblas_dot(const int N, const float  *X, const int incX,
                        const float  *Y, const int incY)
        {
            return cblas_sdot(N, X, incX, Y, incY);
        }

        inline double cblas_dot(const int N, const double *X, const int incX,
                        const double *Y, const int incY)
        {
            return cblas_ddot(N, X, incX, Y, incY);
        }

        inline std::complex<float> cblas_dot(const int N, const std::complex<float> *X, const int incX,
                            const std::complex<float> *Y, const int incY)
        {
            std::complex<float> result;
            cblas_cdotu_sub(N, X, incX, Y, incY, &result);
            return result;
        }

        inline std::complex<double> cblas_dot(const int N, const std::complex<double> *X, const int incX,
                            const std::complex<double> *Y, const int incY)
        {
            std::complex<double> result;
            cblas_zdotu_sub(N, X, incX, Y, incY, &result);
            return result;
        }

    // ----------------------------------------------------------------------------------------

        inline std::complex<float> cblas_dotc(const int N, const std::complex<float> *X, const int incX,
                            const std::complex<float> *Y, const int incY)
        {
            std::complex<float> result;
            cblas_cdotc_sub(N, X, incX, Y, incY, &result);
            return result;
        }

        inline std::complex<double> cblas_dotc(const int N, const std::complex<double> *X, const int incX,
                            const std::complex<double> *Y, const int incY)
        {
            std::complex<double> result;
            cblas_zdotc_sub(N, X, incX, Y, incY, &result);
            return result;
        }

    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------

        // Here we declare some matrix objects for use in the DLIB_ADD_BLAS_BINDING macro.  These
        // extern declarations don't actually correspond to any real matrix objects.  They are
        // simply here so we can build matrix expressions with the DLIB_ADD_BLAS_BINDING marco.


        typedef memory_manager<char>::kernel_1a mm;
        // Note that the fact that these are double matrices isn't important, it is just a placeholder in this case.  
        extern matrix<double,0,0,mm,row_major_layout> rm;     // general matrix with row major order
        extern matrix<double,0,0,mm,column_major_layout> cm;  // general matrix with column major order
        extern matrix<double,1,0> rv;  // general row vector
        extern matrix<double,0,1> cv;  // general column vector
        extern const double s;

    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    //                       GEMM overloads
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, rm*rm)
        {
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const T* A = &src.lhs(0,0);
            const int lda = src.lhs.nc();
            const T* B = &src.rhs(0,0);
            const int ldb = src.rhs.nc();

            const T beta = static_cast<T>(add_to?1:0);
            T* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, trans(rm)*rm)
        {
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasNoTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const T* A = &src.lhs.m(0,0);
            const int lda = src.lhs.m.nc();
            const T* B = &src.rhs(0,0);
            const int ldb = src.rhs.nc();

            const T beta = static_cast<T>(add_to?1:0);
            T* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, rm*trans(rm))
        {
            //cout << "BLAS" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const CBLAS_TRANSPOSE TransB = CblasTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const T* A = &src.lhs(0,0);
            const int lda = src.lhs.nc();
            const T* B = &src.rhs.m(0,0);
            const int ldb = src.rhs.m.nc();

            const T beta = static_cast<T>(add_to?1:0);
            T* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, trans(rm)*trans(rm))
        {
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const CBLAS_TRANSPOSE TransB = CblasTrans;
            const int M = static_cast<int>(src.nr());
            const int N = static_cast<int>(src.nc());
            const int K = static_cast<int>(src.lhs.nc());
            const T* A = &src.lhs.m(0,0);
            const int lda = src.lhs.m.nc();
            const T* B = &src.rhs.m(0,0);
            const int ldb = src.rhs.m.nc();

            const T beta = static_cast<T>(add_to?1:0);
            T* C = &dest(0,0);
            const int ldc = src.nc();

            cblas_gemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        } DLIB_END_BLAS_BINDING

    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    //                       GEMV overloads
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, rm*cv)
        {
            //cout << "BLAS: rm*cv" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const int M = static_cast<int>(src.lhs.nr());
            const int N = static_cast<int>(src.lhs.nc());
            const T* A = &src.lhs(0,0);
            const int lda = src.lhs.nc();
            const T* X = &src.rhs(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, rv*rm)
        {
            // Note that rv*rm is the same as trans(rm)*trans(rv)

            //cout << "BLAS: rv*rm" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const int M = static_cast<int>(src.rhs.nr());
            const int N = static_cast<int>(src.rhs.nc());
            const T* A = &src.rhs(0,0);
            const int lda = src.rhs.nc();
            const T* X = &src.lhs(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, trans(cv)*rm)
        {
            // Note that trans(cv)*rm is the same as trans(rm)*cv

            //cout << "BLAS: trans(cv)*rm" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const int M = static_cast<int>(src.rhs.nr());
            const int N = static_cast<int>(src.rhs.nc());
            const T* A = &src.rhs(0,0);
            const int lda = src.rhs.nc();
            const T* X = &src.lhs.m(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, rm*trans(rv))
        {
            //cout << "BLAS: rm*trans(rv)" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const int M = static_cast<int>(src.lhs.nr());
            const int N = static_cast<int>(src.lhs.nc());
            const T* A = &src.lhs(0,0);
            const int lda = src.lhs.nc();
            const T* X = &src.rhs.m(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------
        // --------------------------------------
        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, trans(rm)*cv)
        {
            //cout << "BLAS: trans(rm)*cv" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const int M = static_cast<int>(src.lhs.m.nr());
            const int N = static_cast<int>(src.lhs.m.nc());
            const T* A = &src.lhs.m(0,0);
            const int lda = src.lhs.m.nc();
            const T* X = &src.rhs(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, rv*trans(rm))
        {
            // Note that rv*trans(rm) is the same as rm*trans(rv)

            //cout << "BLAS: rv*trans(rm)" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.m.nr());
            const int N = static_cast<int>(src.rhs.m.nc());
            const T* A = &src.rhs.m(0,0);
            const int lda = src.rhs.m.nc();
            const T* X = &src.lhs(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, trans(cv)*trans(rm))
        {
            // Note that trans(cv)*trans(rm) is the same as rm*cv

            //cout << "BLAS: trans(cv)*trans(rm)" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasNoTrans;
            const int M = static_cast<int>(src.rhs.m.nr());
            const int N = static_cast<int>(src.rhs.m.nc());
            const T* A = &src.rhs.m(0,0);
            const int lda = src.rhs.m.nc();
            const T* X = &src.lhs.m(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

        DLIB_ADD_BLAS_BINDING(row_major_layout, trans(rm)*trans(rv))
        {
            //cout << "BLAS: trans(rm)*trans(rv)" << endl;
            const CBLAS_ORDER Order = CblasRowMajor;
            const CBLAS_TRANSPOSE TransA = CblasTrans;
            const int M = static_cast<int>(src.lhs.m.nr());
            const int N = static_cast<int>(src.lhs.m.nc());
            const T* A = &src.lhs.m(0,0);
            const int lda = src.lhs.m.nc();
            const T* X = &src.rhs.m(0,0);
            const int incX = 1;

            const T beta = static_cast<T>(add_to?1:0);
            T* Y = &dest(0,0);
            const int incY = 1;

            cblas_gemv(Order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
        } DLIB_END_BLAS_BINDING

        // --------------------------------------

    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    //                       GER overloads
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------


    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    //                       GERC overloads
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------


    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    //                       DOT overloads
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------


    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    //                       DOTC overloads
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------


    // ----------------------------------------------------------------------------------------


    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_BLAS_BINDINGS_

