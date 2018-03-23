#ifndef CBLAS_H
#define CBLAS_H
#include <stddef.h>
#include <stdint.h>

/*
 * Enumerated and derived types
 */
#define CBLAS_INDEX size_t  /* this may vary between platforms */

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CBLAS_INT_TYPE
#define CBLAS_INT_TYPE int 
#endif

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS functions (complex are recast as routines)
 * ===========================================================================
 */
float  cblas_sdsdot(const CBLAS_INT_TYPE N, const float alpha, const float *X,
                    const CBLAS_INT_TYPE incX, const float *Y, const CBLAS_INT_TYPE incY);
double cblas_dsdot(const CBLAS_INT_TYPE N, const float *X, const CBLAS_INT_TYPE incX, const float *Y,
                   const CBLAS_INT_TYPE incY);
float  cblas_sdot(const CBLAS_INT_TYPE N, const float  *X, const CBLAS_INT_TYPE incX,
                  const float  *Y, const CBLAS_INT_TYPE incY);
double cblas_ddot(const CBLAS_INT_TYPE N, const double *X, const CBLAS_INT_TYPE incX,
                  const double *Y, const CBLAS_INT_TYPE incY);

/*
 * Functions having prefixes Z and C only
 */
void   cblas_cdotu_sub(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX,
                       const void *Y, const CBLAS_INT_TYPE incY, void *dotu);
void   cblas_cdotc_sub(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX,
                       const void *Y, const CBLAS_INT_TYPE incY, void *dotc);

void   cblas_zdotu_sub(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX,
                       const void *Y, const CBLAS_INT_TYPE incY, void *dotu);
void   cblas_zdotc_sub(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX,
                       const void *Y, const CBLAS_INT_TYPE incY, void *dotc);


/*
 * Functions having prefixes S D SC DZ
 */
float  cblas_snrm2(const CBLAS_INT_TYPE N, const float *X, const CBLAS_INT_TYPE incX);
float  cblas_sasum(const CBLAS_INT_TYPE N, const float *X, const CBLAS_INT_TYPE incX);

double cblas_dnrm2(const CBLAS_INT_TYPE N, const double *X, const CBLAS_INT_TYPE incX);
double cblas_dasum(const CBLAS_INT_TYPE N, const double *X, const CBLAS_INT_TYPE incX);

float  cblas_scnrm2(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX);
float  cblas_scasum(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX);

double cblas_dznrm2(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX);
double cblas_dzasum(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX);


/*
 * Functions having standard 4 prefixes (S D C Z)
 */
CBLAS_INDEX cblas_isamax(const CBLAS_INT_TYPE N, const float  *X, const CBLAS_INT_TYPE incX);
CBLAS_INDEX cblas_idamax(const CBLAS_INT_TYPE N, const double *X, const CBLAS_INT_TYPE incX);
CBLAS_INDEX cblas_icamax(const CBLAS_INT_TYPE N, const void   *X, const CBLAS_INT_TYPE incX);
CBLAS_INDEX cblas_izamax(const CBLAS_INT_TYPE N, const void   *X, const CBLAS_INT_TYPE incX);

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (s, d, c, z)
 */
void cblas_sswap(const CBLAS_INT_TYPE N, float *X, const CBLAS_INT_TYPE incX, 
                 float *Y, const CBLAS_INT_TYPE incY);
void cblas_scopy(const CBLAS_INT_TYPE N, const float *X, const CBLAS_INT_TYPE incX, 
                 float *Y, const CBLAS_INT_TYPE incY);
void cblas_saxpy(const CBLAS_INT_TYPE N, const float alpha, const float *X,
                 const CBLAS_INT_TYPE incX, float *Y, const CBLAS_INT_TYPE incY);

void cblas_dswap(const CBLAS_INT_TYPE N, double *X, const CBLAS_INT_TYPE incX, 
                 double *Y, const CBLAS_INT_TYPE incY);
void cblas_dcopy(const CBLAS_INT_TYPE N, const double *X, const CBLAS_INT_TYPE incX, 
                 double *Y, const CBLAS_INT_TYPE incY);
void cblas_daxpy(const CBLAS_INT_TYPE N, const double alpha, const double *X,
                 const CBLAS_INT_TYPE incX, double *Y, const CBLAS_INT_TYPE incY);

void cblas_cswap(const CBLAS_INT_TYPE N, void *X, const CBLAS_INT_TYPE incX, 
                 void *Y, const CBLAS_INT_TYPE incY);
void cblas_ccopy(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX, 
                 void *Y, const CBLAS_INT_TYPE incY);
void cblas_caxpy(const CBLAS_INT_TYPE N, const void *alpha, const void *X,
                 const CBLAS_INT_TYPE incX, void *Y, const CBLAS_INT_TYPE incY);

void cblas_zswap(const CBLAS_INT_TYPE N, void *X, const CBLAS_INT_TYPE incX, 
                 void *Y, const CBLAS_INT_TYPE incY);
void cblas_zcopy(const CBLAS_INT_TYPE N, const void *X, const CBLAS_INT_TYPE incX, 
                 void *Y, const CBLAS_INT_TYPE incY);
void cblas_zaxpy(const CBLAS_INT_TYPE N, const void *alpha, const void *X,
                 const CBLAS_INT_TYPE incX, void *Y, const CBLAS_INT_TYPE incY);


/* 
 * Routines with S and D prefix only
 */
void cblas_srotg(float *a, float *b, float *c, float *s);
void cblas_srotmg(float *d1, float *d2, float *b1, const float b2, float *P);
void cblas_srot(const CBLAS_INT_TYPE N, float *X, const CBLAS_INT_TYPE incX,
                float *Y, const CBLAS_INT_TYPE incY, const float c, const float s);
void cblas_srotm(const CBLAS_INT_TYPE N, float *X, const CBLAS_INT_TYPE incX,
                float *Y, const CBLAS_INT_TYPE incY, const float *P);

void cblas_drotg(double *a, double *b, double *c, double *s);
void cblas_drotmg(double *d1, double *d2, double *b1, const double b2, double *P);
void cblas_drot(const CBLAS_INT_TYPE N, double *X, const CBLAS_INT_TYPE incX,
                double *Y, const CBLAS_INT_TYPE incY, const double c, const double  s);
void cblas_drotm(const CBLAS_INT_TYPE N, double *X, const CBLAS_INT_TYPE incX,
                double *Y, const CBLAS_INT_TYPE incY, const double *P);


/* 
 * Routines with S D C Z CS and ZD prefixes
 */
void cblas_sscal(const CBLAS_INT_TYPE N, const float alpha, float *X, const CBLAS_INT_TYPE incX);
void cblas_dscal(const CBLAS_INT_TYPE N, const double alpha, double *X, const CBLAS_INT_TYPE incX);
void cblas_cscal(const CBLAS_INT_TYPE N, const void *alpha, void *X, const CBLAS_INT_TYPE incX);
void cblas_zscal(const CBLAS_INT_TYPE N, const void *alpha, void *X, const CBLAS_INT_TYPE incX);
void cblas_csscal(const CBLAS_INT_TYPE N, const float alpha, void *X, const CBLAS_INT_TYPE incX);
void cblas_zdscal(const CBLAS_INT_TYPE N, const double alpha, void *X, const CBLAS_INT_TYPE incX);

/*
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const float alpha, const float *A, const CBLAS_INT_TYPE lda,
                 const float *X, const CBLAS_INT_TYPE incX, const float beta,
                 float *Y, const CBLAS_INT_TYPE incY);
void cblas_sgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE KL, const CBLAS_INT_TYPE KU, const float alpha,
                 const float *A, const CBLAS_INT_TYPE lda, const float *X,
                 const CBLAS_INT_TYPE incX, const float beta, float *Y, const CBLAS_INT_TYPE incY);
void cblas_strmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const float *A, const CBLAS_INT_TYPE lda, 
                 float *X, const CBLAS_INT_TYPE incX);
void cblas_stbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const float *A, const CBLAS_INT_TYPE lda, 
                 float *X, const CBLAS_INT_TYPE incX);
void cblas_stpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const float *Ap, float *X, const CBLAS_INT_TYPE incX);
void cblas_strsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const float *A, const CBLAS_INT_TYPE lda, float *X,
                 const CBLAS_INT_TYPE incX);
void cblas_stbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const float *A, const CBLAS_INT_TYPE lda,
                 float *X, const CBLAS_INT_TYPE incX);
void cblas_stpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const float *Ap, float *X, const CBLAS_INT_TYPE incX);

void cblas_dgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const double alpha, const double *A, const CBLAS_INT_TYPE lda,
                 const double *X, const CBLAS_INT_TYPE incX, const double beta,
                 double *Y, const CBLAS_INT_TYPE incY);
void cblas_dgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE KL, const CBLAS_INT_TYPE KU, const double alpha,
                 const double *A, const CBLAS_INT_TYPE lda, const double *X,
                 const CBLAS_INT_TYPE incX, const double beta, double *Y, const CBLAS_INT_TYPE incY);
void cblas_dtrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const double *A, const CBLAS_INT_TYPE lda, 
                 double *X, const CBLAS_INT_TYPE incX);
void cblas_dtbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const double *A, const CBLAS_INT_TYPE lda, 
                 double *X, const CBLAS_INT_TYPE incX);
void cblas_dtpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const double *Ap, double *X, const CBLAS_INT_TYPE incX);
void cblas_dtrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const double *A, const CBLAS_INT_TYPE lda, double *X,
                 const CBLAS_INT_TYPE incX);
void cblas_dtbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const double *A, const CBLAS_INT_TYPE lda,
                 double *X, const CBLAS_INT_TYPE incX);
void cblas_dtpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const double *Ap, double *X, const CBLAS_INT_TYPE incX);

void cblas_cgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *X, const CBLAS_INT_TYPE incX, const void *beta,
                 void *Y, const CBLAS_INT_TYPE incY);
void cblas_cgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE KL, const CBLAS_INT_TYPE KU, const void *alpha,
                 const void *A, const CBLAS_INT_TYPE lda, const void *X,
                 const CBLAS_INT_TYPE incX, const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_ctrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *A, const CBLAS_INT_TYPE lda, 
                 void *X, const CBLAS_INT_TYPE incX);
void cblas_ctbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void *A, const CBLAS_INT_TYPE lda, 
                 void *X, const CBLAS_INT_TYPE incX);
void cblas_ctpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *Ap, void *X, const CBLAS_INT_TYPE incX);
void cblas_ctrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *A, const CBLAS_INT_TYPE lda, void *X,
                 const CBLAS_INT_TYPE incX);
void cblas_ctbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void *A, const CBLAS_INT_TYPE lda,
                 void *X, const CBLAS_INT_TYPE incX);
void cblas_ctpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *Ap, void *X, const CBLAS_INT_TYPE incX);

void cblas_zgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *X, const CBLAS_INT_TYPE incX, const void *beta,
                 void *Y, const CBLAS_INT_TYPE incY);
void cblas_zgbmv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE KL, const CBLAS_INT_TYPE KU, const void *alpha,
                 const void *A, const CBLAS_INT_TYPE lda, const void *X,
                 const CBLAS_INT_TYPE incX, const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_ztrmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *A, const CBLAS_INT_TYPE lda, 
                 void *X, const CBLAS_INT_TYPE incX);
void cblas_ztbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void *A, const CBLAS_INT_TYPE lda, 
                 void *X, const CBLAS_INT_TYPE incX);
void cblas_ztpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *Ap, void *X, const CBLAS_INT_TYPE incX);
void cblas_ztrsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *A, const CBLAS_INT_TYPE lda, void *X,
                 const CBLAS_INT_TYPE incX);
void cblas_ztbsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void *A, const CBLAS_INT_TYPE lda,
                 void *X, const CBLAS_INT_TYPE incX);
void cblas_ztpsv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_DIAG Diag,
                 const CBLAS_INT_TYPE N, const void *Ap, void *X, const CBLAS_INT_TYPE incX);


/* 
 * Routines with S and D prefixes only
 */
void cblas_ssymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const float alpha, const float *A,
                 const CBLAS_INT_TYPE lda, const float *X, const CBLAS_INT_TYPE incX,
                 const float beta, float *Y, const CBLAS_INT_TYPE incY);
void cblas_ssbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const float alpha, const float *A,
                 const CBLAS_INT_TYPE lda, const float *X, const CBLAS_INT_TYPE incX,
                 const float beta, float *Y, const CBLAS_INT_TYPE incY);
void cblas_sspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const float alpha, const float *Ap,
                 const float *X, const CBLAS_INT_TYPE incX,
                 const float beta, float *Y, const CBLAS_INT_TYPE incY);
void cblas_sger(const enum CBLAS_ORDER order, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                const float alpha, const float *X, const CBLAS_INT_TYPE incX,
                const float *Y, const CBLAS_INT_TYPE incY, float *A, const CBLAS_INT_TYPE lda);
void cblas_ssyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float alpha, const float *X,
                const CBLAS_INT_TYPE incX, float *A, const CBLAS_INT_TYPE lda);
void cblas_sspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float alpha, const float *X,
                const CBLAS_INT_TYPE incX, float *Ap);
void cblas_ssyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float alpha, const float *X,
                const CBLAS_INT_TYPE incX, const float *Y, const CBLAS_INT_TYPE incY, float *A,
                const CBLAS_INT_TYPE lda);
void cblas_sspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float alpha, const float *X,
                const CBLAS_INT_TYPE incX, const float *Y, const CBLAS_INT_TYPE incY, float *A);

void cblas_dsymv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const double alpha, const double *A,
                 const CBLAS_INT_TYPE lda, const double *X, const CBLAS_INT_TYPE incX,
                 const double beta, double *Y, const CBLAS_INT_TYPE incY);
void cblas_dsbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const double alpha, const double *A,
                 const CBLAS_INT_TYPE lda, const double *X, const CBLAS_INT_TYPE incX,
                 const double beta, double *Y, const CBLAS_INT_TYPE incY);
void cblas_dspmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const double alpha, const double *Ap,
                 const double *X, const CBLAS_INT_TYPE incX,
                 const double beta, double *Y, const CBLAS_INT_TYPE incY);
void cblas_dger(const enum CBLAS_ORDER order, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                const double alpha, const double *X, const CBLAS_INT_TYPE incX,
                const double *Y, const CBLAS_INT_TYPE incY, double *A, const CBLAS_INT_TYPE lda);
void cblas_dsyr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const double *X,
                const CBLAS_INT_TYPE incX, double *A, const CBLAS_INT_TYPE lda);
void cblas_dspr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const double *X,
                const CBLAS_INT_TYPE incX, double *Ap);
void cblas_dsyr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const double *X,
                const CBLAS_INT_TYPE incX, const double *Y, const CBLAS_INT_TYPE incY, double *A,
                const CBLAS_INT_TYPE lda);
void cblas_dspr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const double *X,
                const CBLAS_INT_TYPE incX, const double *Y, const CBLAS_INT_TYPE incY, double *A);


/* 
 * Routines with C and Z prefixes only
 */
void cblas_chemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const void *alpha, const void *A,
                 const CBLAS_INT_TYPE lda, const void *X, const CBLAS_INT_TYPE incX,
                 const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_chbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void *alpha, const void *A,
                 const CBLAS_INT_TYPE lda, const void *X, const CBLAS_INT_TYPE incX,
                 const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_chpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const void *alpha, const void *Ap,
                 const void *X, const CBLAS_INT_TYPE incX,
                 const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_cgeru(const enum CBLAS_ORDER order, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                 const void *Y, const CBLAS_INT_TYPE incY, void *A, const CBLAS_INT_TYPE lda);
void cblas_cgerc(const enum CBLAS_ORDER order, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                 const void *Y, const CBLAS_INT_TYPE incY, void *A, const CBLAS_INT_TYPE lda);
void cblas_cher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float alpha, const void *X, const CBLAS_INT_TYPE incX,
                void *A, const CBLAS_INT_TYPE lda);
void cblas_chpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const float alpha, const void *X,
                const CBLAS_INT_TYPE incX, void *A);
void cblas_cher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE N,
                const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                const void *Y, const CBLAS_INT_TYPE incY, void *A, const CBLAS_INT_TYPE lda);
void cblas_chpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE N,
                const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                const void *Y, const CBLAS_INT_TYPE incY, void *Ap);

void cblas_zhemv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const void *alpha, const void *A,
                 const CBLAS_INT_TYPE lda, const void *X, const CBLAS_INT_TYPE incX,
                 const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_zhbmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K, const void *alpha, const void *A,
                 const CBLAS_INT_TYPE lda, const void *X, const CBLAS_INT_TYPE incX,
                 const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_zhpmv(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                 const CBLAS_INT_TYPE N, const void *alpha, const void *Ap,
                 const void *X, const CBLAS_INT_TYPE incX,
                 const void *beta, void *Y, const CBLAS_INT_TYPE incY);
void cblas_zgeru(const enum CBLAS_ORDER order, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                 const void *Y, const CBLAS_INT_TYPE incY, void *A, const CBLAS_INT_TYPE lda);
void cblas_zgerc(const enum CBLAS_ORDER order, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                 const void *Y, const CBLAS_INT_TYPE incY, void *A, const CBLAS_INT_TYPE lda);
void cblas_zher(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const void *X, const CBLAS_INT_TYPE incX,
                void *A, const CBLAS_INT_TYPE lda);
void cblas_zhpr(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo,
                const CBLAS_INT_TYPE N, const double alpha, const void *X,
                const CBLAS_INT_TYPE incX, void *A);
void cblas_zher2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE N,
                const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                const void *Y, const CBLAS_INT_TYPE incY, void *A, const CBLAS_INT_TYPE lda);
void cblas_zhpr2(const enum CBLAS_ORDER order, const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE N,
                const void *alpha, const void *X, const CBLAS_INT_TYPE incX,
                const void *Y, const CBLAS_INT_TYPE incY, void *Ap);

/*
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
 */

/* 
 * Routines with standard 4 prefixes (S, D, C, Z)
 */
void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE K, const float alpha, const float *A,
                 const CBLAS_INT_TYPE lda, const float *B, const CBLAS_INT_TYPE ldb,
                 const float beta, float *C, const CBLAS_INT_TYPE ldc);
void cblas_ssymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const float alpha, const float *A, const CBLAS_INT_TYPE lda,
                 const float *B, const CBLAS_INT_TYPE ldb, const float beta,
                 float *C, const CBLAS_INT_TYPE ldc);
void cblas_ssyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                 const float alpha, const float *A, const CBLAS_INT_TYPE lda,
                 const float beta, float *C, const CBLAS_INT_TYPE ldc);
void cblas_ssyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const float alpha, const float *A, const CBLAS_INT_TYPE lda,
                  const float *B, const CBLAS_INT_TYPE ldb, const float beta,
                  float *C, const CBLAS_INT_TYPE ldc);
void cblas_strmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const float alpha, const float *A, const CBLAS_INT_TYPE lda,
                 float *B, const CBLAS_INT_TYPE ldb);
void cblas_strsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const float alpha, const float *A, const CBLAS_INT_TYPE lda,
                 float *B, const CBLAS_INT_TYPE ldb);

void cblas_dgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE K, const double alpha, const double *A,
                 const CBLAS_INT_TYPE lda, const double *B, const CBLAS_INT_TYPE ldb,
                 const double beta, double *C, const CBLAS_INT_TYPE ldc);
void cblas_dsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const double alpha, const double *A, const CBLAS_INT_TYPE lda,
                 const double *B, const CBLAS_INT_TYPE ldb, const double beta,
                 double *C, const CBLAS_INT_TYPE ldc);
void cblas_dsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                 const double alpha, const double *A, const CBLAS_INT_TYPE lda,
                 const double beta, double *C, const CBLAS_INT_TYPE ldc);
void cblas_dsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const double alpha, const double *A, const CBLAS_INT_TYPE lda,
                  const double *B, const CBLAS_INT_TYPE ldb, const double beta,
                  double *C, const CBLAS_INT_TYPE ldc);
void cblas_dtrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const double alpha, const double *A, const CBLAS_INT_TYPE lda,
                 double *B, const CBLAS_INT_TYPE ldb);
void cblas_dtrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const double alpha, const double *A, const CBLAS_INT_TYPE lda,
                 double *B, const CBLAS_INT_TYPE ldb);

void cblas_cgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE K, const void *alpha, const void *A,
                 const CBLAS_INT_TYPE lda, const void *B, const CBLAS_INT_TYPE ldb,
                 const void *beta, void *C, const CBLAS_INT_TYPE ldc);
void cblas_csymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                 void *C, const CBLAS_INT_TYPE ldc);
void cblas_csyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *beta, void *C, const CBLAS_INT_TYPE ldc);
void cblas_csyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                  const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                  void *C, const CBLAS_INT_TYPE ldc);
void cblas_ctrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 void *B, const CBLAS_INT_TYPE ldb);
void cblas_ctrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 void *B, const CBLAS_INT_TYPE ldb);

void cblas_zgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const CBLAS_INT_TYPE K, const void *alpha, const void *A,
                 const CBLAS_INT_TYPE lda, const void *B, const CBLAS_INT_TYPE ldb,
                 const void *beta, void *C, const CBLAS_INT_TYPE ldc);
void cblas_zsymm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                 void *C, const CBLAS_INT_TYPE ldc);
void cblas_zsyrk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *beta, void *C, const CBLAS_INT_TYPE ldc);
void cblas_zsyr2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                  const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                  void *C, const CBLAS_INT_TYPE ldc);
void cblas_ztrmm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 void *B, const CBLAS_INT_TYPE ldb);
void cblas_ztrsm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_DIAG Diag, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 void *B, const CBLAS_INT_TYPE ldb);


/* 
 * Routines with prefixes C and Z only
 */
void cblas_chemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                 void *C, const CBLAS_INT_TYPE ldc);
void cblas_cherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                 const float alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const float beta, void *C, const CBLAS_INT_TYPE ldc);
void cblas_cher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                  const void *B, const CBLAS_INT_TYPE ldb, const float beta,
                  void *C, const CBLAS_INT_TYPE ldc);

void cblas_zhemm(const enum CBLAS_ORDER Order, const enum CBLAS_SIDE Side,
                 const enum CBLAS_UPLO Uplo, const CBLAS_INT_TYPE M, const CBLAS_INT_TYPE N,
                 const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const void *B, const CBLAS_INT_TYPE ldb, const void *beta,
                 void *C, const CBLAS_INT_TYPE ldc);
void cblas_zherk(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                 const double alpha, const void *A, const CBLAS_INT_TYPE lda,
                 const double beta, void *C, const CBLAS_INT_TYPE ldc);
void cblas_zher2k(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                  const enum CBLAS_TRANSPOSE Trans, const CBLAS_INT_TYPE N, const CBLAS_INT_TYPE K,
                  const void *alpha, const void *A, const CBLAS_INT_TYPE lda,
                  const void *B, const CBLAS_INT_TYPE ldb, const double beta,
                  void *C, const CBLAS_INT_TYPE ldc);

void cblas_xerbla(int p, const char *rout, const char *form, ...);

#ifdef __cplusplus
}
#endif
#endif
