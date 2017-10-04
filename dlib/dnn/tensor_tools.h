// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TeNSOR_TOOLS_H_
#define DLIB_TeNSOR_TOOLS_H_

#include "tensor.h"
#include "cudnn_dlibapi.h"
#include "cublas_dlibapi.h"
#include "cusolver_dlibapi.h"
#include "curand_dlibapi.h"
#include "cpu_dlib.h"
#include "cuda_dlib.h"
#include "../rand.h"
#include <memory>
#include "../geometry/rectangle.h"

namespace dlib
{
    bool dnn_prefer_fastest_algorithms();
    void set_dnn_prefer_fastest_algorithms();
    void set_dnn_prefer_smallest_algorithms();
}

namespace dlib { namespace tt
{

// ----------------------------------------------------------------------------------------

    void inverse_norms (
        resizable_tensor& invnorms,
        const tensor& data,
        const double eps
    );
    /*!
        ensures
            - #invnorms == reciprocal(sqrt(sum_cols(squared(mat(data))) + eps))
    !*/

    void dot_prods (
        resizable_tensor& out,
        const tensor& lhs,
        const tensor& rhs
    );
    /*!
        requires
            - have_same_dimensions(lhs,rhs) == true
        ensures
            - #out.num_samples() == lhs.num_samples()
            - #out.k() == #out.nr() == #out.nc() == 1
            - #out == sum_cols(pointwise_multiply(mat(lhs), mat(rhs))); 
    !*/

    void scale_columns (
        tensor& out,
        const tensor& m,
        const tensor& v
    );
    /*!
        requires
            - have_same_dimensions(out,m) == true
            - is_vector(v) == true
            - v.size() == mat(m).nc()
        ensures
            - performs: out = scale_columns(mat(m),mat(v));
    !*/

    void scale_rows (
        tensor& out,
        const tensor& m,
        const tensor& v
    );
    /*!
        requires
            - have_same_dimensions(out,m) == true
            - is_vector(v) == true
            - v.size() == m.num_samples()
        ensures
            - performs: out = scale_rows(mat(m),mat(v));
    !*/

    void scale_rows2 (
        float beta, 
        tensor& out,
        const tensor& m1,
        const tensor& m2,
        const tensor& v1,
        const tensor& v2
    );
    /*!
        requires
            - have_same_dimensions(out,m1) == true
            - have_same_dimensions(out,m2) == true
            - have_same_dimensions(v1,v2) == true
            - is_vector(v1) == true
            - v1.size() == m1.num_samples()
        ensures
            - performs: 
                out = beta*out + scale_rows(mat(m1) - scale_rows(mat(m2),mat(v1)), mat(v2));
    !*/

// ----------------------------------------------------------------------------------------
    
    void exp (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - performs: dest = exp(mat(src))
    !*/

// ----------------------------------------------------------------------------------------

    void log (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - performs: dest = log(mat(src))
    !*/

// ----------------------------------------------------------------------------------------

    void log10 (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - dest.size() == src.size()
        ensures
            - performs: dest = log10(mat(src))
    !*/

// ----------------------------------------------------------------------------------------

    void gemm (
        float beta,
        tensor& dest,
        float alpha,
        const tensor& lhs,
        bool trans_lhs,
        const tensor& rhs,
        bool trans_rhs
    );
    /*!
        requires
            - dest does not alias the memory of lhs or rhs
            - The dimensions of lhs and rhs must be compatible for matrix multiplication.
              In particular:
                - Let L == trans_lhs ? trans(mat(lhs)) : mat(lhs)
                - Let R == trans_rhs ? trans(mat(rhs)) : mat(rhs)
                - Let D == mat(dest)
                - D.nr() == L.nr() && D.nc() == R.nc()
                  (i.e. dest must be preallocated and have the correct output dimensions)
                - L.nc() == R.nr()
        ensures
            - performs: dest = alpha*L*R + beta*mat(dest)
    !*/

// ----------------------------------------------------------------------------------------

    class inv
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a functor for doing matrix inversion on the GPU.  The only
                reason it's an object is to avoid the reallocation of some GPU memory
                blocks if you want to do a bunch of matrix inversions in a row.
        !*/
    public:

        void operator() (
            const tensor& m,
            resizable_tensor& out
        );
        /*!
            requires
                - m.size() == m.num_samples()*m.num_samples()
                  (i.e. mat(m) must be a square matrix)
            ensures
                - out == inv(mat(m));
        !*/

    private:
#ifdef DLIB_USE_CUDA
        cuda::inv finv;
#endif
    };

// ----------------------------------------------------------------------------------------

    class tensor_rand
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a tool for filling a tensor with random numbers.  

                Note that the sequence of random numbers output by this object is different
                when dlib is compiled with DLIB_USE_CUDA.  So you should not write code
                that depends on any specific sequence of numbers coming out of a
                tensor_rand.

        !*/

    public:
        // not copyable
        tensor_rand(const tensor_rand&) = delete;
        tensor_rand& operator=(const tensor_rand&) = delete;

        tensor_rand() : tensor_rand(0) {}
        tensor_rand(unsigned long long seed);

        void fill_gaussian (
            tensor& data,
            float mean = 0,
            float stddev = 1
        );
        /*!
            requires
                - data.size()%2 == 0
            ensures
                - Fills data with random numbers drawn from a Gaussian distribution
                  with the given mean and standard deviation.
        !*/

        void fill_uniform (
            tensor& data
        );
        /*!
            ensures
                - Fills data with uniform random numbers in the range (0.0, 1.0].
        !*/

#ifdef DLIB_USE_CUDA
        cuda::curand_generator rnd;
#else
        dlib::rand rnd;
#endif
    };

// ----------------------------------------------------------------------------------------

    void multiply (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        requires
            - dest.k()  == src1.k()  == src2.k()
            - dest.nr() == src1.nr() == src2.nr()
            - dest.nc() == src1.nc() == src2.nc()
            - dest.num_samples(), src1.num_samples(), and src2.num_samples() must each
              either be 1 or whichever ones aren't equal to 1 must have the same values.
        ensures
            - let MD = max(dest.num_samples(), src1.num_samples(), src2.num_samples)
            - This function pointwise multiplies src1 with src2 and stores the result into
              #dest.  However, how the multiplication happens depends on the dimensions of
              the tensors.  First, when src1 and src2 are multiplied together, if either
              has a num_samples() dimension that is != MD, then it is first replicated to
              produce a tensor with num_samples()==MD dimensions and then they are
              pointwise multiplied together.

              Second, if dest.num_samples()==1, then after the pointwise multiplication of
              src1 with src2, the result has its samples summed to produce an output tensor
              with num_samples()==1 which is then assigned to #dest.
            - if (add_to) then
                - Instead of assigning the result to dest, this function adds the result to dest.
    !*/

    void multiply_conv (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        requires
            - if (have_same_dimensions(dest, src1) == true) then
                - src2.num_samples() == 1
                - src2.nr() == 1
                - src2.nc() == 1
                - src2.k() == src1.k()
            - else
                - have_same_dimensions(src1, src2) == true) 
                - dest.num_samples() == 1
                - dest.nr() == 1
                - dest.nc() == 1
                - dest.k() == src1.k()
        ensures
            - Performs #dest == src1*src2 
              In particular, if the elements of dest, src1, and src2 were indexed by (n,k,r,c) then
              we would have:
                - if (have_same_dimensions(dest,src1)) then
                    #dest(n,k,r,c) == src1(n,k,r,c)*src2(k)
                - else
                    #dest(k) == sum over {n,r,c} of src1(n,k,r,c)*src2(n,k,r,c)
            - if (add_to) then
                - Instead of assigning the result to dest, this function adds the result to dest.
    !*/

    void multiply_zero_padded (
        bool add_to,
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        ensures
            - if (add_to) then
                - performs: dest += src1 * src2
            - else
                - performs: dest = src1 * src2
            - In either case, the multiplication happens pointwise according to 4D tensor
              arithmetic.  If the dimensions don't match then missing elements are presumed
              to be equal to 0.
    !*/

// ----------------------------------------------------------------------------------------

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const float A,
        const float B
    );
    /*!
        requires
            - dest.size()==src.size()
        ensures
            - #dest == A*src + B
    !*/

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const float A
    );
    /*!
        requires
            - dest.size()==src.size()
        ensures
            - #dest == A*src 
    !*/

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const float A,
        const float B,
        const float C
    );
    /*!
        requires
            - dest.size()==src1.size()
            - dest.size()==src2.size()
        ensures
            - #dest == A*src1 + B*src2 + C
    !*/

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const float A,
        const float B
    );
    /*!
        requires
            - dest.size()==src1.size()
            - dest.size()==src2.size()
        ensures
            - #dest == A*src1 + B*src2
    !*/

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const tensor& src3,
        const float A,
        const float B,
        const float C,
        const float D
    );
    /*!
        requires 
            - dest.size()==src1.size()
            - dest.size()==src2.size()
            - dest.size()==src3.size()
        ensures
            - #dest == A*src1 + B*src2 + C*src3 + D
    !*/

    void affine_transform(
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const tensor& src3,
        const float A,
        const float B,
        const float C
    );
    /*!
        requires 
            - dest.size()==src1.size()
            - dest.size()==src2.size()
            - dest.size()==src3.size()
        ensures
            - #dest == A*src1 + B*src2 + C*src3
    !*/

    void affine_transform_range(
        size_t begin,
        size_t end,
        tensor& dest,
        const tensor& src1,
        const tensor& src2,
        const tensor& src3,
        const float A,
        const float B,
        const float C
    );
    /*!
        requires 
            - dest.size()==src1.size()
            - dest.size()==src2.size()
            - dest.size()==src3.size()
            - begin <= end <= dest.size()
        ensures
            - This function operates much like
              affine_transform(dest,src1,src2,src3,A,B,C,0), except that it runs over only
              the half open range [begin,end) rather than processing the entire tensor.
              Specifically, it does this:
                - for i in the range [begin, end):
                    - #dest.host()[i] == A*src1.host()[i] + B*src2.host()[i] + C*src3.host()[i]
    !*/

    void affine_transform(
        const rectangle& rect,
        tensor& dest, 
        const tensor& src1, 
        const tensor& src2, 
        const tensor& src3, 
        float A, 
        float B,
        float C
    );
    /*!
        requires
            - dest.size()==src1.size()
            - dest.size()==src2.size()
            - dest.size()==src3.size()
            - dest.num_samples()==src1.num_samples()
            - dest.num_samples()==src2.num_samples()
            - dest.num_samples()==src3.num_samples()
            - get_rect(mat(dest)).contains(rect) == true
              (i.e. rect must be entirely contained within dest)
        ensures
            - This function operates much like
              affine_transform(dest,src1,src2,src3,A,B,C,0), except that it runs over only
              the sub-rectangle indicated by rect.  In particular, this function is equivalent
              to:
                set_subm(dest,rect) = A*subm(mat(src1),rect) + B*subm(mat(src2),rect) + C*subm(mat(src3),rect)
    !*/

// ----------------------------------------------------------------------------------------

    void affine_transform(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    );
    /*!
        requires
            - have_same_dimensions(dest,src) == true
            - if (A.num_samples() == 1) then
                - B.num_samples() == 1
            - else
                - A.num_samples() == src.num_samples()
                - B.num_samples() == src.num_samples()
            - A.nr() == B.nr() == src.nr()
            - A.nc() == B.nc() == src.nc()
            - A.k()  == B.k()  == src.k()
        ensures
            - if (A.num_samples() == 1) then
                - #dest == A*src + B
                    (done for each sample in src)
            - else
                - for all valid i:
                    - #dest.host()[i] == A.host()[i]*src.host()[i] + B.host()[i]  
    !*/

// ----------------------------------------------------------------------------------------

    void affine_transform_conv(
        tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    );
    /*!
        requires
            - have_same_dimensions(dest,src) == true
            - have_same_dimensions(A, B) == true
            - A.num_samples() == 1
            - A.nr() == 1
            - A.nc() == 1
            - A.k() == src.k()
        ensures
            - Performs #dest == A*src + B
              In particular, if the elements of dest and src were indexed by (n,k,r,c) then
              we would have:
                #dest(n,k,r,c) == A(k)*src(n,k,r,c) + B(k).
    !*/

// ----------------------------------------------------------------------------------------

    void compute_adam_update (
        size_t begin,
        size_t end,
        tensor& s,
        tensor& m,
        tensor& v,
        const float t,
        const float learning_rate,
        const float weight_decay,
        const float momentum1,
        const float momentum2,
        const tensor& params,
        const tensor& params_grad
    );
    /*!
        requires
            - s.size() == m.size() = v.size() == params.size() == params_grad.size()
            - t > 0
            - learning_rate > 0
            - weight_decay >= 0
            - 0 <= momentum1 < 1
            - 0 <= momentum2 < 1
            - begin <= end <= params.size()
        ensures
            - This function implements the ADAM parameter update method described in the paper:
                Kingma, Diederik P., and Jimmy Ba Adam. "A method for stochastic
                optimization." International Conference on Learning Representation. 2015.
              Specifically, it implements the method shown as Algorithm 1.
            - #s is the update vector that should be added to the parameters.
            - The function only operates in the half open range [begin,end) of the memory
              blocks of each tensor.  E.g. to make this function run on the entire tensor
              set begin to 0 and end to params.size().
    !*/

// ----------------------------------------------------------------------------------------

    void batch_normalize_inference (
        const double eps,
        resizable_tensor& dest,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta,
        const tensor& running_means,
        const tensor& running_variances
    );
    /*!
        requires
            - eps > 0
            - gamma.num_samples() == 1 
            - gamma.nr() == src.nr() 
            - gamma.nc() == src.nc() 
            - gamma.k()  == src.k()
            - have_same_dimensions(gamma, beta) 
            - have_same_dimensions(gamma, running_means) 
            - have_same_dimensions(gamma, running_variances)
        ensures
            - Linearly transforms src as a call to batch_normalize() would if src had means
              and variances as given by running_means and running_variances.  That is, this
              function performs: 
                dest = gamma*(src-running_means)/sqrt(running_variances+eps) + beta
              Note that it does it in a pointwise fashion over the samples in src.
    !*/

    void batch_normalize (
        const double eps,
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& invstds,
        const double averaging_factor,
        resizable_tensor& running_means,
        resizable_tensor& running_variances,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta 
    );
    /*!
        requires
            - eps > 0
            - src.num_samples() > 1
            - gamma.num_samples() == 1
            - beta.num_samples() == 1
            - gamma.nr() == beta.nr() == src.nr()
            - gamma.nc() == beta.nc() == src.nc()
            - gamma.k()  == beta.k()  == src.k()
            - 0 <= averaging_factor <= 1
            - if (averaging_factor != 1)
                - have_same_dimensions(running_means, means) == true
                - have_same_dimensions(running_variances, invstds) == true
        ensures
            - have_same_dimensions(#dest, src) == true
            - #means.num_samples() == 1
            - #invstds.num_samples() == 1
            - means.nr() == invstds.nr() == src.nr()
            - means.nc() == invstds.nc() == src.nc()
            - means.k()  == invstds.k()  == src.k()
            - #src == the batch normalized version of src.
            - #means == the mean values of the contents of src.
            - #invstds == 1/(the standard deviation values of the contents of src).
            - #running_means = (1-averaging_factor)*mat(#running_means) + averaging_factor*mat(#means);
            - #running_variances = (1-averaging_factor)*mat(#running_variances) + averaging_factor*(variance of contents of src);
    !*/

    void batch_normalize_gradient (
        const double eps,
        const tensor& gradient_input,
        const tensor& means,
        const tensor& invstds,
        const tensor& src,
        const tensor& gamma,
        tensor& src_grad,
        tensor& gamma_grad, 
        tensor& beta_grad 
    );
    /*!
        requires
            - eps > 0
            - invstds and means should be the output of a call to
              batch_normalize(eps,dest,means,invstds,src,gamma,beta)
            - have_same_dimensions(gradient_input, src) == true
            - have_same_dimensions(src, src_grad) == true
            - src.num_samples() > 1
            - gamma.num_samples() == 1
            - have_same_dimensions(gamma, gamma_grad) == true
            - have_same_dimensions(gamma, beta_grad) == true
            - gamma.nr() == src.nr()
            - gamma.nc() == src.nc()
            - gamma.k()  == src.k()
            - have_same_dimensions(means, gamma) == true
            - have_same_dimensions(invstds, gamma) == true
        ensures
            - Let f(src,gamma,beta) == dot(gradient_input, dest output of
              batch_normalize(eps,dest,means,invstds,src,gamma,beta))
            - Adds the gradient of f() with respect to src to #src_grad.
            - Assigns the gradient of f() with respect to gamma to #gamma_grad.
            - Assigns the gradient of f() with respect to beta to #beta_grad.
    !*/

// ----------------------------------------------------------------------------------------

    void batch_normalize_conv_inference (
        const double eps,
        resizable_tensor& dest,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta,
        const tensor& running_means,
        const tensor& running_variances
    );
    /*!
        requires
            - eps > 0
            - gamma.num_samples() == 1 
            - gamma.nr() == 1 
            - gamma.nc() == 1 
            - gamma.k()  == src.k()
            - have_same_dimensions(gamma, beta) 
            - have_same_dimensions(gamma, running_means) 
            - have_same_dimensions(gamma, running_variances)
        ensures
            - Linearly transforms src as a call to batch_normalize_conv() would if src had
              means and variances as given by running_means and running_variances.  That
              is, this function performs: 
                dest = gamma*(src-running_means)/sqrt(running_variances+eps) + beta
              Note that it does this in a pointwise fashion over the samples, rows, and
              columns in src.
    !*/

    void batch_normalize_conv (
        const double eps,
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& invstds,
        const double averaging_factor,
        resizable_tensor& running_means,
        resizable_tensor& running_variances,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta 
    );
    /*!
        requires
            - eps > 0
            - src.num_samples() > 1
            - gamma.num_samples()==gamma.nr()==gamma.nc() == 1
            - beta.num_samples() ==beta.nr() ==gamma.nc() == 1
            - gamma.k()  == beta.k()  == src.k()
            - 0 <= averaging_factor <= 1
            - if (averaging_factor != 1)
                - have_same_dimensions(running_means, means) == true
                - have_same_dimensions(running_variances, invstds) == true
        ensures
            - have_same_dimensions(#dest, src) == true
            - #means.num_samples()==means.nr()==means.nc() == 1
            - #invstds.num_samples() ==invstds.nr() ==invstds.nc() == 1
            - means.k()  == invstds.k()  == src.k()
            - #src == the batch normalized version of src.
            - #means == the mean values of the contents of src.
            - #invstds == 1/(the standard deviation values of the contents of src).
            - #running_means = (1-averaging_factor)*mat(#running_means) + averaging_factor*mat(#means);
            - #running_variances = (1-averaging_factor)*mat(#running_variances) + averaging_factor*(variance of contents of src);
    !*/

    void batch_normalize_conv_gradient (
        const double eps,
        const tensor& gradient_input,
        const tensor& means,
        const tensor& invstds,
        const tensor& src,
        const tensor& gamma,
        tensor& src_grad,
        tensor& gamma_grad, 
        tensor& beta_grad 
    );
    /*!
        requires
            - eps > 0
            - invstds and means should be the output of a call to
              batch_normalize_conv(eps,dest,means,invstds,src,gamma,beta)
            - have_same_dimensions(gradient_input, src) == true
            - have_same_dimensions(src, src_grad) == true
            - src.num_samples() > 1
            - gamma.num_samples()==gamma.nr()==gamma.nc() == 1
            - have_same_dimensions(gamma, gamma_grad) == true
            - have_same_dimensions(gamma, beta_grad) == true
            - gamma.k()  == src.k()
            - have_same_dimensions(means, gamma) == true
            - have_same_dimensions(invstds, gamma) == true
        ensures
            - Let f(src,gamma,beta) == dot(gradient_input, dest output of
              batch_normalize_conv(eps,dest,means,invstds,src,gamma,beta))
            - Adds the gradient of f() with respect to src to #src_grad.
            - Assigns the gradient of f() with respect to gamma to #gamma_grad.
            - Assigns the gradient of f() with respect to beta to #beta_grad.
    !*/

// -----------------------------------------------------------------------------------

    void threshold (
        tensor& data,
        float thresh
    );
    /*!
        ensures
            - Sets all elements of data to 1 or 0 depending on if they are above or below
              the given threshold.  Specifically, for all valid i:
                - #data.host()[i] == data.host()[i]>thresh ? 1 : 0
    !*/

    void dot (
        const tensor& a,
        const tensor& b,
        tensor& result,
        size_t idx
    );
    /*!
        requires
            - a.size() == b.size()
            - idx < result.size()
        ensures
            - #result.host()[idx] == result.host()[idx] + dot(a,b);
              I.e. Adds the dot product between a and b into the idx-th element of result.
              The reason you might want to use this more complex version of dot() is
              because, when using CUDA, it runs by generating asynchronous kernel launches
              whereas the version of dot() that returns the result immediately as a scalar
              must block the host while we wait for the result to be computed and then
              transfered from the GPU do the host for return by dot().  So this version of
              dot() might be much faster in some cases.
    !*/

// ----------------------------------------------------------------------------------------

    void add(
        float beta,
        tensor& dest,
        float alpha,
        const tensor& src
    );
    /*!
        requires
            - One of the following is true: 
                - have_same_dimensions(src, dest)
                - src.num_samples()==1 && src.k()==dest.k() && src.nr()==1 && src.nc()==1
                - src.num_samples()==1 && src.k()==dest.k() && src.nr()==dest.nr() && src.nc()==dest.nc()
                - src.num_samples()==1 && src.k()==1 && src.nr()==dest.nr() && src.nc()==dest.nc()
                - src.num_samples()==dest.num_samples() && src.k()==1 && src.nr()==1 && src.nc()==1
            - is_same_object(src,dest) == false
        ensures
            - performs: dest = beta*dest + alpha*src
              However, how the addition happens depends on the dimensions of src.  In
              particular, this function adds the scaled values of one src tensor to dest.
              Each dimension of the src tensor must match the corresponding dimension of
              the dest tensor or must be equal to 1. In the latter case, the same value
              from the src tensor, for those dimensions, will be used to add into the dest
              tensor.
    !*/

// ----------------------------------------------------------------------------------------

    void add (
        tensor& dest,
        const tensor& src1,
        const tensor& src2
    );
    /*!
        ensures
            - performs: dest = src1 + src2
              The addition happens pointwise according to 4D tensor arithmetic.  If the
              dimensions don't match then missing elements are presumed to be equal to 0.
    !*/

// ----------------------------------------------------------------------------------------

    void assign_conv_bias_gradient (
        tensor& grad,
        const tensor& gradient_input
    );
    /*!
        requires
            - grad.num_samples() == 1
            - grad.k()  >= 1
            - grad.nr() == 1
            - grad.nc() == 1
            - gradient_input.k() == grad.k()
            - gradient_input.size() > 0
            - is_same_object(grad,gradient_input) == false
        ensures
            - let BIAS be a tensor with the same dimensions as grad.
            - let OUT be the output of add(1,OUT,1,BIAS)
            - let f(gradient_input,BIAS) == dot(gradient_input,OUT)
            - Then this function computes the gradient of f() with respect to BIAS and
              assigns it to grad.
    !*/

// ----------------------------------------------------------------------------------------

    void assign_bias_gradient (
        tensor& grad,
        const tensor& gradient_input
    );
    /*!
        requires
            - grad.num_samples() == 1
            - gradient_input.k() == grad.k()
            - gradient_input.nr() == grad.nr()
            - gradient_input.nc() == grad.nc()
            - gradient_input.size() > 0
            - is_same_object(grad,gradient_input) == false
        ensures
            - let BIAS be a tensor with the same dimensions as grad.
            - let OUT be the output of add(1,OUT,1,BIAS)
            - let f(gradient_input,BIAS) == dot(gradient_input,OUT)
            - Then this function computes the gradient of f() with respect to BIAS and
              assigns it to grad.
    !*/

// ----------------------------------------------------------------------------------------

    class tensor_conv
    {
    public:
        tensor_conv(const tensor_conv&) = delete;
        tensor_conv& operator=(const tensor_conv&) = delete;

        tensor_conv() {}

        void clear(
        ) { impl.clear(); }

        void operator() (
            const bool add_to_output,
            tensor& output,
            const tensor& data,
            const tensor& filters
        ) { impl(add_to_output,output,data,filters); }
        /*!
            requires
                - setup() has been called.  Specifically, setup() has been called like this:
                    this->setup(data, filters, stride_y, stride_x, padding_y, padding_x);
                - is_same_object(output,data) == false
                - is_same_object(output,filters) == false
                - filters.k() == data.k()
                - filters.nr() <= src.nr() + 2*padding_y
                - filters.nc() <= src.nc() + 2*padding_x
                - #output.num_samples() == data.num_samples()
                - #output.k() == filters.num_samples()
                - #output.nr() == 1+(data.nr() + 2*padding_y - filters.nr())/stride_y
                - #output.nc() == 1+(data.nc() + 2*padding_x - filters.nc())/stride_x
            ensures
                - Convolves filters over data.  If add_to_output==true then we add the
                  results to output, otherwise we assign to output, overwriting the
                  previous values in output.
                - filters contains filters.num_samples() filters. 
        !*/

        void operator() (
            const bool add_to_output,
            resizable_tensor& output,
            const tensor& data,
            const tensor& filters
        ) { impl(add_to_output,output,data,filters); }
        /*!
            requires
                - setup() has been called.  Specifically, setup() has been called like this:
                    this->setup(data, filters, stride_y, stride_x, padding_y, padding_x);
                - is_same_object(output,data) == false
                - is_same_object(output,filters) == false
                - filters.k() == data.k()
                - filters.nr() <= src.nr() + 2*padding_y
                - filters.nc() <= src.nc() + 2*padding_x
            ensures
                - Convolves filters over data.  If add_to_output==true then we add the
                  results to output, otherwise we assign to output, overwriting the
                  previous values in output.  
                - filters contains filters.num_samples() filters. 
                - #output.num_samples() == data.num_samples()
                - #output.k() == filters.num_samples()
                - #output.nr() == 1+(data.nr() + 2*padding_y - filters.nr())/stride_y
                - #output.nc() == 1+(data.nc() + 2*padding_x - filters.nc())/stride_x
        !*/

        void get_gradient_for_data (
            const bool add_to_output,
            const tensor& gradient_input, 
            const tensor& filters,
            tensor& data_gradient
        ) { impl.get_gradient_for_data(add_to_output,gradient_input,filters,data_gradient); }
        /*!
            requires
                - One of the following must be true:
                    - filters has the same dimensions as the filters object given to the
                      last call to operator().  Also, data_gradient has the same dimensions
                      as the data object given to the last call to operator().
                    - setup() has been called.  Specifically, setup() has been called like this:
                      this->setup(data_gradient, filters, stride_y, stride_x, padding_y, padding_x);
                - gradient_input has the following dimensions:
                    - gradient_input.num_samples() == data_gradient.num_samples()
                    - gradient_input.k() == filters.num_samples()
                    - gradient_input.nr() == 1+(data_gradient.nr() + 2*padding_y - filters.nr())/stride_y
                    - gradient_input.nc() == 1+(data_gradient.nc() + 2*padding_x - filters.nc())/stride_x
                    - NOTE, these dimensions are what you would obtain if gradient_input
                      has the same dimensions as the last output of operator().  
                - is_same_object(data_gradient,filters) == false
                - is_same_object(data_gradient,gradient_input) == false
            ensures
                - let OUT be the output of (*this)(OUT,data,filters,sx,sy).
                - let f(data,filters) == dot(OUT, gradient_input)
                - if (add_to_output) then
                    - This function finds the gradient of f() with respect to data and adds
                      this gradient to data_gradient.
                - else
                    - This function finds the gradient of f() with respect to data and
                      assigns this gradient to data_gradient, overwriting the previous
                      values in data_gradient.
        !*/

        void get_gradient_for_filters (
            const bool add_to_output,
            const tensor& gradient_input, 
            const tensor& data,
            tensor& filters_gradient
        ) { impl.get_gradient_for_filters(add_to_output,gradient_input,data,filters_gradient); }
        /*!
            requires
                - One of the following must be true:
                    - filters_gradient has the same dimensions as the filters object given
                      to the last call to operator().  Also, data has the same dimensions
                      as the data object given to the last call to operator().
                    - setup() has been called.  Specifically, setup() has been called like this:
                      this->setup(data, filters_gradient, stride_y, stride_x, padding_y, padding_x);
                - gradient_input has the following dimensions:
                    - gradient_input.num_samples() == data.num_samples()
                    - gradient_input.k() == filters.num_samples()
                    - gradient_input.nr() == 1+(data.nr() + 2*padding_y - filters.nr())/stride_y
                    - gradient_input.nc() == 1+(data.nc() + 2*padding_x - filters.nc())/stride_x
                    - NOTE, these dimensions are what you would obtain if gradient_input
                      has the same dimensions as the last output of operator().  
                - is_same_object(filters_gradient,data) == false
                - is_same_object(filters_gradient,gradient_input) == false
            ensures
                - let OUT be the output of (*this)(OUT,data,filters,sx,sy).
                - let f(data,filters) == dot(OUT, gradient_input)
                - if (add_to_output) then
                    - This function finds the gradient of f() with respect to filters and
                      adds this gradient to filters_gradient.
                - else 
                    - This function finds the gradient of f() with respect to filters and
                      assigns this gradient to filters_gradient, overwriting the previous
                      values in filters_gradient.
        !*/

 
        void setup(
            const tensor& data,
            const tensor& filters,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        ) {impl.setup(data,filters,stride_y,stride_x,padding_y,padding_x); }
        /*!
            requires
                - filters.k() == data.k()
                - stride_y > 0
                - stride_x > 0
                - 0 <= padding_y < filters.nr()
                - 0 <= padding_x < filters.nc()
            ensures
                - When operator() is called, the output tensor will have these dimensions:
                    - output.nr() == 1+(data.nr() + 2*padding_y - filters.nr())/stride_y
                    - output.nc() == 1+(data.nc() + 2*padding_x - filters.nc())/stride_x
                    - output.num_samples() == data.num_samples()
                    - output.k() == filters.num_samples()
                - The point of setup() is to allow this object to gather information about
                  all the tensor sizes and filter layouts involved in the computation.  In
                  particular, the reason the tensors are input into setup() is just to
                  observe their sizes.  setup() doesn't do anything with the contents of
                  the tensors, or store any kind of references to the data or filter
                  tensors. 
        !*/
       
    private:
#ifdef DLIB_USE_CUDA
        cuda::tensor_conv impl;
#else
        cpu::tensor_conv impl;
#endif

    };

// ----------------------------------------------------------------------------------------

    class pooling
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                The pooling object is a tool for performing spatial pooling over a tensor.
                It can be configured to do either max or average pooling.
        !*/
    public:

        pooling(const pooling&) = delete;
        pooling& operator=(const pooling&) = delete;

        pooling (
        ) = default;

        void clear(
        ) { impl.clear(); }

        void setup_max_pooling(
            int window_height,
            int window_width,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        ) { impl.setup_max_pooling(window_height, window_width, stride_y, stride_x, padding_y, padding_x); }
        /*!
            requires
                - window_height > 0
                - window_width > 0
                - stride_y > 0
                - stride_x > 0
                - 0 <= padding_y < window_height
                - 0 <= padding_x < window_width
            ensures
                - When you call operator() it will do max pooling with the given
                  parameters.
        !*/

        void setup_avg_pooling(
            int window_height,
            int window_width,
            int stride_y,
            int stride_x,
            int padding_y,
            int padding_x
        ) { impl.setup_avg_pooling(window_height, window_width, stride_y, stride_x, padding_y, padding_x); }
        /*!
            requires
                - window_height > 0
                - window_width > 0
                - stride_y > 0
                - stride_x > 0
                - 0 <= padding_y < window_height
                - 0 <= padding_x < window_width
            ensures
                - When you call operator() it will do average pooling with the given
                  parameters.
        !*/

        bool does_max_pooling(
        ) const { return impl.does_max_pooling(); }

        void operator() (
            resizable_tensor& dest,
            const tensor& src
        ) { impl(dest, src); }
        /*!
            requires
                - is_same_object(dest,src) == false
                - either setup_max_pooling() or setup_avg_pooling() has been called.
                - window_width  <= src.nc() + 2*padding_x
                - window_height <= src.nr() + 2*padding_y
            ensures
                - #dest.num_samples() == src.num_samples()
                - #dest.k() == src.k()
                - #dest.nr() == 1 + (src.nr() + 2*padding_y - window_height)/stride_y
                - #dest.nc() == 1 + (src.nc() + 2*padding_x - window_width)/stride_x
                - WINDOW == centered_rect(x*stride_x + window_width/2 - padding_x,
                                          y*stride_y + window_height/2 - padding_y,
                                          window_width,
                                          window_height)
                - for all valid s, k, r, and c:
                    - if (does_max_pooling()) then
                        - image_plane(#dest,s,k)(r,c) == max(subm_clipped(image_plane(src,s,k),WINDOW(c,r)))
                    - else
                        - image_plane(#dest,s,k)(r,c) == mean(subm_clipped(image_plane(src,s,k),WINDOW(c,r)))
        !*/

        void get_gradient(
            const tensor& gradient_input, 
            const tensor& dest,
            const tensor& src,
            tensor& grad 
        ) { impl.get_gradient(gradient_input, dest, src, grad); }
        /*!
            requires
                - have_same_dimensions(gradient_input,dest) == true
                - have_same_dimensions(src,grad) == true
                - dest contains the result of calling (*this)(dest,src)
                - is_same_object(grad,gradient_input) == false
                - is_same_object(grad,dest) == false
                - is_same_object(grad,src) == false
            ensures
                - Recalling that dest is the output of (*this)(dest,src),
                  let f(src) == dot(gradient_input,dest)
                - Then this function computes the gradient of f() with respect to src and
                  adds it to grad.
        !*/

        private:
#ifdef DLIB_USE_CUDA
        cuda::pooling impl;
#else
        cpu::pooling impl;
#endif
    };

// ----------------------------------------------------------------------------------------

    void softmax (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - Note that the softmax function is a vector valued function: 
                s(x) == exp(x)/sum(exp(x)) 
            - Computes the softmax function on src and writes the results to dest.  The
              softmax is computed per spatial location across the different channels at
              each location.  That is, softmax() outputs a new tensor, #dest, where each of
              the spatial locations in dest (i.e. image idx, row idx, and column idx)
              contains the output of s() evaluated over the channel values at each
              location.
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

    void softmax_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(dest,gradient_input) == true 
            - have_same_dimensions(dest,grad) == true 
        ensures
            - We interpret dest as the output of softmax(dest,SRC) for some SRC tensor.
              Then let f(SRC) == dot(gradient_input,dest).  Then this function computes the
              gradient of f() with respect to SRC and stores it to grad.  Moreover, if
              is_same_object(grad,gradient_input)==true then the output is assigned to
              grad, replacing its previous contents.  Otherwise the output is added to
              grad.
            - This function supports in-place operation, i.e. having
              is_same_object(grad, gradient_input)==true
    !*/

// ----------------------------------------------------------------------------------------

    void sigmoid (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - for all valid i:
                - #dest.host()[i] == 1/(1+std::exp(-src.host()[i])) 
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

    void sigmoid_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(dest,gradient_input) == true 
            - have_same_dimensions(dest,grad) == true 
        ensures
            - Recalling that dest is the output of sigmoid(dest,SRC) for some SRC tensor,
              let f(SRC) == dot(gradient_input,dest).  Then this function computes the
              gradient of f() with respect to SRC and stores it to grad.  Moreover, if
              is_same_object(grad,gradient_input)==true then the output is assigned to
              grad, replacing its previous contents.  Otherwise the output is added to
              grad.
            - This function supports in-place operation, i.e. having
              is_same_object(grad, gradient_input)==true
    !*/

// ----------------------------------------------------------------------------------------

    void relu (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - for all valid i:
                - #dest.host()[i] == std::max(0,src.host()[i]) 
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

    void relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(dest,gradient_input) == true 
            - have_same_dimensions(dest,grad) == true 
        ensures
            - Recalling that dest is the output of relu(dest,SRC) for some SRC tensor,
              let f(SRC) == dot(gradient_input,dest).  Then this function computes the
              gradient of f() with respect to SRC and stores it to grad.  Moreover, if
              is_same_object(grad,gradient_input)==true then the output is assigned to
              grad, replacing its previous contents.  Otherwise the output is added to
              grad.
            - This function supports in-place operation, i.e. having
              is_same_object(grad, gradient_input)==true
    !*/

// ----------------------------------------------------------------------------------------

    void prelu (
        tensor& dest,
        const tensor& src,
        const tensor& param
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
            - param.size() == 1
        ensures
            - for all valid i:
                - if (src.host()[i] > 0) then
                    - #dest.host()[i] == src.host()[i]
                - else
                    - #dest.host()[i] == src.host()[i] * param.host()[0]
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

    void prelu_gradient (
        tensor& grad,
        const tensor& src,
        const tensor& gradient_input,
        const tensor& param,
        tensor& params_grad 
    );
    /*!
        requires
            - have_same_dimensions(grad,src) == true 
            - have_same_dimensions(grad,gradient_input) == true 
            - param.size() == 1
            - params_grad.size() == 1
            - is_same_object(grad, gradient_input) == false
        ensures
            - Recalling that dest is the output of prelu(dest,src,param) let 
              f(src,param) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to src and
              param.  It assigns the gradient with respect to param to #params_grad and
              adds the gradient with respect to src to #grad.
    !*/

// ----------------------------------------------------------------------------------------

    void tanh (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest, src) == true
        ensures
            - for all valid i:
                - #dest.host()[i] == std::tanh(src.host()[i]) 
            - This function supports in-place operation, i.e. having
              is_same_object(dest, src)==true
    !*/

    void tanh_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(dest,gradient_input) == true 
            - have_same_dimensions(dest,grad) == true 
        ensures
            - Recalling that dest is the output of tanh(dest,SRC) for some SRC tensor,
              let f(SRC) == dot(gradient_input,dest).  Then this function computes the
              gradient of f() with respect to SRC and stores it to grad.  Moreover, if
              is_same_object(grad,gradient_input)==true then the output is assigned to
              grad, replacing its previous contents.  Otherwise the output is added to
              grad.
            - This function supports in-place operation, i.e. having
              is_same_object(grad, gradient_input)==true
    !*/

// ----------------------------------------------------------------------------------------

    void resize_bilinear (
        tensor& dest,
        long dest_row_stride,
        long dest_channel_stride,
        const tensor& src,
        long src_row_stride,
        long src_channel_stride
    );
    /*!
        requires
            - is_same_object(dest, src)==false
            - dest.num_samples() == src.num_samples()
            - dest.k() == src.k()
        ensures
            - for all valid i,k:  image_plane(dest,i,k) is a copy of image_plane(src,i,k)
              that has been bilinearly interpolated to fit into the shape of
              image_plane(dest,i,k).
            - Instead of supposing the row stride and channel stride in the tensors is
              given by tensor::nc() and tensor::nr()*tensor::nc() respectively, we use the
              provided stride values to transition from one row and channel to the next.
              This is useful in combination with alias_tensor objects since it allows you
              to operate on subwindows in an image.
    !*/

    void resize_bilinear_gradient (
        tensor& grad,
        long grad_row_stride,
        long grad_channel_stride,
        const tensor& gradient_input,
        long gradient_input_row_stride,
        long gradient_input_channel_stride
    );
    /*!
        requires
            - is_same_object(grad, gradient_input)==false
            - gradient_input.num_samples() == grad.num_samples()
            - gradient_input.k() == grad.k()
        ensures
            - Suppose that DEST is the output of resize_bilinear(DEST,SRC) for some SRC
              tensor, let f(SRC) == dot(gradient_input,DEST).  Then this function computes
              the gradient of f() with respect to SRC and adds it to grad.   It should be
              noted that we don't need to know the contents of DEST to compute this
              gradient.  All that matters is that gradient_input have the same dimensions
              as DEST.
            - Instead of supposing the row stride and channel stride in the tensors is
              given by tensor::nc() and tensor::nr()*tensor::nc() respectively, we use the
              provided stride values to transition from one row and channel to the next.
              This is useful in combination with alias_tensor objects since it allows you
              to operate on subwindows in an image.
    !*/

    inline void resize_bilinear (
        tensor& dest,
        const tensor& src
    ) { resize_bilinear(dest, dest.nc(), dest.nr()*dest.nc(), src, src.nc(), src.nr()*src.nc()); }
    /*!
        requires
            - is_same_object(dest, src)==false
            - dest.num_samples() == src.num_samples()
            - dest.k() == src.k()
        ensures
            - for all valid i,k:  image_plane(dest,i,k) is a copy of image_plane(src,i,k)
              that has been bilinearly interpolated to fit into the shape of
              image_plane(dest,i,k).
    !*/

    inline void resize_bilinear_gradient (
        tensor& grad,
        const tensor& gradient_input
    ) { resize_bilinear_gradient(grad, grad.nc(), grad.nr()*grad.nc(), gradient_input, gradient_input.nc(), gradient_input.nr()*gradient_input.nc()); }
    /*!
        requires
            - is_same_object(grad, gradient_input)==false
            - gradient_input.num_samples() == grad.num_samples()
            - gradient_input.k() == grad.k()
        ensures
            - Suppose that DEST is the output of resize_bilinear(DEST,SRC) for some SRC
              tensor, let f(SRC) == dot(gradient_input,DEST).  Then this function computes
              the gradient of f() with respect to SRC and adds it to grad.   It should be
              noted that we don't need to know the contents of DEST to compute this
              gradient.  All that matters is that gradient_input have the same dimensions
              as DEST.
    !*/

// ----------------------------------------------------------------------------------------

    class multi_device_tensor_averager
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for very quickly averaging a bunch of tensors
                together.
        !*/
    public:

        multi_device_tensor_averager(const multi_device_tensor_averager&) = delete;
        multi_device_tensor_averager& operator=(const multi_device_tensor_averager&) = delete;

        multi_device_tensor_averager() = default;

        void set(
            std::vector<tensor*> items
        )
        /*!
            requires
                - All the tensors in items are the same size
            ensures
                - When you call average() we will average the tensors in items.
                - It's important that the tensors already be allocated to their devices
                  before you call set().  This is because set() will setup the types of
                  between device transfers now and use them when you call average().  
        !*/
        {
            using namespace ::dlib::cuda;
            accessible_groups.clear();
            epa.clear();
            if (items.size() < 1)
                return;

            scale = 1.0/items.size();

            // split item into groups of accessible devices
            std::vector<tensor*> group, unused;
            while(items.size() > 0)
            {
                group.push_back(items[0]);
                for(size_t i = 1; i < items.size(); ++i)
                {
                    if (can_access_peer(*items[0], *items[i]))
                        group.push_back(items[i]);
                    else
                        unused.push_back(items[i]);
                }
                accessible_groups.push_back(group);
                unused.swap(items);
                unused.clear();
                group.clear();
            }
            for (auto&& g : accessible_groups)
            {
                for (size_t i = 1; i < g.size(); ++i)
                {
                    epa.emplace_back(new enable_peer_access(*g[0], *g[i]));
                }
            }
        }

        size_t num_device_groups(
        ) const { return accessible_groups.size(); }
        /*!
            ensures
                - The devices given to set() are grouped together when they can directly
                  access each other using GPUDirect.  This function returns the number of
                  such groups.  For example, if all devices can directly access each other
                  then the number of groups is 1.
        !*/

        void average()
        /*!
            requires
                - All the devices have stopped writing to the tensors given to set().  So
                  you should probably call cudaDeviceSynchronize() on each of the relevant
                  devices before calling average().
            ensures
                - Computes the average of all the tensors given to set() and then sets them
                  all equal to the average.
        !*/
        {
            using namespace ::dlib::cuda;


            // First we average things within each group
            for (auto&& g : accessible_groups)
            {
                raii_set_device set_dev(*g[0]);
                if (g.size() == 1)
                    tt::affine_transform(*g[0], *g[0], scale);
                else 
                    tt::affine_transform(*g[0], *g[0], *g[1], scale, scale);

                for (size_t i = 2; i < g.size(); ++i)
                    tt::affine_transform(*g[0], *g[0], *g[i], 1, scale);
            }

            if (accessible_groups.size() > 1)
            {
                tensor& total_avg = *accessible_groups[0][0];
                raii_set_device set_dev(total_avg);
                accum_buffer.copy_size(total_avg);
                // now we need to average things across groups
                for (size_t i = 1; i < accessible_groups.size(); ++i)
                {
                    memcpy(accum_buffer, *accessible_groups[i][0]);
                    tt::add(total_avg, total_avg, accum_buffer);
                }

                // Now total_avg has the final average in it.  So we need to send
                // copies of it back to each of the groups.
                for (size_t i = 1; i < accessible_groups.size(); ++i)
                {
                    memcpy(*accessible_groups[i][0], total_avg);
                }
            }


            // Now propagate averages back out to each element using point to point
            // communication inside a group.
            for (auto&& g : accessible_groups)
            {
                raii_set_device set_dev(*g[0]);
                for (size_t i = 1; i < g.size(); ++i)
                    memcpy(*g[i], *g[0]); 
            }
        }

    private:
        std::vector<std::unique_ptr<::dlib::cuda::enable_peer_access>> epa;
        std::vector<std::vector<tensor*>> accessible_groups;
        float scale;

        resizable_tensor accum_buffer;
    };

// ----------------------------------------------------------------------------------------

    void copy_tensor(
            bool add_to,
            tensor& dest,
            size_t dest_k_offset,
            const tensor& src,
            size_t src_k_offset,
            size_t count_k
    );
    /*!
        requires
            - dest.nc() == src.nc()
            - dest.nr() == src.nr()
            - dest.num_samples() == src.num_samples()
            - dest.k() - dest_k_offset >= count_k
            - src.k() - src_k_offset >= count_k
            - is_same_object(dest,src) == false
            - The memory areas of src and dest do not overlap.
        ensures
            - if (add_to) then
                - performs: dest[i, k + dest_k_offset, r, c] += src[i, k + src_k_offset, r, c], where k in [0..count_k]
                  i.e., adds content of each sample from src in to corresponding place of sample at dest.
            - else
                - performs: dest[i, k + dest_k_offset, r, c]  = src[i, k + src_k_offset, r, c], where k in [0..count_k]
                  i.e., copies content of each sample from src in to corresponding place of sample at dest.
    !*/

// ----------------------------------------------------------------------------------------

}}

#ifdef NO_MAKEFILE
#include "tensor_tools.cpp"
#endif

#endif // DLIB_TeNSOR_TOOLS_H_


