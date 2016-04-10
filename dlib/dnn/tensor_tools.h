// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TeNSOR_TOOLS_H_
#define DLIB_TeNSOR_TOOLS_H_

#include "tensor.h"
#include "cudnn_dlibapi.h"
#include "cublas_dlibapi.h"
#include "curand_dlibapi.h"
#include "cpu_dlib.h"
#include "cuda_dlib.h"
#include "../rand.h"

namespace dlib
{
    bool dnn_prefer_fastest_algorithms();
    void set_dnn_prefer_fastest_algorithms();
    void set_dnn_prefer_smallest_algorithms();
}

namespace dlib { namespace tt
{

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
            float mean,
            float stddev
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
    !*/

    void multiply_conv (
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
            - #dest == A*src1 + src2*B + C
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
            - #dest == A*src1 + src2*B + src3*C + D
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
        ensures
            - This function implements the ADAM parameter update method described in the paper:
                Kingma, Diederik P., and Jimmy Ba Adam. "A method for stochastic
                optimization." International Conference on Learning Representation. 2015.
              Specifically, it implements the method shown as Algorithm 1.
            - #s is the update vector that should be added to the parameters.
    !*/

// ----------------------------------------------------------------------------------------

    const double BATCH_NORM_EPS = 0.00001;

    void batch_normalize_inference (
        resizable_tensor& dest,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta,
        const tensor& running_means,
        const tensor& running_variances
    );
    /*!
        requires
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
                dest = gamma*(src-running_means)/sqrt(running_variances+BATCH_NORM_EPS) + beta
              Note that it does it in a pointwise fashion over the samples in src.
    !*/

    void batch_normalize (
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
            - invstds and means should be the output of a call to
              batch_normalize(dest,means,invstds,src,gamma,beta)
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
              batch_normalize(dest,means,invstds,src,gamma,beta))
            - Adds the gradient of f() with respect to src to #src_grad.
            - Assigns the gradient of f() with respect to gamma to #gamma_grad.
            - Assigns the gradient of f() with respect to beta to #beta_grad.
    !*/

// ----------------------------------------------------------------------------------------

    void batch_normalize_conv_inference (
        resizable_tensor& dest,
        const tensor& src,
        const tensor& gamma, 
        const tensor& beta,
        const tensor& running_means,
        const tensor& running_variances
    );
    /*!
        requires
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
                dest = gamma*(src-running_means)/sqrt(running_variances+BATCH_NORM_EPS) + beta
              Note that it does this in a pointwise fashion over the samples, rows, and
              columns in src.
    !*/

    void batch_normalize_conv (
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
            - invstds and means should be the output of a call to
              batch_normalize_conv(dest,means,invstds,src,gamma,beta)
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
              batch_normalize_conv(dest,means,invstds,src,gamma,beta))
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
            resizable_tensor& output,
            const tensor& data,
            const tensor& filters,
            int stride_y,
            int stride_x
        ) { impl(output,data,filters,stride_y,stride_x); }
        /*!
            requires
                - stride_y > 0
                - stride_x > 0
                - is_same_object(output,data) == false
                - is_same_object(output,filters) == false
                - filters.k() == data.k()
            ensures
                - convolves filters over data.  
                - filters contains filters.num_samples() filters. 
                - #output.num_samples() == data.num_samples()
                - #output.k() == filters.num_samples()
                - #output.nr() == 1+(data.nr()-filters.nr()%2)/stride_y
                - #output.nc() == 1+(data.nc()-filters.nc()%2)/stride_x
        !*/

        void get_gradient_for_data (
            const tensor& gradient_input, 
            const tensor& filters,
            tensor& data_gradient
        ) { impl.get_gradient_for_data(gradient_input,filters,data_gradient); }
        /*!
            requires
                - filters has the same dimensions as the filters object given to the last
                  call to operator().
                - data_gradient has the same dimensions as the data object given to the last
                  call to operator().
                - gradient_input has the same dimensions as the last output of operator().
                - is_same_object(data_gradient,filters) == false
                - is_same_object(data_gradient,gradient_input) == false
            ensures
                - let OUT be the output of (*this)(OUT,data,filters,sx,sy).
                - let f(data,filters) == dot(OUT, gradient_input)
                - This function finds the gradient of f() with respect to data and adds
                  this gradient to data_gradient.
        !*/

        void get_gradient_for_filters (
            const tensor& gradient_input, 
            const tensor& data,
            tensor& filters_gradient
        ) { impl.get_gradient_for_filters(gradient_input,data,filters_gradient); }
        /*!
            requires
                - filters_gradient has the same dimensions as the filters object given to
                  the last call to operator().
                - data has the same dimensions as the data object given to the last call to
                  operator().
                - gradient_input has the same dimensions as the last output of operator().
                - is_same_object(filters_gradient,data) == false
                - is_same_object(filters_gradient,gradient_input) == false
            ensures
                - let OUT be the output of (*this)(OUT,data,filters,sx,sy).
                - let f(data,filters) == dot(OUT, gradient_input)
                - This function finds the gradient of f() with respect to filters and assigns 
                  this gradient to filters_gradient.
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
            int stride_x
        ) { impl.setup_max_pooling(window_height, window_width, stride_y, stride_x); }
        /*!
            requires
                - window_height > 0
                - window_width > 0
                - stride_y > 0
                - stride_x > 0
            ensures
                - When you call operator() it will do max pooling with the given
                  parameters.
        !*/

        void setup_avg_pooling(
            int window_height,
            int window_width,
            int stride_y,
            int stride_x
        ) { impl.setup_avg_pooling(window_height, window_width, stride_y, stride_x); }
        /*!
            requires
                - window_height > 0
                - window_width > 0
                - stride_y > 0
                - stride_x > 0
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
            ensures
                - #dest.num_samples() == src.num_samples()
                - #dest.k() == src.k()
                - #dest.nr() == 1+(src.nr()-window_height%2)/stride_y
                - #dest.nc() == 1+(src.nc()-window_width%2)/stride_x
                - for all valid s, k, r, and c:
                    - if (does_max_pooling()) then
                        - image_plane(#dest,s,k)(r,c) == max(subm_clipped(image_plane(src,s,k),
                                                                      centered_rect(c*stride_x,
                                                                                    r*stride_y,
                                                                                    window_width,
                                                                                    window_height)))
                    - else
                        - image_plane(#dest,s,k)(r,c) == mean(subm_clipped(image_plane(src,s,k),
                                                                      centered_rect(c*stride_x,
                                                                                    r*stride_y,
                                                                                    window_width,
                                                                                    window_height)))
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
            - is_same_object(grad, dest)==false
        ensures
            - We interpret dest as the output of softmax(dest,SRC) for some SRC tensor.
              Then let f(SRC) == dot(gradient_input,dest) Then this function computes the
              gradient of f() with respect to SRC and adds it to grad.
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
            - is_same_object(grad,dest) == false
        ensures
            - Recalling that dest is the output of sigmoid(dest,SRC) for some SRC tensor,
              let f(SRC) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to SRC and
              assigns it to grad.
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
            - is_same_object(grad,dest) == false
        ensures
            - Recalling that dest is the output of relu(dest,SRC) for some SRC tensor,
              let f(SRC) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to SRC and
              assigns it to grad.
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
            - is_same_object(grad,dest) == false
        ensures
            - Recalling that dest is the output of tanh(dest,SRC) for some SRC tensor,
              let f(SRC) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to SRC and
              assigns it to grad.
            - This function supports in-place operation, i.e. having
              is_same_object(grad, gradient_input)==true
    !*/

// ----------------------------------------------------------------------------------------

}}

#ifdef NO_MAKEFILE
#include "tensor_tools.cpp"
#endif

#endif // DLIB_TeNSOR_TOOLS_H_


