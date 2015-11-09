// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_TeNSOR_TOOLS_H_
#define DLIB_TeNSOR_TOOLS_H_

#include "tensor.h"

namespace dlib
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
    public:
    };

// ----------------------------------------------------------------------------------------

    void multiply (
        tensor& dest,
        const tensor& src
    );
    /*!
        requires
            - have_same_dimensions(dest,src) == true
        ensures
            - #dest == dest*src 
              That is, for all valid i:
                #dest.host()[i] == dest.host()[i]*src.host()[i]
    !*/

// ----------------------------------------------------------------------------------------

    void affine_transform(
        resizable_tensor& dest,
        const tensor& src,
        const float A,
        const float B
    );
    /*!
        ensures
            - have_same_dimensions(#dest,src) == true
            - #dest == A*src + B
    !*/

// ----------------------------------------------------------------------------------------

    void affine_transform(
        resizable_tensor& dest,
        const tensor& src,
        const tensor& A,
        const tensor& B
    );
    /*!
        requires
            - if (A.num_samples() == 1) then
                - B.num_samples() == 1
            - else
                - A.num_samples() == src.num_samples()
                - B.num_samples() == src.num_samples()
            - A.nr() == B.nr() == src.nr()
            - A.nc() == B.nc() == src.nc()
            - A.k()  == B.k()  == src.k()
        ensures
            - have_same_dimensions(#dest,src) == true
            - if (A.num_samples() == 1) then
                - #dest == A*src + B
                    (done for each sample in src)
            - else
                - for all valid i:
                    - #dest.host()[i] == A.host()[i]*src.host()[i] + B.host()[i]  
    !*/

// ----------------------------------------------------------------------------------------

    void batch_normalize (
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& vars,
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
        ensures
            - have_same_dimensions(#dest, src) == true
            - #means.num_samples() == 1
            - #vars.num_samples() == 1
            - means.nr() == vars.nr() == src.nr()
            - means.nc() == vars.nc() == src.nc()
            - means.k()  == vars.k()  == src.k()
            - #src == the batch normalized version of src.
            - #means == the mean values of the contents of src.
            - #vars == the variance values of the contents of src.
    !*/

// ----------------------------------------------------------------------------------------

    void batch_normalize_gradient (
        const tensor& gradient_input,
        const tensor& means,
        const tensor& vars,
        const tensor& src,
        const tensor& gamma,
        tensor& src_grad,
        tensor& gamma_grad, 
        tensor& beta_grad 
    );
    /*!
        requires
            - vars and means should be the output of a call to
              batch_normalize(dest,means,vars,src,gamma,beta)
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
            - have_same_dimensions(vars, gamma) == true
        ensures
            - Let f(src,gamma,beta) == dot(gradient_input, dest output of
              batch_normalize(dest,means,vars,src,gamma,beta))
            - Adds the gradient of f() with respect to src to #src_grad.
            - Adds the gradient of f() with respect to gamma to #gamma_grad.
            - Adds the gradient of f() with respect to beta to #beta_grad.
    !*/

    void batch_normalize_conv (
        resizable_tensor& dest,
        resizable_tensor& means,
        resizable_tensor& vars,
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
        ensures
            - have_same_dimensions(#dest, src) == true
            - #means.num_samples()==means.nr()==means.nc() == 1
            - #vars.num_samples() ==vars.nr() ==vars.nc() == 1
            - means.k()  == vars.k()  == src.k()
            - #src == the batch normalized version of src.
            - #means == the mean values of the contents of src.
            - #vars == the variance values of the contents of src.
    !*/

    void batch_normalize_conv_gradient (
        const tensor& gradient_input,
        const tensor& means,
        const tensor& vars,
        const tensor& src,
        const tensor& gamma,
        tensor& src_grad,
        tensor& gamma_grad, 
        tensor& beta_grad 
    );
    /*!
        requires
            - vars and means should be the output of a call to
              batch_normalize_conv(dest,means,vars,src,gamma,beta)
            - have_same_dimensions(gradient_input, src) == true
            - have_same_dimensions(src, src_grad) == true
            - src.num_samples() > 1
            - gamma.num_samples()==gamma.nr()==gamma.nc() == 1
            - have_same_dimensions(gamma, gamma_grad) == true
            - have_same_dimensions(gamma, beta_grad) == true
            - gamma.k()  == src.k()
            - have_same_dimensions(means, gamma) == true
            - have_same_dimensions(vars, gamma) == true
        ensures
            - Let f(src,gamma,beta) == dot(gradient_input, dest output of
              batch_normalize_conv(dest,means,vars,src,gamma,beta))
            - Adds the gradient of f() with respect to src to #src_grad.
            - Adds the gradient of f() with respect to gamma to #gamma_grad.
            - Adds the gradient of f() with respect to beta to #beta_grad.
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

// ----------------------------------------------------------------------------------------

    void add(
        float beta,
        tensor& dest,
        float alpha,
        const tensor& src
    );
    /*!
        requires
            - dest.num_samples()==src.num_samples() || src.num_samples()==1
            - dest.nr()==src.nr() || src.nr()==1
            - dest.nc()==src.nc() || src.nc()==1
            - dest.k()==src.k()   || src.k()==1
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

    void add_conv_bias_gradient (
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
        ensures
            - let BIAS be a tensor with all dimensions equal to 1 except for k which is >= 1.
            - let OUT be the output of add(1,OUT,1,BIAS)
            - let f(gradient_input,BIAS) == dot(gradient_input,OUT)
            - Then this function computes the gradient of f() with respect to BIAS and adds
              it to grad.
    !*/

// ----------------------------------------------------------------------------------------

    class conv
    {
    public:
        conv(const conv&) = delete;
        conv& operator=(const conv&) = delete;

        conv();

        void clear(
        );

        void setup(
            const tensor& data,
            const tensor& filters,
            int stride_y,
            int stride_x
        );
        /*!
            requires
                - filters.k() == data.k()
                    - stride_y > 0
                    - stride_x > 0
        !*/

        ~conv (
        );

        void operator() (
            resizable_tensor& output,
            const tensor& data,
            const tensor& filters
        );
        /*!
            requires
                - The dimensions of data and filters are the same as the ones given 
                  to the last call to setup().
            ensures
                - convolves filters over data.  
                    - filters contains filters.num_samples() filters. 
                    - #output.num_samples() == data.num_samples()
                    - #output.k() == filters.num_samples()
                    - #output.nr() == 1+(data.nr()-1)/stride_y
                    - #output.nc() == 1+(data.nc()-1)/stride_x
        !*/

        void get_gradient_for_data (
            const tensor& gradient_input, 
            const tensor& filters,
            tensor& data_gradient
        );
        /*!
            requires
                - filters has the same dimensions as the filters object give to the last
                  call to setup().
                - data_gradient has the same dimensions as the data object give to the last
                  call to setup().
                - gradient_input has the same dimensions as the output of operator().
            ensures
                - let OUT be the output of (*this)(OUT,data,filters).
                - let f(data,filters) == dot(OUT, gradient_input)
                - This function finds the gradient of f() with respect to data and adds
                  this gradient to data_gradient.
        !*/

        void get_gradient_for_filters (
            const tensor& gradient_input, 
            const tensor& data,
            tensor& filters_gradient
        );
        /*!
            requires
                - filters_gradient has the same dimensions as the filters object give to
                  the last call to setup().
                - data has the same dimensions as the data object give to the last call to
                  setup().
                - gradient_input has the same dimensions as the output of operator().
            ensures
                - let OUT be the output of (*this)(OUT,data,filters).
                - let f(data,filters) == dot(OUT, gradient_input)
                - This function finds the gradient of f() with respect to filters and adds
                  this gradient to filters_gradient.
        !*/

    private:

    };

// ----------------------------------------------------------------------------------------

    class max_pool
    {
        /*!
        !*/
    public:

        max_pool(const max_pool&) = delete;
        max_pool& operator=(const max_pool&) = delete;

        max_pool (
        );

        ~max_pool(
        );

        void clear(
        );

        void setup(
            int window_height,
            int window_width,
            int stride_y,
            int stride_x
        );

        void operator() (
            resizable_tensor& dest,
            const tensor& src
        );
        /*!
            ensures
                - #dest.num_samples() == src.num_samples()
                - #dest.k() == src.k()
                - #dest.nr() == src.nr()/stride_y
                - #dest.nc() == src.nc()/stride_x
                - for all valid s, k, r, and c:
                    - image_plane(#dest,s,k)(r,c) == max(subm_clipped(image_plane(src,s,k),
                                                                        r*stride_y,
                                                                        c*stride_x,
                                                                        window_height,
                                                                        window_width))
        !*/

        void get_gradient(
            const tensor& gradient_input, 
            const tensor& dest,
            const tensor& src,
            tensor& grad 
        );
        /*!
            requires
                - have_same_dimensions(gradient_input,dest) == true
                - have_same_dimensions(src,grad) == true
                - dest contains the result of calling (*this)(dest,src)
            ensures
                - Recalling that dest is the output of (*this)(dest,src),
                  let f(src) == dot(gradient_input,dest)
                - Then this function computes the gradient of f() with respect to src and
                  adds it to grad.
        !*/

        private:
    };

// ----------------------------------------------------------------------------------------

    void softmax (
        resizable_tensor& dest,
        const tensor& src
    );
    /*!
        ensures
            - have_same_dimensions(#dest, src) == true
            - Note that the softmax function is a vector valued function: 
                s(x) == exp(x)/sum(exp(x)) 
            - Computes the softmax function on src and writes the results to dest.  The
              softmax is computed per spatial location across the different channels at
              each location.  That is, softmax() outputs a new tensor, #dest, where each of
              the spatial locations in dest (i.e. image idx, row idx, and column idx)
              contains the output of s() evaluated over the channel values at each
              location.
    !*/

    void softmax_gradient (
        tensor& grad,
        const tensor& softmaxed_data,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(softmaxed_data,gradient_input) == true 
            - have_same_dimensions(softmaxed_data,grad) == true 
        ensures
            - We interpret softmaxed_data as the output of softmax(softmaxed_data,SRC) for
              some SRC tensor.  Then let f(SRC) == dot(gradient_input,softmaxed_data) Then
              this function computes the gradient of f() with respect to SRC and adds it to
              grad.
    !*/

// ----------------------------------------------------------------------------------------

    void sigmoid (
        resizable_tensor& dest,
        const tensor& src
    );
    /*!
        ensures
            - have_same_dimensions(#dest, src) == true
            - for all valid i:
                - #dest.host()[i] == 1/(1+std::exp(-src.host()[i])) 
    !*/

    void sigmoid_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& src,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(src,gradient_input) == true 
            - have_same_dimensions(src,grad) == true 
            - have_same_dimensions(src,dest) == true 
            - dest contains the result of calling sigmoid(dest,src)
        ensures
            - Recalling that dest is the output of sigmoid(dest,src),
              let f(src) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to src and
              adds it to grad.
    !*/

// ----------------------------------------------------------------------------------------

    void relu (
        resizable_tensor& dest,
        const tensor& src
    );
    /*!
        ensures
            - have_same_dimensions(#dest, src) == true
            - for all valid i:
                - #dest.host()[i] == std::max(0,src.host()[i]) 
    !*/

    void relu_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& src,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(src,gradient_input) == true 
            - have_same_dimensions(src,grad) == true 
            - have_same_dimensions(src,dest) == true 
            - dest contains the result of calling relu(dest,src)
        ensures
            - Recalling that dest is the output of relu(dest,src),
              let f(src) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to src and adds
              it to grad.
    !*/

// ----------------------------------------------------------------------------------------

    void tanh (
        resizable_tensor& dest,
        const tensor& src
    );
    /*!
        ensures
            - have_same_dimensions(#dest, src) == true
            - for all valid i:
                - #dest.host()[i] == std::tanh(src.host()[i]) 
    !*/

    void tanh_gradient (
        tensor& grad,
        const tensor& dest,
        const tensor& src,
        const tensor& gradient_input
    );
    /*!
        requires
            - have_same_dimensions(src,gradient_input) == true 
            - have_same_dimensions(src,grad) == true 
            - have_same_dimensions(src,dest) == true 
            - dest contains the result of calling tanh(dest,src)
        ensures
            - Recalling that dest is the output of tanh(dest,src),
              let f(src) == dot(gradient_input,dest)
            - Then this function computes the gradient of f() with respect to src and adds
              it to grad.
    !*/

// ----------------------------------------------------------------------------------------

    } 
}

#endif // DLIB_TeNSOR_TOOLS_H_


