// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_H_
#define DLIB_DNN_CuDA_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"

namespace dlib
{
    namespace cuda 
    {

    // -----------------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------------------

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

    // -----------------------------------------------------------------------------------

        void batch_normalize (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
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
                - #invstds.num_samples() == 1
                - means.nr() == invstds.nr() == src.nr()
                - means.nc() == invstds.nc() == src.nc()
                - means.k()  == invstds.k()  == src.k()
                - #src == the batch normalized version of src.
                - #means == the mean values of the contents of src.
                - #invstds == 1/(the standard deviation values of the contents of src).
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
                - Adds the gradient of f() with respect to gamma to #gamma_grad.
                - Adds the gradient of f() with respect to beta to #beta_grad.
        !*/

        void batch_normalize_conv (
            resizable_tensor& dest,
            resizable_tensor& means,
            resizable_tensor& invstds,
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
                - #invstds.num_samples() ==invstds.nr() ==invstds.nc() == 1
                - means.k()  == invstds.k()  == src.k()
                - #src == the batch normalized version of src.
                - #means == the mean values of the contents of src.
                - #invstds == 1/(the standard deviation values of the contents of src).
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
                - Sets all elements of data to 1 or 0 depending on if they are above or
                  below the given threshold.  Specifically, for all valid i:
                    - #data.host()[i] == data.host()[i]>thresh ? 1 : 0
        !*/

    // ------------------------------------------------------------------------------------

    } 
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDA_H_

