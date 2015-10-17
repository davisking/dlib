// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNN_CuDA_H_
#define DLIB_DNN_CuDA_H_

#ifdef DLIB_USE_CUDA

#include "tensor.h"

// TODO, remove this cruft
void hello_cuda();

namespace dlib
{
    namespace cuda 
    {

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
                - A.num_samples() == 1
                - B.num_samples() == 1
                - A.nr() == B.nr() == src.nr()
                - A.nc() == B.nc() == src.nc()
                - A.k()  == B.k()  == src.k()
            ensures
                - have_same_dimensions(#dest,src) == true
                - #dest == A*src + B
                  (done for each sample in src)
        !*/

    // -----------------------------------------------------------------------------------

    // TODO, add versions of batch_normalize() that output the gradients.
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

    // -----------------------------------------------------------------------------------

        class dropout
        {
            /*!
            !*/
        public:

            // not copyable
            dropout(const dropout&) = delete;
            dropout& operator=(const dropout&) = delete;
            // but is movable
            dropout(dropout&&) = default;
            dropout& operator=(dropout&&) = default;

            dropout(float drop_rate = 0.5);
            dropout(float drop_rate, int seed);

            void operator() (
                resizable_tensor& dest,
                resizable_tensor& random_mask,
                const tensor& src
            );
            /*!
                ensures
                    - have_same_dimensions(src, #dest) == true
                    - have_same_dimensions(src, #random_mask) == true
            !*/

            void get_gradient(
                const tensor& gradient_input, 
                const tensor& random_mask,
                tensor& grad 
            );
            /*!
                requires
                    - have_same_dimensions(gradient_input, random_mask) == true
                    - have_same_dimensions(gradient_input, grad) == true
                ensures
                    - let OUT and MASK be the output of (*this)(OUT,MASK,src)
                    - let f(src) == dot(gradient_input,OUT)
                    - Then this function computes the gradient of f() with respect to src
                      and adds it to grad.
            !*/
        };

    // -----------------------------------------------------------------------------------

    } 
}

#endif // DLIB_USE_CUDA

#endif // DLIB_DNN_CuDA_H_

