// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_SOLVERS_H_
#define DLIB_DNn_SOLVERS_H_

#include "tensor.h"
#include <iostream>

namespace dlib
{
    /*
        class EXAMPLE_SOLVER 
        {
        };
    */

    struct sgd
    {

        matrix<float> v;
        float weight_decay;
        float eps;
        float momentum;
        sgd(double eps_ = 0.001) 
        { 
            weight_decay = 0.0005;
            eps = eps_;
            //eps = 0.001;
            momentum = 0.9;
        }

        template <typename layer_type>
        void operator() (layer_type& l, const tensor& params_grad)
        /*!
            requires
                - l.get_layer_params().size() != 0
                - l.get_layer_params() and params_grad have the same dimensions.
        !*/
        {
            if (v.size() != 0)
                v = momentum*v - weight_decay*eps*mat(l.get_layer_params()) - eps*mat(params_grad);
            else
                v =            - weight_decay*eps*mat(l.get_layer_params()) - eps*mat(params_grad);

            l.get_layer_params() += v;
        }
    };


}

#endif // #define DLIB_DNn_SOLVERS_H_



