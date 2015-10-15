// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_SOLVERS_H_
#define DLIB_DNn_SOLVERS_H_

#include "solvers_abstract.h"
#include "tensor.h"
#include <iostream>

namespace dlib
{
    class sgd
    {
    public:

        sgd(
            float learning_rate_ = 0.001,
            float weight_decay_ = 0.0005,
            float momentum_ = 0.9 
        ) 
        { 
            weight_decay = weight_decay_;
            learning_rate = learning_rate_;
            momentum = momentum_;
        }

        float get_momentum (
        ) const { return momentum; }

        float get_weight_decay (
        ) const { return weight_decay; }

        float get_learning_rate (
        ) const { return learning_rate; }

        template <typename LAYER_DETAILS>
        void operator() (
            LAYER_DETAILS& l, 
            const tensor& params_grad
        )
        {
            DLIB_CASSERT(l.get_layer_params().size() != 0,"");
            if (v.size() != 0)
                v = momentum*v - weight_decay*learning_rate*mat(l.get_layer_params()) - learning_rate*mat(params_grad);
            else
                v =            - weight_decay*learning_rate*mat(l.get_layer_params()) - learning_rate*mat(params_grad);

            l.get_layer_params() += v;
        }

        friend void serialize(const sgd& item, std::ostream& out)
        {
            serialize("sgd", out);
            serialize(item.v, out);
            serialize(item.weight_decay, out);
            serialize(item.learning_rate, out);
            serialize(item.momentum, out);
        }

        friend void deserialize(sgd& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "sgd")
                throw serialization_error("Unexpected version found while deserializing dlib::sgd.");
            deserialize(item.v, in);
            deserialize(item.weight_decay, in);
            deserialize(item.learning_rate, in);
            deserialize(item.momentum, in);
        }

    private:
        matrix<float> v;
        float weight_decay;
        float learning_rate;
        float momentum;
    };


}

#endif // DLIB_DNn_SOLVERS_H_

