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
            float learning_rate_ = 0.01,
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

        const tensor& operator() (
            const tensor& params,
            const tensor& params_grad
        )
        {
            DLIB_CASSERT(params.size() != 0,"");
            if (v.size() == 0)
            {
                v.copy_size(params_grad);
                v = 0;
            }
            
            //perform: v = momentum*mat(v) - weight_decay*learning_rate*mat(params) - learning_rate*mat(params_grad);
            tt::affine_transform(v, v, params, params_grad, 
                               momentum, -weight_decay*learning_rate, -learning_rate, 0);

            return v;
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
        resizable_tensor v;
        float weight_decay;
        float learning_rate;
        float momentum;
    };

// ----------------------------------------------------------------------------------------

    class adam 
    {
    public:

        adam(
            float learning_rate_ = 0.001,
            float weight_decay_ = 0.0005,
            float momentum1_ = 0.9, 
            float momentum2_ = 0.999 
        ) 
        { 
            weight_decay = weight_decay_;
            learning_rate = learning_rate_;
            momentum1 = momentum1_;
            momentum2 = momentum2_;
            t = 0;
        }

        float get_momentum1 (
        ) const { return momentum1; }

        float get_momentum2 (
        ) const { return momentum2; }

        float get_weight_decay (
        ) const { return weight_decay; }

        float get_learning_rate (
        ) const { return learning_rate; }

        const tensor& operator() (
            const tensor& params,
            const tensor& params_grad
        )
        {
            DLIB_CASSERT(params.size() != 0,"");
            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
            }

            ++t;
            
            tt::compute_adam_update(s, m, v, t, learning_rate, weight_decay, momentum1, momentum2, params, params_grad);

            return s;
        }

        friend void serialize(const adam& item, std::ostream& out)
        {
            serialize("adam", out);
            serialize(item.m, out);
            serialize(item.v, out);
            serialize(item.s, out);
            serialize(item.weight_decay, out);
            serialize(item.learning_rate, out);
            serialize(item.momentum1, out);
            serialize(item.momentum2, out);
            serialize(item.t, out);
        }

        friend void deserialize(adam& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "adam")
                throw serialization_error("Unexpected version found while deserializing dlib::adam.");
            deserialize(item.m, in);
            deserialize(item.v, in);
            deserialize(item.s, in);
            deserialize(item.weight_decay, in);
            deserialize(item.learning_rate, in);
            deserialize(item.momentum1, in);
            deserialize(item.momentum2, in);
            deserialize(item.t, in);
        }

    private:
        resizable_tensor m;
        resizable_tensor v;
        resizable_tensor s;
        float weight_decay;
        float learning_rate;
        float momentum1;
        float momentum2;
        float t;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_H_

