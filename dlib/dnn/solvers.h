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
            float weight_decay_,
            float momentum_
        )
        {
            weight_decay = weight_decay_;
            momentum = momentum_;
        }

        sgd(
        ) : sgd(0.0005, 0.9)
        {
        }

        float get_momentum (
        ) const { return momentum; }

        float get_weight_decay (
        ) const { return weight_decay; }

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
        )
        {
            const tensor& params = l.get_layer_params();

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
            serialize("sgd2", out);
            serialize(item.v, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum, out);
        }

        friend void deserialize(sgd& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "sgd2")
                throw serialization_error("Unexpected version found while deserializing dlib::sgd.");
            deserialize(item.v, in);
            deserialize(item.weight_decay, in);
            deserialize(item.momentum, in);
        }

    private:
        resizable_tensor v;
        float weight_decay;
        float momentum;
    };

// ----------------------------------------------------------------------------------------

    class adam
    {
    public:

        adam(
            float weight_decay_,
            float momentum1_,
            float momentum2_
        )
        {
            weight_decay = weight_decay_;
            momentum1 = momentum1_;
            momentum2 = momentum2_;
            t = 0;
        }

        adam(
        ) : adam(0.0005, 0.9, 0.999)
        {}

        float get_momentum1 (
        ) const { return momentum1; }

        float get_momentum2 (
        ) const { return momentum2; }

        float get_weight_decay (
        ) const { return weight_decay; }

        template <typename layer_type>
        const tensor& operator() (
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad
        )
        {
            const tensor& params = l.get_layer_params();
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
            serialize("adam2", out);
            serialize(item.m, out);
            serialize(item.v, out);
            serialize(item.s, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum1, out);
            serialize(item.momentum2, out);
            serialize(item.t, out);
        }

        friend void deserialize(adam& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "adam2")
                throw serialization_error("Unexpected version found while deserializing dlib::adam.");
            deserialize(item.m, in);
            deserialize(item.v, in);
            deserialize(item.s, in);
            deserialize(item.weight_decay, in);
            deserialize(item.momentum1, in);
            deserialize(item.momentum2, in);
            deserialize(item.t, in);
        }

    private:
        resizable_tensor m;
        resizable_tensor v;
        resizable_tensor s;
        float weight_decay;
        float momentum1;
        float momentum2;
        float t;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_H_

