// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_SOLVERS_H_
#define DLIB_DNn_SOLVERS_H_

#include "solvers_abstract.h"
#include "tensor.h"
#include <iostream>
#include "layers.h"

namespace dlib
{
    class sgd
    {
    public:

        explicit sgd(
            float weight_decay_,
            float momentum_ = 0.9
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

            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                v.copy_size(params_grad);
                v = 0;
            }

            const double lr = learning_rate*get_learning_rate_multiplier(l);
            const double wd = weight_decay*get_weight_decay_multiplier(l);
            
            //perform: v = momentum*mat(v) - wd*lr*mat(params) - lr*mat(params_grad);
            tt::affine_transform(v, v, params, params_grad, momentum, -wd*lr, -lr);

            return v;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N,FC_HAS_BIAS>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.get_num_outputs());
            return v;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const con_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return v;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const cont_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return v;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()/2);
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

        friend std::ostream& operator<< (std::ostream& out, const sgd& item)
        {
            out << "sgd: weight_decay="<<item.get_weight_decay() << ", momentum="<<item.get_momentum(); 
            return out;
        }

    private:

        template <typename layer_type> 
        void update_considering_bias(
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad,
            unsigned long bias_offset
        )
        {
            const tensor& params = l.get_layer_params();

            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                v.copy_size(params_grad);
                v = 0;
            }

            double lr = learning_rate*get_learning_rate_multiplier(l);
            double wd = weight_decay*get_weight_decay_multiplier(l);
            
            //perform: v = momentum*mat(v) - wd*lr*mat(params) - lr*mat(params_grad);

            if (l.get_bias_learning_rate_multiplier() == 1 && l.get_bias_weight_decay_multiplier() == 1)
            {
                tt::affine_transform(v, v, params, params_grad, momentum, -wd*lr, -lr);
            }
            else
            {

                tt::affine_transform_range(0, bias_offset, v, v, params, params_grad, momentum, -wd*lr, -lr);

                // now update the biases but apply their multipliers
                lr *= l.get_bias_learning_rate_multiplier();
                wd *= l.get_bias_weight_decay_multiplier();
                tt::affine_transform_range(bias_offset, v.size(), v, v, params, params_grad, momentum, -wd*lr, -lr);
            }
        }

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
            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
            }

            ++t;

            
            tt::compute_adam_update(0, params.size(), s, m, v, t,
                learning_rate*get_learning_rate_multiplier(l),
                weight_decay*get_weight_decay_multiplier(l), 
                momentum1, momentum2, params, params_grad);

            return s;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N,FC_HAS_BIAS>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.get_num_outputs());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const con_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return s;
        }

        template <
            long _num_filters,
            long _nr,
            long _nc,
            int _stride_y,
            int _stride_x,
            int _padding_y,
            int _padding_x
            >
        const tensor& operator() (
            const float learning_rate,
            const cont_<_num_filters,_nr,_nc,_stride_y,_stride_x,_padding_y,_padding_x>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()-l.num_filters());
            return s;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
        )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size()/2);
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

        friend std::ostream& operator<< (std::ostream& out, const adam& item)
        {
            out << "adam: weight_decay="<<item.get_weight_decay() << ", momentum1="<<item.get_momentum1() << ", momentum2="<<item.get_momentum2(); 
            return out;
        }

    private:

        template <typename layer_type> 
        void update_considering_bias(
            const float learning_rate,
            const layer_type& l,
            const tensor& params_grad,
            unsigned long bias_offset
        )
        {
            const tensor& params = l.get_layer_params();
            DLIB_CASSERT(params.size() != 0);
            if (v.size() == 0)
            {
                m.copy_size(params_grad);
                m = 0;
                v.copy_size(params_grad);
                v = 0;
                s.copy_size(params_grad);
            }


            ++t;

            if (l.get_bias_learning_rate_multiplier() == 1 && l.get_bias_weight_decay_multiplier() == 1)
            {
                tt::compute_adam_update(0, params.size(), s, m, v, t,
                    learning_rate*get_learning_rate_multiplier(l),
                    weight_decay*get_weight_decay_multiplier(l), 
                    momentum1, momentum2, params, params_grad);
            }
            else
            {
                tt::compute_adam_update(0, bias_offset, s, m, v, t,
                    learning_rate*get_learning_rate_multiplier(l),
                    weight_decay*get_weight_decay_multiplier(l), 
                    momentum1, momentum2, params, params_grad);

                tt::compute_adam_update(bias_offset, params.size(), s, m, v, t,
                    learning_rate*get_learning_rate_multiplier(l)*l.get_bias_learning_rate_multiplier(),
                    weight_decay*get_weight_decay_multiplier(l)*l.get_bias_weight_decay_multiplier(), 
                    momentum1, momentum2, params, params_grad);
            }
        }
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

