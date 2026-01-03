// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_SOLVERS_H_
#define DLIB_DNn_SOLVERS_H_

#include "solvers_abstract.h"
#include "../cuda/tensor.h"
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
        ) : sgd(0.0005f, 0.9f)
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
        ) : adam(0.0005f, 0.9f, 0.999f)
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

    /*!
        AdamW optimizer with decoupled weight decay regularization.

        This optimizer implements the AdamW algorithm from "Decoupled Weight Decay
        Regularization" (Loshchilov & Hutter, ICLR 2019). Unlike standard Adam,
        AdamW decouples the weight decay from the gradient-based optimization step,
        leading to better generalization and easier hyperparameter tuning.

        THEORETICAL FOUNDATION:
            Standard Adam with L2 regularization computes:
                theta_t = theta_{t-1} - alpha * m_hat_t / sqrt(v_hat_t + epsilon)
                where gradients include the L2 regularization term

            AdamW decouples weight decay and computes:
                m_t = beta1 * m_{t-1} + (1-beta1) * gradient_L
                v_t = beta2 * v_{t-1} + (1-beta2) * (gradient_L)^2
                theta_t = theta_{t-1} - alpha * (m_hat_t/sqrt(v_hat_t) + lambda*theta_{t-1})

            This formulation makes the optimal weight decay factor independent of
            the learning rate, improving generalization especially for long training runs.

        IMPLEMENTATION STRATEGY:
            1. Compute standard Adam update with weight_decay = 0 (decoupled)
            2. Explicitly apply weight decay: update = update - lr * wd * params
            3. The update is then added to parameters by the trainer

        KEY DIFFERENCES FROM ADAM:
            - Weight decay is applied directly to parameters (multiplicative)
            - Weight decay does not interact with adaptive learning rates
            - Better hyperparameter independence (learning rate vs weight decay)
            - Superior generalization on image classification and NLP tasks

        CONSTRUCTOR PARAMETERS:
            - weight_decay: Decoupled weight decay coefficient (default: 0.01)
                           Typical range: 0.0001 to 0.1
                           Higher values = stronger regularization
            - momentum1 (beta1): Exponential decay rate for first moment (default: 0.9)
                                Controls the momentum of gradient moving average
            - momentum2 (beta2): Exponential decay rate for second moment (default: 0.999)
                                Controls the momentum of squared gradient moving average

        REFERENCES:
            - Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization"
              ICLR 2019. https://arxiv.org/abs/1711.05101
            - Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization"
              ICLR 2015. https://arxiv.org/abs/1412.6980

        NOTE: AdamW is the standard optimizer for modern transformer models including
              GPT, BERT, LLaMA, Mistral, Qwen, DeepSeek, and other large language models.
              It consistently outperforms standard Adam with L2 regularization.
    !*/
    class adamw
    {
    public:

        explicit adamw(
            float weight_decay_ = 0.01f,
            float momentum1_ = 0.9f,
            float momentum2_ = 0.999f
        )
        {
            weight_decay = weight_decay_;
            momentum1 = momentum1_;
            momentum2 = momentum2_;
            t = 0;
        }

        float get_momentum1() const { return momentum1; }
        float get_momentum2() const { return momentum2; }
        float get_weight_decay() const { return weight_decay; }

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

            // Step 1: compute standard Adam update with decoupled weight decay (wd = 0)
            // This populates 's' with the adaptive gradient step: -alpha * m_hat_t / sqrt(v_hat_t)
            // By passing weight_decay = 0, we decouple the regularization from the adaptive update
            tt::compute_adam_update(0, params.size(), s, m, v, t,
                learning_rate * get_learning_rate_multiplier(l),
                0, // Critical: weight_decay = 0 for decoupled regularization
                momentum1, momentum2, params, params_grad);

            // Step 2: apply decoupled weight decay explicitly
            // Formula: s = s - alpha * lambda * theta_{t-1}
            // This implements the AdamW update: theta_t = theta_{t-1} - alpha * (m_hat_t/sqrt(v_hat_t) + lambda * theta_{t-1})
            const double lr = learning_rate * get_learning_rate_multiplier(l);
            const double wd = weight_decay * get_weight_decay_multiplier(l);

            if (wd != 0)
            {
                // Compute: s = s + params * (-lr * wd)
                tt::affine_transform(s, s, params, 1.0, -lr * wd);
            }

            return s;
        }

        template <unsigned long N>
        const tensor& operator() (
            const float learning_rate,
            const fc_<N, FC_HAS_BIAS>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.get_num_outputs());
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
            const con_<_num_filters, _nr, _nc, _stride_y, _stride_x, _padding_y, _padding_x>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.num_filters());
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
            const cont_<_num_filters, _nr, _nc, _stride_y, _stride_x, _padding_y, _padding_x>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() - l.num_filters());
            return s;
        }

        template < layer_mode mode >
        const tensor& operator() (
            const float learning_rate,
            const bn_<mode>& l,
            const tensor& params_grad
            )
        {
            update_considering_bias(learning_rate, l, params_grad, params_grad.size() / 2);
            return s;
        }

        friend void serialize(const adamw& item, std::ostream& out)
        {
            serialize("adamw", out);
            serialize(item.m, out);
            serialize(item.v, out);
            serialize(item.s, out);
            serialize(item.weight_decay, out);
            serialize(item.momentum1, out);
            serialize(item.momentum2, out);
            serialize(item.t, out);
        }

        friend void deserialize(adamw& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "adamw")
                throw serialization_error("Unexpected version found while deserializing dlib::adamw.");
            deserialize(item.m, in);
            deserialize(item.v, in);
            deserialize(item.s, in);
            deserialize(item.weight_decay, in);
            deserialize(item.momentum1, in);
            deserialize(item.momentum2, in);
            deserialize(item.t, in);
        }

        friend std::ostream& operator<< (std::ostream& out, const adamw& item)
        {
            out << "adamw: weight_decay=" << item.get_weight_decay()
                << ", momentum1=" << item.get_momentum1()
                << ", momentum2=" << item.get_momentum2();
            return out;
        }

    private:

        /*!
            Updates parameters that may have different learning rate and weight decay
            multipliers for weights vs biases (e.g., fully connected and convolutional layers).

            BIAS HANDLING:
                Most layers separate weights and biases:
                - Weights: indices [0, bias_offset)
                - Biases: indices [bias_offset, end)

                Different multipliers may apply to each section:
                - bias_learning_rate_multiplier (typically 1.0 or 2.0)
                - bias_weight_decay_multiplier (typically 0.0 - no decay on biases)

            PARAMETERS:
                - learning_rate: base learning rate from trainer
                - l: layer containing parameters and multiplier settings
                - params_grad: gradient tensor
                - bias_offset: index where biases start in the parameter tensor
        !*/
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

            // Step 1: compute adaptive gradient update with decoupled weight decay
            if (l.get_bias_learning_rate_multiplier() == 1)
            {
                // Simple case: uniform learning rate for all parameters
                tt::compute_adam_update(0, params.size(), s, m, v, t,
                    learning_rate * get_learning_rate_multiplier(l),
                    0, // Decoupled: weight_decay = 0 in Adam computation
                    momentum1, momentum2, params, params_grad);
            }
            else
            {
                // Complex case: different learning rates for weights and biases

                // Process weights: indices [0, bias_offset)
                tt::compute_adam_update(0, bias_offset, s, m, v, t,
                    learning_rate * get_learning_rate_multiplier(l),
                    0, // Decoupled weight decay
                    momentum1, momentum2, params, params_grad);

                // Process biases: indices [bias_offset, end)
                // Apply bias learning rate multiplier
                tt::compute_adam_update(bias_offset, params.size(), s, m, v, t,
                    learning_rate * get_learning_rate_multiplier(l) * l.get_bias_learning_rate_multiplier(),
                    0, // Decoupled weight decay
                    momentum1, momentum2, params, params_grad);
            }

            // Step 2: apply decoupled weight decay
            // Formula: s = s - lr * wd * params
            // This is applied separately to weights and biases because they may have
            // different weight decay multipliers
            double lr = learning_rate * get_learning_rate_multiplier(l);
            double wd = weight_decay * get_weight_decay_multiplier(l);

            if (l.get_bias_learning_rate_multiplier() == 1 && l.get_bias_weight_decay_multiplier() == 1)
            {
                // Simple case: uniform weight decay for all parameters
                if (wd != 0)
                    tt::affine_transform(s, s, params, 1.0, -lr * wd);
            }
            else
            {
                // Complex case: different weight decay for weights vs biases

                // Apply weight decay to weights: indices [0, bias_offset)
                // Computation: s[i] = 1.0 * s[i] + (-lr * wd) * params[i] + 0.0 * params[i]
                // The third source (params) is not used since C = 0.0
                if (wd != 0)
                {
                    tt::affine_transform_range(0, bias_offset,
                        s,          // dest
                        s,          // src1 (A coefficient)
                        params,     // src2 (B coefficient) 
                        params,     // src3 (C coefficient = 0, so this is unused)
                        1.0,        // A: keep current update
                        -lr * wd,   // B: subtract weight decay term
                        0.0);       // C: ignore third source
                }

                // Apply weight decay to biases: indices [bias_offset, end)
                // Note: typically bias_weight_decay_multiplier = 0 (no regularization on biases)
                // This is a common practice in deep learning to prevent biases from becoming too small
                lr *= l.get_bias_learning_rate_multiplier();
                wd *= l.get_bias_weight_decay_multiplier();

                if (wd != 0)
                {
                    tt::affine_transform_range(bias_offset, v.size(),
                        s,
                        s,
                        params,
                        params,
                        1.0,
                        -lr * wd,
                        0.0);
                }
            }
        }

        resizable_tensor m;  // First moment estimate (exponential moving average of gradients)
        resizable_tensor v;  // Second moment estimate (exponential moving average of squared gradients)
        resizable_tensor s;  // Parameter update computed by the optimizer
        float weight_decay;  // Weight decay coefficient (lambda in the paper)
        float momentum1;     // Beta1: decay rate for first moment
        float momentum2;     // Beta2: decay rate for second moment
        float t;             // Time step counter for bias correction
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_SOLVERS_H_

