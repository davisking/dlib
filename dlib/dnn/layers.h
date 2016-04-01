// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_LAYERS_H_
#define DLIB_DNn_LAYERS_H_

#include "layers_abstract.h"
#include "tensor.h"
#include "core.h"
#include <iostream>
#include <string>
#include "../rand.h"
#include "../string.h"
#include "tensor_tools.h"
#include "../vectorstream.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class con_
    {
    public:

        con_ (
        ) : 
            _num_filters(1),
            _nr(3),
            _nc(3),
            _stride_y(1),
            _stride_x(1)
        {}

        con_(
            long num_filters_,
            long nr_,
            long nc_,
            int stride_y_ = 1,
            int stride_x_ = 1
        ) : 
            _num_filters(num_filters_), 
            _nr(nr_),
            _nc(nc_),
            _stride_y(stride_y_),
            _stride_x(stride_x_)
        {}

        long num_filters() const { return _num_filters; }
        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }

        con_ (
            const con_& item
        ) : 
            params(item.params),
            _num_filters(item._num_filters), 
            _nr(item._nr),
            _nc(item._nc),
            _stride_y(item._stride_y),
            _stride_x(item._stride_x),
            filters(item.filters),
            biases(item.biases)
        {
            // this->conv is non-copyable and basically stateless, so we have to write our
            // own copy to avoid trying to copy it and getting an error.
        }

        con_& operator= (
            const con_& item
        )
        {
            if (this == &item)
                return *this;

            // this->conv is non-copyable and basically stateless, so we have to write our
            // own copy to avoid trying to copy it and getting an error.
            params = item.params;
            _num_filters = item._num_filters;
            _nr = item._nr;
            _nc = item._nc;
            _stride_y = item._stride_y;
            _stride_x = item._stride_x;
            filters = item.filters;
            biases = item.biases;
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            long num_inputs = _nr*_nc*sub.get_output().k();
            long num_outputs = _num_filters;
            // allocate params for the filters and also for the filter bias values.
            params.set_size(num_inputs*_num_filters + _num_filters);

            dlib::rand rnd("con_"+cast_to_string(num_outputs+num_inputs));
            randomize_parameters(params, num_inputs+num_outputs, rnd);

            filters = alias_tensor(_num_filters, sub.get_output().k(), _nr, _nc);
            biases = alias_tensor(1,_num_filters);

            // set the initial bias values to zero
            biases(params,filters.size()) = 0;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            conv(output,
                sub.get_output(),
                filters(params,0),
                _stride_y,
                _stride_x);

            tt::add(1,output,1,biases(params,filters.size()));
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            conv.get_gradient_for_data (gradient_input, filters(params,0), sub.get_gradient_input());
            auto filt = filters(params_grad,0);
            conv.get_gradient_for_filters (gradient_input, sub.get_output(), filt);
            auto b = biases(params_grad, filters.size());
            tt::assign_conv_bias_gradient(b, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const con_& item, std::ostream& out)
        {
            serialize("con_", out);
            serialize(item.params, out);
            serialize(item._num_filters, out);
            serialize(item._nr, out);
            serialize(item._nc, out);
            serialize(item._stride_y, out);
            serialize(item._stride_x, out);
            serialize(item.filters, out);
            serialize(item.biases, out);
        }

        friend void deserialize(con_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "con_")
                throw serialization_error("Unexpected version found while deserializing dlib::con_.");
            deserialize(item.params, in);
            deserialize(item._num_filters, in);
            deserialize(item._nr, in);
            deserialize(item._nc, in);
            deserialize(item._stride_y, in);
            deserialize(item._stride_x, in);
            deserialize(item.filters, in);
            deserialize(item.biases, in);
        }

    private:

        resizable_tensor params;
        long _num_filters;
        long _nr;
        long _nc;
        int _stride_y;
        int _stride_x;
        alias_tensor filters, biases;

        tt::tensor_conv conv;

    };

    template <typename SUBNET>
    using con = add_layer<con_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class max_pool_
    {
    public:

        max_pool_ (
        ) : 
            _nr(3),
            _nc(3),
            _stride_y(1),
            _stride_x(1)
        {}

        max_pool_(
            long nr_,
            long nc_,
            int stride_y_ = 1,
            int stride_x_ = 1
        ) : 
            _nr(nr_),
            _nc(nc_),
            _stride_y(stride_y_),
            _stride_x(stride_x_)
        {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }

        max_pool_ (
            const max_pool_& item
        ) : 
            _nr(item._nr),
            _nc(item._nc),
            _stride_y(item._stride_y),
            _stride_x(item._stride_x)
        {
            // this->mp is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            mp.setup_max_pooling(_nr, _nc, _stride_y, _stride_x);
        }

        max_pool_& operator= (
            const max_pool_& item
        )
        {
            if (this == &item)
                return *this;

            // this->mp is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            _nr = item._nr;
            _nc = item._nc;
            _stride_y = item._stride_y;
            _stride_x = item._stride_x;

            mp.setup_max_pooling(_nr, _nc, _stride_y, _stride_x);
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
            mp.setup_max_pooling(_nr, _nc, _stride_y, _stride_x);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            mp(output, sub.get_output());
        } 

        template <typename SUBNET>
        void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            mp.get_gradient(gradient_input, computed_output, sub.get_output(), sub.get_gradient_input());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const max_pool_& item, std::ostream& out)
        {
            serialize("max_pool_", out);
            serialize(item._nr, out);
            serialize(item._nc, out);
            serialize(item._stride_y, out);
            serialize(item._stride_x, out);
        }

        friend void deserialize(max_pool_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "max_pool_")
                throw serialization_error("Unexpected version found while deserializing dlib::max_pool_.");
            deserialize(item._nr, in);
            deserialize(item._nc, in);
            deserialize(item._stride_y, in);
            deserialize(item._stride_x, in);

            item.mp.setup_max_pooling(item._nr, item._nc, item._stride_y, item._stride_x);
        }

    private:

        long _nr;
        long _nc;
        int _stride_y;
        int _stride_x;

        tt::pooling mp;
        resizable_tensor params;
    };

    template <typename SUBNET>
    using max_pool = add_layer<max_pool_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class avg_pool_
    {
    public:

        avg_pool_ (
        ) : 
            _nr(3),
            _nc(3),
            _stride_y(1),
            _stride_x(1)
        {}

        avg_pool_(
            long nr_,
            long nc_,
            int stride_y_ = 1,
            int stride_x_ = 1
        ) : 
            _nr(nr_),
            _nc(nc_),
            _stride_y(stride_y_),
            _stride_x(stride_x_)
        {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }

        avg_pool_ (
            const avg_pool_& item
        ) : 
            _nr(item._nr),
            _nc(item._nc),
            _stride_y(item._stride_y),
            _stride_x(item._stride_x)
        {
            // this->ap is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            ap.setup_avg_pooling(_nr, _nc, _stride_y, _stride_x);
        }

        avg_pool_& operator= (
            const avg_pool_& item
        )
        {
            if (this == &item)
                return *this;

            // this->ap is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            _nr = item._nr;
            _nc = item._nc;
            _stride_y = item._stride_y;
            _stride_x = item._stride_x;

            ap.setup_avg_pooling(_nr, _nc, _stride_y, _stride_x);
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
            ap.setup_avg_pooling(_nr, _nc, _stride_y, _stride_x);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            ap(output, sub.get_output());
        } 

        template <typename SUBNET>
        void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            ap.get_gradient(gradient_input, computed_output, sub.get_output(), sub.get_gradient_input());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const avg_pool_& item, std::ostream& out)
        {
            serialize("avg_pool_", out);
            serialize(item._nr, out);
            serialize(item._nc, out);
            serialize(item._stride_y, out);
            serialize(item._stride_x, out);
        }

        friend void deserialize(avg_pool_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "avg_pool_")
                throw serialization_error("Unexpected version found while deserializing dlib::avg_pool_.");
            deserialize(item._nr, in);
            deserialize(item._nc, in);
            deserialize(item._stride_y, in);
            deserialize(item._stride_x, in);

            item.ap.setup_avg_pooling(item._nr, item._nc, item._stride_y, item._stride_x);
        }

    private:

        long _nr;
        long _nc;
        int _stride_y;
        int _stride_x;

        tt::pooling ap;
        resizable_tensor params;
    };

    template <typename SUBNET>
    using avg_pool = add_layer<avg_pool_, SUBNET>;

// ----------------------------------------------------------------------------------------

    enum layer_mode
    {
        CONV_MODE = 0,
        FC_MODE = 1
    };

    class bn_
    {
    public:
        bn_() : num_updates(0), running_stats_window_size(1000), mode(FC_MODE)
        {}

        explicit bn_(layer_mode mode_) : num_updates(0), running_stats_window_size(1000), mode(mode_)
        {}

        bn_(layer_mode mode_, unsigned long window_size) : num_updates(0), running_stats_window_size(window_size), mode(mode_)
        {}

        layer_mode get_mode() const { return mode; }
        unsigned long get_running_stats_window_size () const { return running_stats_window_size; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            if (mode == FC_MODE)
            {
                gamma = alias_tensor(1,
                                sub.get_output().k(),
                                sub.get_output().nr(),
                                sub.get_output().nc());
            }
            else
            {
                gamma = alias_tensor(1, sub.get_output().k());
            }
            beta = gamma;

            params.set_size(gamma.size()+beta.size());

            gamma(params,0) = 1;
            beta(params,gamma.size()) = 0;

            running_means.copy_size(gamma(params,0));
            running_invstds.copy_size(gamma(params,0));
            running_means = 0;
            running_invstds = 1;
            num_updates = 0;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());
            if (sub.get_output().num_samples() > 1)
            {
                const double decay = 1.0 - num_updates/(num_updates+1.0);
                if (num_updates <running_stats_window_size)
                    ++num_updates;
                if (mode == FC_MODE)
                    tt::batch_normalize(output, means, invstds, decay, running_means, running_invstds, sub.get_output(), g, b);
                else 
                    tt::batch_normalize_conv(output, means, invstds, decay, running_means, running_invstds, sub.get_output(), g, b);
            }
            else // we are running in testing mode so we just linearly scale the input tensor.
            {
                if (mode == FC_MODE)
                    tt::batch_normalize_inference(output, sub.get_output(), g, b, running_means, running_invstds);
                else
                    tt::batch_normalize_conv_inference(output, sub.get_output(), g, b, running_means, running_invstds);
            }
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            auto g = gamma(params,0);
            auto g_grad = gamma(params_grad, 0);
            auto b_grad = beta(params_grad, gamma.size());
            if (mode == FC_MODE)
                tt::batch_normalize_gradient(gradient_input, means, invstds, sub.get_output(), g, sub.get_gradient_input(), g_grad, b_grad );
            else
                tt::batch_normalize_conv_gradient(gradient_input, means, invstds, sub.get_output(), g, sub.get_gradient_input(), g_grad, b_grad );
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const bn_& item, std::ostream& out)
        {
            serialize("bn_", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.beta, out);
            serialize(item.means, out);
            serialize(item.invstds, out);
            serialize(item.running_means, out);
            serialize(item.running_invstds, out);
            serialize(item.num_updates, out);
            serialize(item.running_stats_window_size, out);
            serialize((int)item.mode, out);
        }

        friend void deserialize(bn_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "bn_")
                throw serialization_error("Unexpected version found while deserializing dlib::bn_.");
            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.beta, in);
            deserialize(item.means, in);
            deserialize(item.invstds, in);
            deserialize(item.running_means, in);
            deserialize(item.running_invstds, in);
            deserialize(item.num_updates, in);
            deserialize(item.running_stats_window_size, in);
            int mode;
            deserialize(mode, in);
            item.mode = (layer_mode)mode;
        }

    private:

        friend class affine_;

        resizable_tensor params;
        alias_tensor gamma, beta;
        resizable_tensor means, running_means;
        resizable_tensor invstds, running_invstds;
        unsigned long num_updates;
        unsigned long running_stats_window_size;
        layer_mode mode;
    };

    template <typename SUBNET>
    using bn = add_layer<bn_, SUBNET>;

// ----------------------------------------------------------------------------------------

    enum fc_bias_mode{
        FC_HAS_BIAS = 0,
        FC_NO_BIAS = 1
    };

    class fc_
    {
    public:
        fc_() : num_outputs(1), num_inputs(0), bias_mode(FC_HAS_BIAS)
        {
        }

        explicit fc_(
            unsigned long num_outputs_, 
            fc_bias_mode mode = FC_HAS_BIAS
        ) : num_outputs(num_outputs_), num_inputs(0), bias_mode(mode)
        {
        }

        unsigned long get_num_outputs (
        ) const { return num_outputs; }

        fc_bias_mode get_bias_mode (
        ) const { return bias_mode; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            num_inputs = sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k();
            if (bias_mode == FC_HAS_BIAS)
                params.set_size(num_inputs+1, num_outputs);
            else
                params.set_size(num_inputs, num_outputs);

            dlib::rand rnd("fc_"+cast_to_string(num_outputs));
            randomize_parameters(params, num_inputs+num_outputs, rnd);

            weights = alias_tensor(num_inputs, num_outputs);

            if (bias_mode == FC_HAS_BIAS)
            {
                biases = alias_tensor(1,num_outputs);
                // set the initial bias values to zero
                biases(params,weights.size()) = 0;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            output.set_size(sub.get_output().num_samples(), num_outputs);

            auto w = weights(params, 0);
            tt::gemm(0,output, 1,sub.get_output(),false, w,false);
            if (bias_mode == FC_HAS_BIAS)
            {
                auto b = biases(params, weights.size());
                tt::add(1,output,1,b);
            }
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            // compute the gradient of the weight parameters.  
            auto pw = weights(params_grad, 0);
            tt::gemm(0,pw, 1,sub.get_output(),true, gradient_input,false);

            if (bias_mode == FC_HAS_BIAS)
            {
                // compute the gradient of the bias parameters.  
                auto pb = biases(params_grad, weights.size());
                tt::assign_bias_gradient(pb, gradient_input);
            }

            // compute the gradient for the data
            auto w = weights(params, 0);
            tt::gemm(1,sub.get_gradient_input(), 1,gradient_input,false, w,true);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const fc_& item, std::ostream& out)
        {
            serialize("fc_", out);
            serialize(item.num_outputs, out);
            serialize(item.num_inputs, out);
            serialize(item.params, out);
            serialize(item.weights, out);
            serialize(item.biases, out);
            serialize((int)item.bias_mode, out);
        }

        friend void deserialize(fc_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "fc_")
                throw serialization_error("Unexpected version found while deserializing dlib::fc_.");
            deserialize(item.num_outputs, in);
            deserialize(item.num_inputs, in);
            deserialize(item.params, in);
            deserialize(item.weights, in);
            deserialize(item.biases, in);
            int bmode = 0;
            deserialize(bmode, in);
            item.bias_mode = (fc_bias_mode)bmode;
        }

    private:

        unsigned long num_outputs;
        unsigned long num_inputs;
        resizable_tensor params;
        alias_tensor weights, biases;
        fc_bias_mode bias_mode;
    };

    template <typename SUBNET>
    using fc = add_layer<fc_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class dropout_
    {
    public:
        explicit dropout_(
            float drop_rate_ = 0.5
        ) :
            drop_rate(drop_rate_)
        {
            DLIB_CASSERT(0 <= drop_rate && drop_rate <= 1,"");
        }

        // We have to add a copy constructor and assignment operator because the rnd object
        // is non-copyable.
        dropout_(
            const dropout_& item
        ) : drop_rate(item.drop_rate), mask(item.mask)
        {}

        dropout_& operator= (
            const dropout_& item
        )
        {
            if (this == &item)
                return *this;

            drop_rate = item.drop_rate;
            mask = item.mask;
            return *this;
        }

        float get_drop_rate (
        ) const { return drop_rate; }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            // create a random mask and use it to filter the data
            mask.copy_size(input);
            rnd.fill_uniform(mask);
            tt::threshold(mask, drop_rate);
            tt::multiply(output, input, mask);
        } 

        void backward_inplace(
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& /*params_grad*/
        )
        {
            tt::multiply(data_grad, mask, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const dropout_& item, std::ostream& out)
        {
            serialize("dropout_", out);
            serialize(item.drop_rate, out);
            serialize(item.mask, out);
        }

        friend void deserialize(dropout_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "dropout_")
                throw serialization_error("Unexpected version found while deserializing dlib::dropout_.");
            deserialize(item.drop_rate, in);
            deserialize(item.mask, in);
        }

    private:
        float drop_rate;
        resizable_tensor mask;

        tt::tensor_rand rnd;
        resizable_tensor params; // unused
    };


    template <typename SUBNET>
    using dropout = add_layer<dropout_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class multiply_
    {
    public:
        explicit multiply_(
            float val_ = 0.5
        ) :
            val(val_)
        {
        }

        multiply_ (
            const dropout_& item
        ) : val(1-item.get_drop_rate()) {}

        float get_multiply_value (
        ) const { return val; }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::affine_transform(output, input, val, 0);
        } 

        void backward_inplace(
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& /*params_grad*/
        )
        {
            tt::affine_transform(data_grad, gradient_input, val, 0);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const multiply_& item, std::ostream& out)
        {
            serialize("multiply_", out);
            serialize(item.val, out);
        }

        friend void deserialize(multiply_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "dropout_")
            {
                // Since we can build a multiply_ from a dropout_ we check if that's what
                // is in the stream and if so then just convert it right here.
                unserialize sin(version, in);
                dropout_ temp;
                deserialize(temp, sin);
                item = temp;
                return;
            }

            if (version != "multiply_")
                throw serialization_error("Unexpected version found while deserializing dlib::multiply_.");
            deserialize(item.val, in);
        }

    private:
        float val;
        resizable_tensor params; // unused
    };

    template <typename SUBNET>
    using multiply = add_layer<multiply_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class affine_
    {
    public:
        affine_(
        ) : mode(FC_MODE)
        {
        }

        explicit affine_(
            layer_mode mode_
        ) : mode(mode_)
        {
        }

        affine_(
            const bn_& item
        )
        {
            gamma = item.gamma;
            beta = item.beta;
            mode = item.mode;

            params.copy_size(item.params);

            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());
            
            resizable_tensor temp(item.params);
            auto sg = gamma(temp,0);
            auto sb = beta(temp,gamma.size());

            g = pointwise_multiply(mat(sg), mat(item.running_invstds));
            b = mat(sb) - pointwise_multiply(mat(g), mat(item.running_means));
        }

        layer_mode get_mode() const { return mode; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            if (mode == FC_MODE)
            {
                gamma = alias_tensor(1,
                                sub.get_output().k(),
                                sub.get_output().nr(),
                                sub.get_output().nc());
            }
            else
            {
                gamma = alias_tensor(1, sub.get_output().k());
            }
            beta = gamma;

            params.set_size(gamma.size()+beta.size());

            gamma(params,0) = 1;
            beta(params,gamma.size()) = 0;
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());
            if (mode == FC_MODE)
                tt::affine_transform(output, input, g, b);
            else
                tt::affine_transform_conv(output, input, g, b);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& params_grad
        )
        {
            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());

            // We are computing the gradient of dot(gradient_input, computed_output*g + b)
            if (mode == FC_MODE)
            {
                tt::multiply(data_grad, gradient_input, g);
            }
            else
            {
                tt::multiply_conv(data_grad, gradient_input, g);
            }
        }

        const tensor& get_layer_params() const { return empty_params; }
        tensor& get_layer_params() { return empty_params; }

        friend void serialize(const affine_& item, std::ostream& out)
        {
            serialize("affine_", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.beta, out);
            serialize((int)item.mode, out);
        }

        friend void deserialize(affine_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "bn_")
            {
                // Since we can build an affine_ from a bn_ we check if that's what is in
                // the stream and if so then just convert it right here.
                unserialize sin(version, in);
                bn_ temp;
                deserialize(temp, sin);
                item = temp;
                return;
            }

            if (version != "affine_")
                throw serialization_error("Unexpected version found while deserializing dlib::affine_.");
            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.beta, in);
            int mode;
            deserialize(mode, in);
            item.mode = (layer_mode)mode;
        }

    private:
        resizable_tensor params, empty_params; 
        alias_tensor gamma, beta;
        layer_mode mode;
    };

    template <typename SUBNET>
    using affine = add_layer<affine_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class tag
        >
    class add_prev_
    {
    public:
        add_prev_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            output.copy_size(sub.get_output());
            tt::add(output, sub.get_output(), layer<tag>(sub).get_output());
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            // The gradient just flows backwards to the two layers that forward() added
            // together.
            tt::add(sub.get_gradient_input(), sub.get_gradient_input(), gradient_input);
            tt::add(layer<tag>(sub).get_gradient_input(), layer<tag>(sub).get_gradient_input(), gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const add_prev_& , std::ostream& out)
        {
            serialize("add_prev_", out);
        }

        friend void deserialize(add_prev_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "add_prev_")
                throw serialization_error("Unexpected version found while deserializing dlib::add_prev_.");
        }

    private:
        resizable_tensor params;
    };

    template <
        template<typename> class tag,
        typename SUBNET
        >
    using add_prev = add_layer<add_prev_<tag>, SUBNET>;

    template <typename SUBNET> using add_prev1  = add_prev<tag1, SUBNET>;
    template <typename SUBNET> using add_prev2  = add_prev<tag2, SUBNET>;
    template <typename SUBNET> using add_prev3  = add_prev<tag3, SUBNET>;
    template <typename SUBNET> using add_prev4  = add_prev<tag4, SUBNET>;
    template <typename SUBNET> using add_prev5  = add_prev<tag5, SUBNET>;
    template <typename SUBNET> using add_prev6  = add_prev<tag6, SUBNET>;
    template <typename SUBNET> using add_prev7  = add_prev<tag7, SUBNET>;
    template <typename SUBNET> using add_prev8  = add_prev<tag8, SUBNET>;
    template <typename SUBNET> using add_prev9  = add_prev<tag9, SUBNET>;
    template <typename SUBNET> using add_prev10 = add_prev<tag10, SUBNET>;

    using add_prev1_  = add_prev_<tag1>;
    using add_prev2_  = add_prev_<tag2>;
    using add_prev3_  = add_prev_<tag3>;
    using add_prev4_  = add_prev_<tag4>;
    using add_prev5_  = add_prev_<tag5>;
    using add_prev6_  = add_prev_<tag6>;
    using add_prev7_  = add_prev_<tag7>;
    using add_prev8_  = add_prev_<tag8>;
    using add_prev9_  = add_prev_<tag9>;
    using add_prev10_ = add_prev_<tag10>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:
        relu_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::relu(output, input);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& 
        )
        {
            tt::relu_gradient(data_grad, computed_output, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const relu_& , std::ostream& out)
        {
            serialize("relu_", out);
        }

        friend void deserialize(relu_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "relu_")
                throw serialization_error("Unexpected version found while deserializing dlib::relu_.");
        }

    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using relu = add_layer<relu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class sig_
    {
    public:
        sig_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::sigmoid(output, input);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& 
        )
        {
            tt::sigmoid_gradient(data_grad, computed_output, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const sig_& , std::ostream& out)
        {
            serialize("sig_", out);
        }

        friend void deserialize(sig_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "sig_")
                throw serialization_error("Unexpected version found while deserializing dlib::sig_.");
        }

    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using sig = add_layer<sig_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class htan_
    {
    public:
        htan_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::tanh(output, input);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& 
        )
        {
            tt::tanh_gradient(data_grad, computed_output, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const htan_& , std::ostream& out)
        {
            serialize("htan_", out);
        }

        friend void deserialize(htan_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "htan_")
                throw serialization_error("Unexpected version found while deserializing dlib::htan_.");
        }

    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using htan = add_layer<htan_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class softmax_
    {
    public:
        softmax_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::softmax(output, input);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& 
        )
        {
            tt::softmax_gradient(data_grad, computed_output, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const softmax_& , std::ostream& out)
        {
            serialize("softmax_", out);
        }

        friend void deserialize(softmax_& , std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "softmax_")
                throw serialization_error("Unexpected version found while deserializing dlib::softmax_.");
        }

    private:
        resizable_tensor params;
    };

    template <typename SUBNET>
    using softmax = add_layer<softmax_, SUBNET>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_LAYERS_H_


