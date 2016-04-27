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

    template <
        long _num_filters,
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x
        >
    class con_
    {
    public:

        static_assert(_num_filters > 0, "The number of filters must be > 0");
        static_assert(_nr > 0, "The number of rows in a filter must be > 0");
        static_assert(_nc > 0, "The number of columns in a filter must be > 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");

        con_(
        )  
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

            dlib::rand rnd(std::rand());
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
            serialize(_num_filters, out);
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(item.filters, out);
            serialize(item.biases, out);
        }

        friend void deserialize(con_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "con_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::con_.");
            deserialize(item.params, in);


            long num_filters;
            long nr;
            long nc;
            int stride_y;
            int stride_x;
            deserialize(num_filters, in);
            deserialize(nr, in);
            deserialize(nc, in);
            deserialize(stride_y, in);
            deserialize(stride_x, in);
            deserialize(item.filters, in);
            deserialize(item.biases, in);

            if (num_filters != _num_filters) throw serialization_error("Wrong num_filters found while deserializing dlib::con_");
            if (nr != _nr) throw serialization_error("Wrong nr found while deserializing dlib::con_");
            if (nc != _nc) throw serialization_error("Wrong nc found while deserializing dlib::con_");
            if (stride_y != _stride_y) throw serialization_error("Wrong stride_y found while deserializing dlib::con_");
            if (stride_x != _stride_x) throw serialization_error("Wrong stride_x found while deserializing dlib::con_");
        }


        friend std::ostream& operator<<(std::ostream& out, const con_& item)
        {
            out << "con\t ("
                << "num_filters="<<_num_filters
                << ", nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ")";
            return out;
        }


    private:

        resizable_tensor params;
        alias_tensor filters, biases;

        tt::tensor_conv conv;

    };

    template <
        long num_filters,
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using con = add_layer<con_<num_filters,nr,nc,stride_y,stride_x>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x
        >
    class max_pool_
    {
        static_assert(_nr > 0, "The number of rows in a filter must be > 0");
        static_assert(_nc > 0, "The number of columns in a filter must be > 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
    public:


        max_pool_(
        ) {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }

        max_pool_ (
            const max_pool_& 
        )  
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
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
        }

        friend void deserialize(max_pool_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "max_pool_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::max_pool_.");

            item.mp.setup_max_pooling(_nr, _nc, _stride_y, _stride_x);

            long nr;
            long nc;
            int stride_y;
            int stride_x;

            deserialize(nr, in);
            deserialize(nc, in);
            deserialize(stride_y, in);
            deserialize(stride_x, in);
            if (_nr != nr) throw serialization_error("Wrong nr found while deserializing dlib::max_pool_");
            if (_nc != nc) throw serialization_error("Wrong nc found while deserializing dlib::max_pool_");
            if (_stride_y != stride_y) throw serialization_error("Wrong stride_y found while deserializing dlib::max_pool_");
            if (_stride_x != stride_x) throw serialization_error("Wrong stride_x found while deserializing dlib::max_pool_");
        }

        friend std::ostream& operator<<(std::ostream& out, const max_pool_& item)
        {
            out << "max_pool ("
                << "nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ")";
            return out;
        }


    private:


        tt::pooling mp;
        resizable_tensor params;
    };

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using max_pool = add_layer<max_pool_<nr,nc,stride_y,stride_x>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x
        >
    class avg_pool_
    {
    public:
        static_assert(_nr > 0, "The number of rows in a filter must be > 0");
        static_assert(_nc > 0, "The number of columns in a filter must be > 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");

        avg_pool_(
        ) {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }

        avg_pool_ (
            const avg_pool_& 
        )  
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
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
        }

        friend void deserialize(avg_pool_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "avg_pool_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::avg_pool_.");

            item.ap.setup_avg_pooling(_nr, _nc, _stride_y, _stride_x);

            long nr;
            long nc;
            int stride_y;
            int stride_x;

            deserialize(nr, in);
            deserialize(nc, in);
            deserialize(stride_y, in);
            deserialize(stride_x, in);
            if (_nr != nr) throw serialization_error("Wrong nr found while deserializing dlib::avg_pool_");
            if (_nc != nc) throw serialization_error("Wrong nc found while deserializing dlib::avg_pool_");
            if (_stride_y != stride_y) throw serialization_error("Wrong stride_y found while deserializing dlib::avg_pool_");
            if (_stride_x != stride_x) throw serialization_error("Wrong stride_x found while deserializing dlib::avg_pool_");
        }

        friend std::ostream& operator<<(std::ostream& out, const avg_pool_& item)
        {
            out << "avg_pool ("
                << "nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ")";
            return out;
        }
    private:

        tt::pooling ap;
        resizable_tensor params;
    };

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using avg_pool = add_layer<avg_pool_<nr,nc,stride_y,stride_x>, SUBNET>;

// ----------------------------------------------------------------------------------------

    enum layer_mode
    {
        CONV_MODE = 0,
        FC_MODE = 1
    };

    template <
        layer_mode mode
        >
    class bn_
    {
    public:
        bn_() : num_updates(0), running_stats_window_size(1000)
        {}

        explicit bn_(unsigned long window_size) : num_updates(0), running_stats_window_size(window_size)
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
            running_variances.copy_size(gamma(params,0));
            running_means = 0;
            running_variances = 1;
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
                    tt::batch_normalize(output, means, invstds, decay, running_means, running_variances, sub.get_output(), g, b);
                else 
                    tt::batch_normalize_conv(output, means, invstds, decay, running_means, running_variances, sub.get_output(), g, b);
            }
            else // we are running in testing mode so we just linearly scale the input tensor.
            {
                if (mode == FC_MODE)
                    tt::batch_normalize_inference(output, sub.get_output(), g, b, running_means, running_variances);
                else
                    tt::batch_normalize_conv_inference(output, sub.get_output(), g, b, running_means, running_variances);
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
            if (mode == CONV_MODE)
                serialize("bn_con", out);
            else // if FC_MODE
                serialize("bn_fc", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.beta, out);
            serialize(item.means, out);
            serialize(item.invstds, out);
            serialize(item.running_means, out);
            serialize(item.running_variances, out);
            serialize(item.num_updates, out);
            serialize(item.running_stats_window_size, out);
        }

        friend void deserialize(bn_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "bn_")
            {
                if (mode == CONV_MODE) 
                {
                    if (version != "bn_con")
                        throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::bn_.");
                }
                else // must be in FC_MODE
                {
                    if (version != "bn_fc")
                        throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::bn_.");
                }
            }

            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.beta, in);
            deserialize(item.means, in);
            deserialize(item.invstds, in);
            deserialize(item.running_means, in);
            deserialize(item.running_variances, in);
            deserialize(item.num_updates, in);
            deserialize(item.running_stats_window_size, in);

            // if this is the older "bn_" version then check its saved mode value and make
            // sure it is the one we are really using.  
            if (version == "bn_")
            {
                int _mode;
                deserialize(_mode, in);
                if (mode != (layer_mode)_mode) throw serialization_error("Wrong mode found while deserializing dlib::bn_");

                // We also need to flip the running_variances around since the previous
                // format saved the inverse standard deviations instead of variances.
                item.running_variances = 1.0f/squared(mat(item.running_variances)) - tt::BATCH_NORM_EPS;
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const bn_& item)
        {
            if (mode == CONV_MODE)
                out << "bn_con";
            else
                out << "bn_fc";
            return out;
        }

    private:

        friend class affine_;

        resizable_tensor params;
        alias_tensor gamma, beta;
        resizable_tensor means, running_means;
        resizable_tensor invstds, running_variances;
        unsigned long num_updates;
        unsigned long running_stats_window_size;
    };

    template <typename SUBNET>
    using bn_con = add_layer<bn_<CONV_MODE>, SUBNET>;
    template <typename SUBNET>
    using bn_fc = add_layer<bn_<FC_MODE>, SUBNET>;

// ----------------------------------------------------------------------------------------

    enum fc_bias_mode
    {
        FC_HAS_BIAS = 0,
        FC_NO_BIAS = 1
    };

    struct num_fc_outputs
    {
        num_fc_outputs(unsigned long n) : num_outputs(n) {}
        unsigned long num_outputs;
    };

    template <
        unsigned long num_outputs_,
        fc_bias_mode bias_mode
        >
    class fc_
    {
        static_assert(num_outputs_ > 0, "The number of outputs from a fc_ layer must be > 0");

    public:
        fc_() : num_outputs(num_outputs_), num_inputs(0)
        {
        }

        fc_(num_fc_outputs o) : num_outputs(o.num_outputs), num_inputs(0) {}

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

            dlib::rand rnd(std::rand());
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
            serialize((int)bias_mode, out);
        }

        friend void deserialize(fc_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "fc_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::fc_.");

            deserialize(item.num_outputs, in);
            deserialize(item.num_inputs, in);
            deserialize(item.params, in);
            deserialize(item.weights, in);
            deserialize(item.biases, in);
            int bmode = 0;
            deserialize(bmode, in);
            if (bias_mode != (fc_bias_mode)bmode) throw serialization_error("Wrong fc_bias_mode found while deserializing dlib::fc_");
        }

        friend std::ostream& operator<<(std::ostream& out, const fc_& item)
        {
            if (bias_mode == FC_HAS_BIAS)
            {
                out << "fc\t ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
            }
            else
            {
                out << "fc_no_bias ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
            }
            return out;
        }

    private:

        unsigned long num_outputs;
        unsigned long num_inputs;
        resizable_tensor params;
        alias_tensor weights, biases;
    };

    template <
        unsigned long num_outputs,
        typename SUBNET
        >
    using fc = add_layer<fc_<num_outputs,FC_HAS_BIAS>, SUBNET>;

    template <
        unsigned long num_outputs,
        typename SUBNET
        >
    using fc_no_bias = add_layer<fc_<num_outputs,FC_NO_BIAS>, SUBNET>;

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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::dropout_.");
            deserialize(item.drop_rate, in);
            deserialize(item.mask, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const dropout_& item)
        {
            out << "dropout\t ("
                << "drop_rate="<<item.drop_rate
                << ")";
            return out;
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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::multiply_.");
            deserialize(item.val, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const multiply_& item)
        {
            out << "multiply ("
                << "val="<<item.val
                << ")";
            return out;
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

        affine_(
            layer_mode mode_
        ) : mode(mode_)
        {
        }

        template <
            layer_mode bnmode
            >
        affine_(
            const bn_<bnmode>& item
        )
        {
            gamma = item.gamma;
            beta = item.beta;
            mode = bnmode;

            params.copy_size(item.params);

            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());
            
            resizable_tensor temp(item.params);
            auto sg = gamma(temp,0);
            auto sb = beta(temp,gamma.size());

            g = pointwise_multiply(mat(sg), 1.0f/sqrt(mat(item.running_variances)+tt::BATCH_NORM_EPS));
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
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& /*params_grad*/
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
            if (version == "bn_con")
            {
                // Since we can build an affine_ from a bn_ we check if that's what is in
                // the stream and if so then just convert it right here.
                unserialize sin(version, in);
                bn_<CONV_MODE> temp;
                deserialize(temp, sin);
                item = temp;
                return;
            }
            else if (version == "bn_fc")
            {
                // Since we can build an affine_ from a bn_ we check if that's what is in
                // the stream and if so then just convert it right here.
                unserialize sin(version, in);
                bn_<FC_MODE> temp;
                deserialize(temp, sin);
                item = temp;
                return;
            }

            if (version != "affine_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::affine_.");
            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.beta, in);
            int mode;
            deserialize(mode, in);
            item.mode = (layer_mode)mode;
        }

        friend std::ostream& operator<<(std::ostream& out, const affine_& )
        {
            out << "affine";
            return out;
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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::add_prev_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const add_prev_& item)
        {
            out << "add_prev";
            return out;
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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::relu_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const relu_& )
        {
            out << "relu";
            return out;
        }


    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using relu = add_layer<relu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class prelu_
    {
    public:
        explicit prelu_(
            float initial_param_value_ = 0.25
        ) : initial_param_value(initial_param_value_)
        {
        }

        float get_initial_param_value (
        ) const { return initial_param_value; }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
            params.set_size(1);
            params = initial_param_value;
        }

        template <typename SUBNET>
        void forward(
            const SUBNET& sub, 
            resizable_tensor& data_output
        )
        {
            data_output.copy_size(sub.get_output());
            tt::prelu(data_output, sub.get_output(), params);
        }

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input, 
            SUBNET& sub, 
            tensor& params_grad
        )
        {
            tt::prelu_gradient(sub.get_gradient_input(), sub.get_output(), 
                gradient_input, params, params_grad);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const prelu_& item, std::ostream& out)
        {
            serialize("prelu_", out);
            serialize(item.params, out);
            serialize(item.initial_param_value, out);
        }

        friend void deserialize(prelu_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "prelu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::prelu_.");
            deserialize(item.params, in);
            deserialize(item.initial_param_value, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const prelu_& item)
        {
            out << "prelu\t ("
                << "initial_param_value="<<item.initial_param_value
                << ")";
            return out;
        }

    private:
        resizable_tensor params;
        float initial_param_value;
    };

    template <typename SUBNET>
    using prelu = add_layer<prelu_, SUBNET>;

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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::sig_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const sig_& )
        {
            out << "sig";
            return out;
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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::htan_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const htan_& )
        {
            out << "htan";
            return out;
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
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::softmax_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const softmax_& )
        {
            out << "softmax";
            return out;
        }

    private:
        resizable_tensor params;
    };

    template <typename SUBNET>
    using softmax = add_layer<softmax_, SUBNET>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_LAYERS_H_


