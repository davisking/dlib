// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_LAYERS_H_
#define DLIB_DNn_LAYERS_H_

#include "layers_abstract.h"
#include "../cuda/tensor.h"
#include "core.h"
#include <iostream>
#include <string>
#include "../rand.h"
#include "../string.h"
#include "../cuda/tensor_tools.h"
#include "../vectorstream.h"
#include "utilities.h"
#include <sstream>


namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct num_con_outputs
    {
        num_con_outputs(unsigned long n) : num_outputs(n) {}
        unsigned long num_outputs;
    };

    template <
        long _num_filters,
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = _stride_y!=1? 0 : _nr/2,
        int _padding_x = _stride_x!=1? 0 : _nc/2
        >
    class con_
    {
    public:

        static_assert(_num_filters > 0, "The number of filters must be > 0");
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(_nr==0 || (0 <= _padding_y && _padding_y < _nr), "The padding must be smaller than the filter size.");
        static_assert(_nc==0 || (0 <= _padding_x && _padding_x < _nc), "The padding must be smaller than the filter size.");
        static_assert(_nr!=0 || 0 == _padding_y, "If _nr==0 then the padding must be set to 0 as well.");
        static_assert(_nc!=0 || 0 == _padding_x, "If _nr==0 then the padding must be set to 0 as well.");

        con_(
            num_con_outputs o
        ) : 
            learning_rate_multiplier(1),
            weight_decay_multiplier(1),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(0),
            num_filters_(o.num_outputs),
            padding_y_(_padding_y),
            padding_x_(_padding_x),
            use_bias(true),
            use_relu(false)
        {
            DLIB_CASSERT(num_filters_ > 0);
        }

        con_() : con_(num_con_outputs(_num_filters)) {}

        long num_filters() const { return num_filters_; }
        long nr() const 
        { 
            if (_nr==0)
                return filters.nr();
            else
                return _nr;
        }
        long nc() const 
        { 
            if (_nc==0)
                return filters.nc();
            else
                return _nc;
        }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }
        long padding_y() const { return padding_y_; }
        long padding_x() const { return padding_x_; }

        void set_num_filters(long num) 
        {
            DLIB_CASSERT(num > 0);
            if (num != num_filters_)
            {
                DLIB_CASSERT(get_layer_params().size() == 0, 
                    "You can't change the number of filters in con_ if the parameter tensor has already been allocated.");
                num_filters_ = num;
            }
        }

        double get_learning_rate_multiplier () const  { return learning_rate_multiplier; }
        double get_weight_decay_multiplier () const   { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val)  { weight_decay_multiplier  = val; }

        double get_bias_learning_rate_multiplier () const  { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier () const   { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val)  { bias_weight_decay_multiplier  = val; }

        bool relu_is_disabled() const { return !use_relu; }

        void disable_relu()
        {
            use_relu = false;
        }

        void enable_relu()
        {
            use_relu = true;
        }

        bool bias_is_disabled() const { return !use_bias; }

        void disable_bias()
        {
            if (use_bias == false)
                return;

            use_bias = false;
            if (params.size() == 0)
                return;

            DLIB_CASSERT(params.size() == filters.size() + num_filters_);
            auto temp = params;
            params.set_size(params.size() - num_filters_);
            std::copy(temp.begin(), temp.end() - num_filters_, params.begin());
            biases = alias_tensor();
        }

        void enable_bias()
        {
            if (use_bias == true)
                return;

            use_bias = true;
            if (params.size() == 0)
                return;

            DLIB_CASSERT(params.size() == filters.size());
            auto temp = params;
            params.set_size(params.size() + num_filters_);
            std::copy(temp.begin(), temp.end(), params.begin());
            biases = alias_tensor(1, num_filters_);
            biases(params, filters.size()) = 0;
        }

        inline dpoint map_input_to_output (
            dpoint p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline dpoint map_output_to_input (
            dpoint p
        ) const
        {
            p.x() = p.x()*stride_x() - padding_x() + nc()/2;
            p.y() = p.y()*stride_y() - padding_y() + nr()/2;
            return p;
        }

        con_ (
            const con_& item
        ) : 
            params(item.params),
            filters(item.filters),
            biases(item.biases),
            learning_rate_multiplier(item.learning_rate_multiplier),
            weight_decay_multiplier(item.weight_decay_multiplier),
            bias_learning_rate_multiplier(item.bias_learning_rate_multiplier),
            bias_weight_decay_multiplier(item.bias_weight_decay_multiplier),
            num_filters_(item.num_filters_),
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_),
            use_bias(item.use_bias),
            use_relu(item.use_relu)
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
            padding_y_ = item.padding_y_;
            padding_x_ = item.padding_x_;
            learning_rate_multiplier = item.learning_rate_multiplier;
            weight_decay_multiplier = item.weight_decay_multiplier;
            bias_learning_rate_multiplier = item.bias_learning_rate_multiplier;
            bias_weight_decay_multiplier = item.bias_weight_decay_multiplier;
            num_filters_ = item.num_filters_;
            use_bias = item.use_bias;
            use_relu = item.use_relu;
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            const long filt_nr = _nr!=0 ? _nr : sub.get_output().nr();
            const long filt_nc = _nc!=0 ? _nc : sub.get_output().nc();

            long num_inputs = filt_nr*filt_nc*sub.get_output().k();
            long num_outputs = num_filters_;
            // allocate params for the filters and also for the filter bias values.
            params.set_size(num_inputs*num_filters_ + static_cast<int>(use_bias) * num_filters_);

            dlib::rand rnd(std::rand());
            randomize_parameters(params, num_inputs+num_outputs, rnd);

            filters = alias_tensor(num_filters_, sub.get_output().k(), filt_nr, filt_nc);
            if (use_bias)
            {
                biases = alias_tensor(1,num_filters_);
                // set the initial bias values to zero
                biases(params,filters.size()) = 0;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            conv.setup(sub.get_output(),
                       filters(params,0),
                       _stride_y,
                       _stride_x,
                       padding_y_,
                       padding_x_);

            if (use_bias)
            {
                conv(false, output,
                     sub.get_output(),
                     filters(params,0),
                     biases(params, filters.size()),
                     use_relu);
            }
            else
            {
                conv(false, output,
                     sub.get_output(),
                     filters(params,0));
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            conv.get_gradient_for_data (true, gradient_input, filters(params,0), sub.get_gradient_input());
            // no point computing the parameter gradients if they won't be used.
            if (learning_rate_multiplier != 0)
            {
                auto filt = filters(params_grad,0);
                conv.get_gradient_for_filters (false, gradient_input, sub.get_output(), filt);
                if (use_bias)
                {
                    auto b = biases(params_grad, filters.size());
                    tt::assign_conv_bias_gradient(b, gradient_input);
                }
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const con_& item, std::ostream& out)
        {
            serialize("con_6", out);
            serialize(item.params, out);
            serialize(item.num_filters_, out);
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(item.padding_y_, out);
            serialize(item.padding_x_, out);
            serialize(item.filters, out);
            serialize(item.biases, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.weight_decay_multiplier, out);
            serialize(item.bias_learning_rate_multiplier, out);
            serialize(item.bias_weight_decay_multiplier, out);
            serialize(item.use_bias, out);
            serialize(item.use_relu, out);
        }

        friend void deserialize(con_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            long nr;
            long nc;
            int stride_y;
            int stride_x;
            if (version == "con_4" || version == "con_5" || version == "con_6")
            {
                deserialize(item.params, in);
                deserialize(item.num_filters_, in);
                deserialize(nr, in);
                deserialize(nc, in);
                deserialize(stride_y, in);
                deserialize(stride_x, in);
                deserialize(item.padding_y_, in);
                deserialize(item.padding_x_, in);
                deserialize(item.filters, in);
                deserialize(item.biases, in);
                deserialize(item.learning_rate_multiplier, in);
                deserialize(item.weight_decay_multiplier, in);
                deserialize(item.bias_learning_rate_multiplier, in);
                deserialize(item.bias_weight_decay_multiplier, in);
                if (item.padding_y_ != _padding_y) throw serialization_error("Wrong padding_y found while deserializing dlib::con_");
                if (item.padding_x_ != _padding_x) throw serialization_error("Wrong padding_x found while deserializing dlib::con_");
                if (nr != _nr) throw serialization_error("Wrong nr found while deserializing dlib::con_");
                if (nc != _nc) throw serialization_error("Wrong nc found while deserializing dlib::con_");
                if (stride_y != _stride_y) throw serialization_error("Wrong stride_y found while deserializing dlib::con_");
                if (stride_x != _stride_x) throw serialization_error("Wrong stride_x found while deserializing dlib::con_");
                if (version == "con_5" || version == "con_6")
                {
                    deserialize(item.use_bias, in);
                }
                if (version == "con_6")
                {
                    deserialize(item.use_relu, in);
                }
            }
            else
            {
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::con_.");
            }
        }


        friend std::ostream& operator<<(std::ostream& out, const con_& item)
        {
            out << "con\t ("
                << "num_filters="<<item.num_filters_
                << ", nr="<<item.nr()
                << ", nc="<<item.nc()
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            if (item.use_bias)
            {
                out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            }
            else
            {
                out << " use_bias=false";
            }
            if (item.use_relu)
            {
                out << " use_relu="<< std::boolalpha << item.use_relu;
            }
            return out;
        }

        friend void to_xml(const con_& item, std::ostream& out)
        {
            out << "<con"
                << " num_filters='"<<item.num_filters_<<"'"
                << " nr='"<<item.nr()<<"'"
                << " nc='"<<item.nc()<<"'"
                << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<item.padding_y_<<"'"
                << " padding_x='"<<item.padding_x_<<"'"
                << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'"
                << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'"
                << " use_bias='"<<(item.use_bias?"true":"false")<<"'"
                << " use_relu='"<<(item.use_relu?"true":"false")<<"'"
                << ">\n";
            out << mat(item.params);
            out << "</con>\n";
        }

    private:

        resizable_tensor params;
        alias_tensor filters, biases;

        tt::tensor_conv conv;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        long num_filters_;

        // These are here only because older versions of con (which you might encounter
        // serialized to disk) used different padding settings.
        int padding_y_;
        int padding_x_;
        bool use_bias;
        bool use_relu;
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
        long _num_filters,
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = _stride_y!=1? 0 : _nr/2,
        int _padding_x = _stride_x!=1? 0 : _nc/2
        >
    class cont_
    {
    public:

        static_assert(_num_filters > 0, "The number of filters must be > 0");
        static_assert(_nr > 0, "The number of rows in a filter must be > 0");
        static_assert(_nc > 0, "The number of columns in a filter must be > 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(0 <= _padding_y && _padding_y < _nr, "The padding must be smaller than the filter size.");
        static_assert(0 <= _padding_x && _padding_x < _nc, "The padding must be smaller than the filter size.");

        cont_(
            num_con_outputs o
        ) : 
            learning_rate_multiplier(1),
            weight_decay_multiplier(1),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(0),
            num_filters_(o.num_outputs),
            padding_y_(_padding_y),
            padding_x_(_padding_x),
            use_bias(true)
        {
            DLIB_CASSERT(num_filters_ > 0);
        }

        cont_() : cont_(num_con_outputs(_num_filters)) {}

        long num_filters() const { return num_filters_; }
        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }
        long padding_y() const { return padding_y_; }
        long padding_x() const { return padding_x_; }

        void set_num_filters(long num)
        {
            DLIB_CASSERT(num > 0);
            if (num != num_filters_)
            {
                DLIB_CASSERT(get_layer_params().size() == 0,
                    "You can't change the number of filters in cont_ if the parameter tensor has already been allocated.");
                num_filters_ = num;
            }
        }

        double get_learning_rate_multiplier () const  { return learning_rate_multiplier; }
        double get_weight_decay_multiplier () const   { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val)  { weight_decay_multiplier  = val; }

        double get_bias_learning_rate_multiplier () const  { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier () const   { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val)  { bias_weight_decay_multiplier  = val; }
        void disable_bias() { use_bias = false; }
        bool bias_is_disabled() const { return !use_bias; }

        inline dpoint map_output_to_input (
            dpoint p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline dpoint map_input_to_output (
            dpoint p
        ) const
        {
            p.x() = p.x()*stride_x() - padding_x() + nc()/2;
            p.y() = p.y()*stride_y() - padding_y() + nr()/2;
            return p;
        }

        cont_ (
            const cont_& item
        ) : 
            params(item.params),
            filters(item.filters),
            biases(item.biases),
            learning_rate_multiplier(item.learning_rate_multiplier),
            weight_decay_multiplier(item.weight_decay_multiplier),
            bias_learning_rate_multiplier(item.bias_learning_rate_multiplier),
            bias_weight_decay_multiplier(item.bias_weight_decay_multiplier),
            num_filters_(item.num_filters_),
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_),
            use_bias(item.use_bias)
        {
            // this->conv is non-copyable and basically stateless, so we have to write our
            // own copy to avoid trying to copy it and getting an error.
        }

        cont_& operator= (
            const cont_& item
        )
        {
            if (this == &item)
                return *this;

            // this->conv is non-copyable and basically stateless, so we have to write our
            // own copy to avoid trying to copy it and getting an error.
            params = item.params;
            filters = item.filters;
            biases = item.biases;
            padding_y_ = item.padding_y_;
            padding_x_ = item.padding_x_;
            learning_rate_multiplier = item.learning_rate_multiplier;
            weight_decay_multiplier = item.weight_decay_multiplier;
            bias_learning_rate_multiplier = item.bias_learning_rate_multiplier;
            bias_weight_decay_multiplier = item.bias_weight_decay_multiplier;
            num_filters_ = item.num_filters_;
            use_bias = item.use_bias;
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            long num_inputs = _nr*_nc*sub.get_output().k();
            long num_outputs = num_filters_;
            // allocate params for the filters and also for the filter bias values.
            params.set_size(num_inputs*num_filters_ + num_filters_ * static_cast<int>(use_bias));

            dlib::rand rnd(std::rand());
            randomize_parameters(params, num_inputs+num_outputs, rnd);

            filters = alias_tensor(sub.get_output().k(), num_filters_, _nr, _nc);
            if (use_bias)
            {
                biases = alias_tensor(1,num_filters_);
                // set the initial bias values to zero
                biases(params,filters.size()) = 0;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto filt = filters(params,0);
            unsigned int gnr = _stride_y * (sub.get_output().nr() - 1) + filt.nr() - 2 * padding_y_;
            unsigned int gnc = _stride_x * (sub.get_output().nc() - 1) + filt.nc() - 2 * padding_x_;
            unsigned int gnsamps = sub.get_output().num_samples();
            unsigned int gk = filt.k();
            output.set_size(gnsamps,gk,gnr,gnc);
            conv.setup(output,filt,_stride_y,_stride_x,padding_y_,padding_x_);
            conv.get_gradient_for_data(false, sub.get_output(),filt,output);            
            if (use_bias)
            {
                tt::add(1,output,1,biases(params,filters.size()));
            }
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            auto filt = filters(params,0);           
            conv(true, sub.get_gradient_input(),gradient_input, filt);
            // no point computing the parameter gradients if they won't be used.
            if (learning_rate_multiplier != 0)
            {
                auto filt = filters(params_grad,0);                
                conv.get_gradient_for_filters (false, sub.get_output(),gradient_input, filt);
                if (use_bias)
                {
                    auto b = biases(params_grad, filters.size());
                    tt::assign_conv_bias_gradient(b, gradient_input);
                }
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const cont_& item, std::ostream& out)
        {
            serialize("cont_2", out);
            serialize(item.params, out);
            serialize(item.num_filters_, out);
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(item.padding_y_, out);
            serialize(item.padding_x_, out);
            serialize(item.filters, out);
            serialize(item.biases, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.weight_decay_multiplier, out);
            serialize(item.bias_learning_rate_multiplier, out);
            serialize(item.bias_weight_decay_multiplier, out);
            serialize(item.use_bias, out);
        }

        friend void deserialize(cont_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            long nr;
            long nc;
            int stride_y;
            int stride_x;
            if (version == "cont_1" || version == "cont_2")
            {
                deserialize(item.params, in);
                deserialize(item.num_filters_, in);
                deserialize(nr, in);
                deserialize(nc, in);
                deserialize(stride_y, in);
                deserialize(stride_x, in);
                deserialize(item.padding_y_, in);
                deserialize(item.padding_x_, in);
                deserialize(item.filters, in);
                deserialize(item.biases, in);
                deserialize(item.learning_rate_multiplier, in);
                deserialize(item.weight_decay_multiplier, in);
                deserialize(item.bias_learning_rate_multiplier, in);
                deserialize(item.bias_weight_decay_multiplier, in);
                if (item.padding_y_ != _padding_y) throw serialization_error("Wrong padding_y found while deserializing dlib::con_");
                if (item.padding_x_ != _padding_x) throw serialization_error("Wrong padding_x found while deserializing dlib::con_");
                if (nr != _nr) throw serialization_error("Wrong nr found while deserializing dlib::con_");
                if (nc != _nc) throw serialization_error("Wrong nc found while deserializing dlib::con_");
                if (stride_y != _stride_y) throw serialization_error("Wrong stride_y found while deserializing dlib::con_");
                if (stride_x != _stride_x) throw serialization_error("Wrong stride_x found while deserializing dlib::con_");
                if (version == "cont_2")
                {
                    deserialize(item.use_bias, in);
                }
            }
            else
            {
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::con_.");
            }
        }


        friend std::ostream& operator<<(std::ostream& out, const cont_& item)
        {
            out << "cont\t ("
                << "num_filters="<<item.num_filters_
                << ", nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            if (item.use_bias)
            {
                out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            }
            else
            {
                out << " use_bias=false";
            }
            return out;
        }

        friend void to_xml(const cont_& item, std::ostream& out)
        {
            out << "<cont"
                << " num_filters='"<<item.num_filters_<<"'"
                << " nr='"<<_nr<<"'"
                << " nc='"<<_nc<<"'"
                << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<item.padding_y_<<"'"
                << " padding_x='"<<item.padding_x_<<"'"
                << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'"
                << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'"
                << " use_bias='"<<(item.use_bias?"true":"false")<<"'>\n";
            out << mat(item.params);
            out << "</cont>\n";
        }

    private:

        resizable_tensor params;
        alias_tensor filters, biases;

        tt::tensor_conv conv;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        long num_filters_;

        int padding_y_;
        int padding_x_;

        bool use_bias;

    };

    template <
        long num_filters,
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using cont = add_layer<cont_<num_filters,nr,nc,stride_y,stride_x>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        int scale_y, 
        int scale_x 
        >
    class upsample_
    {
    public:
        static_assert(scale_y >= 1, "upsampling scale factor can't be less than 1.");
        static_assert(scale_x >= 1, "upsampling scale factor can't be less than 1.");

        upsample_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            output.set_size(
                sub.get_output().num_samples(),
                sub.get_output().k(),
                scale_y*sub.get_output().nr(),
                scale_x*sub.get_output().nc());
            tt::resize_bilinear(output, sub.get_output());
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tt::resize_bilinear_gradient(sub.get_gradient_input(), gradient_input);
        }

        inline dpoint map_input_to_output (dpoint p) const 
        { 
            p.x() = p.x()*scale_x;
            p.y() = p.y()*scale_y;
            return p; 
        }
        inline dpoint map_output_to_input (dpoint p) const 
        { 
            p.x() = p.x()/scale_x;
            p.y() = p.y()/scale_y;
            return p; 
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const upsample_& /*item*/, std::ostream& out)
        {
            serialize("upsample_", out);
            serialize(scale_y, out);
            serialize(scale_x, out);
        }

        friend void deserialize(upsample_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "upsample_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::upsample_.");

            int _scale_y;
            int _scale_x;
            deserialize(_scale_y, in);
            deserialize(_scale_x, in);
            if (_scale_y != scale_y || _scale_x != scale_x)
                throw serialization_error("Wrong scale found while deserializing dlib::upsample_");
        }

        friend std::ostream& operator<<(std::ostream& out, const upsample_& /*item*/)
        {
            out << "upsample\t ("
                << "scale_y="<<scale_y
                << ", scale_x="<<scale_x
                << ")";
            return out;
        }

        friend void to_xml(const upsample_& /*item*/, std::ostream& out)
        {
            out << "<upsample"
                << " scale_y='"<<scale_y<<"'"
                << " scale_x='"<<scale_x<<"'/>\n";
        }

    private:
        resizable_tensor params;
    };

    template <
        int scale,
        typename SUBNET
        >
    using upsample = add_layer<upsample_<scale,scale>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long NR_, 
        long NC_
        >
    class resize_to_
    {
    public:
        static_assert(NR_ >= 1, "NR resize parameter can't be less than 1.");
        static_assert(NC_ >= 1, "NC resize parameter can't be less than 1.");
        
        resize_to_()
        {
        }
        
        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }
    
        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            scale_y = (double)NR_/(double)sub.get_output().nr();
            scale_x = (double)NC_/(double)sub.get_output().nc();
            
            output.set_size(
                sub.get_output().num_samples(),
                sub.get_output().k(),
                NR_,
                NC_);
            tt::resize_bilinear(output, sub.get_output());
        } 
        
        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tt::resize_bilinear_gradient(sub.get_gradient_input(), gradient_input);
        }
        
        inline dpoint map_input_to_output (dpoint p) const 
        { 
            p.x() = p.x()*scale_x;
            p.y() = p.y()*scale_y;
            return p; 
        }

        inline dpoint map_output_to_input (dpoint p) const 
        { 
            p.x() = p.x()/scale_x;
            p.y() = p.y()/scale_y;
            return p; 
        }
        
        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }
        
        friend void serialize(const resize_to_& item, std::ostream& out)
        {
            serialize("resize_to_", out);
            serialize(NR_, out);
            serialize(NC_, out);
            serialize(item.scale_y, out);
            serialize(item.scale_x, out);
        }
        
        friend void deserialize(resize_to_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "resize_to_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::resize_to_.");

            long _nr;
            long _nc;
            deserialize(_nr, in);
            deserialize(_nc, in);
            deserialize(item.scale_y, in);
            deserialize(item.scale_x, in);
            if (_nr != NR_ || _nc != NC_)
                throw serialization_error("Wrong size found while deserializing dlib::resize_to_");
        }
        
        friend std::ostream& operator<<(std::ostream& out, const resize_to_& /*item*/)
        {
            out << "resize_to ("
                << "nr=" << NR_
                << ", nc=" << NC_
                << ")";
            return out;
        }
        
        friend void to_xml(const resize_to_& /*item*/, std::ostream& out)
        {
            out << "<resize_to";
            out << " nr='" << NR_ << "'" ;
            out << " nc='" << NC_ << "'/>\n";
        }
    private:
        resizable_tensor params;
        double scale_y;
        double scale_x;
    
    };  // end of class resize_to_
    
    
    template <
        long NR,
        long NC,
        typename SUBNET
        >
    using resize_to = add_layer<resize_to_<NR,NC>, SUBNET>;
    
// ----------------------------------------------------------------------------------------

    template <
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = _stride_y!=1? 0 : _nr/2,
        int _padding_x = _stride_x!=1? 0 : _nc/2
        >
    class max_pool_
    {
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(0 <= _padding_y && ((_nr==0 && _padding_y == 0) || (_nr!=0 && _padding_y < _nr)), 
            "The padding must be smaller than the filter size, unless the filters size is 0.");
        static_assert(0 <= _padding_x && ((_nc==0 && _padding_x == 0) || (_nc!=0 && _padding_x < _nc)), 
            "The padding must be smaller than the filter size, unless the filters size is 0.");
    public:


        max_pool_(
        ) :
            padding_y_(_padding_y),
            padding_x_(_padding_x)
        {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }
        long padding_y() const { return padding_y_; }
        long padding_x() const { return padding_x_; }

        inline dpoint map_input_to_output (
            dpoint p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline dpoint map_output_to_input (
            dpoint p
        ) const
        {
            p.x() = p.x()*stride_x() - padding_x() + nc()/2;
            p.y() = p.y()*stride_y() - padding_y() + nr()/2;
            return p;
        }

        max_pool_ (
            const max_pool_& item
        )  :
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_)
        {
            // this->mp is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
        }

        max_pool_& operator= (
            const max_pool_& item
        )
        {
            if (this == &item)
                return *this;

            padding_y_ = item.padding_y_;
            padding_x_ = item.padding_x_;

            // this->mp is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            mp.setup_max_pooling(_nr!=0?_nr:sub.get_output().nr(), 
                                 _nc!=0?_nc:sub.get_output().nc(),
                                 _stride_y, _stride_x, padding_y_, padding_x_);

            mp(output, sub.get_output());
        } 

        template <typename SUBNET>
        void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            mp.setup_max_pooling(_nr!=0?_nr:sub.get_output().nr(), 
                                 _nc!=0?_nc:sub.get_output().nc(),
                                 _stride_y, _stride_x, padding_y_, padding_x_);

            mp.get_gradient(gradient_input, computed_output, sub.get_output(), sub.get_gradient_input());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const max_pool_& item, std::ostream& out)
        {
            serialize("max_pool_2", out);
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(item.padding_y_, out);
            serialize(item.padding_x_, out);
        }

        friend void deserialize(max_pool_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            long nr;
            long nc;
            int stride_y;
            int stride_x;
            if (version == "max_pool_2")
            {
                deserialize(nr, in);
                deserialize(nc, in);
                deserialize(stride_y, in);
                deserialize(stride_x, in);
                deserialize(item.padding_y_, in);
                deserialize(item.padding_x_, in);
            }
            else
            {
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::max_pool_.");
            }

            if (item.padding_y_ != _padding_y) throw serialization_error("Wrong padding_y found while deserializing dlib::max_pool_");
            if (item.padding_x_ != _padding_x) throw serialization_error("Wrong padding_x found while deserializing dlib::max_pool_");
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
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            return out;
        }

        friend void to_xml(const max_pool_& item, std::ostream& out)
        {
            out << "<max_pool"
                << " nr='"<<_nr<<"'"
                << " nc='"<<_nc<<"'"
                << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<item.padding_y_<<"'"
                << " padding_x='"<<item.padding_x_<<"'"
                << "/>\n";
        }


    private:


        tt::pooling mp;
        resizable_tensor params;

        int padding_y_;
        int padding_x_;
    };

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using max_pool = add_layer<max_pool_<nr,nc,stride_y,stride_x>, SUBNET>;

    template <
        typename SUBNET
        >
    using max_pool_everything = add_layer<max_pool_<0,0,1,1>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long _nr,
        long _nc,
        int _stride_y,
        int _stride_x,
        int _padding_y = _stride_y!=1? 0 : _nr/2,
        int _padding_x = _stride_x!=1? 0 : _nc/2
        >
    class avg_pool_
    {
    public:
        static_assert(_nr >= 0, "The number of rows in a filter must be >= 0");
        static_assert(_nc >= 0, "The number of columns in a filter must be >= 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(0 <= _padding_y && ((_nr==0 && _padding_y == 0) || (_nr!=0 && _padding_y < _nr)), 
            "The padding must be smaller than the filter size, unless the filters size is 0.");
        static_assert(0 <= _padding_x && ((_nc==0 && _padding_x == 0) || (_nc!=0 && _padding_x < _nc)), 
            "The padding must be smaller than the filter size, unless the filters size is 0.");

        avg_pool_(
        ) :
            padding_y_(_padding_y),
            padding_x_(_padding_x)
        {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }
        long padding_y() const { return padding_y_; }
        long padding_x() const { return padding_x_; }

        inline dpoint map_input_to_output (
            dpoint p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline dpoint map_output_to_input (
            dpoint p
        ) const
        {
            p.x() = p.x()*stride_x() - padding_x() + nc()/2;
            p.y() = p.y()*stride_y() - padding_y() + nr()/2;
            return p;
        }

        avg_pool_ (
            const avg_pool_& item
        )  :
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_)
        {
            // this->ap is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
        }

        avg_pool_& operator= (
            const avg_pool_& item
        )
        {
            if (this == &item)
                return *this;

            padding_y_ = item.padding_y_;
            padding_x_ = item.padding_x_;

            // this->ap is non-copyable so we have to write our own copy to avoid trying to
            // copy it and getting an error.
            return *this;
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            ap.setup_avg_pooling(_nr!=0?_nr:sub.get_output().nr(), 
                                 _nc!=0?_nc:sub.get_output().nc(),
                                 _stride_y, _stride_x, padding_y_, padding_x_);

            ap(output, sub.get_output());
        } 

        template <typename SUBNET>
        void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            ap.setup_avg_pooling(_nr!=0?_nr:sub.get_output().nr(), 
                                 _nc!=0?_nc:sub.get_output().nc(),
                                 _stride_y, _stride_x, padding_y_, padding_x_);

            ap.get_gradient(gradient_input, computed_output, sub.get_output(), sub.get_gradient_input());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const avg_pool_& item, std::ostream& out)
        {
            serialize("avg_pool_2", out);
            serialize(_nr, out);
            serialize(_nc, out);
            serialize(_stride_y, out);
            serialize(_stride_x, out);
            serialize(item.padding_y_, out);
            serialize(item.padding_x_, out);
        }

        friend void deserialize(avg_pool_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);

            long nr;
            long nc;
            int stride_y;
            int stride_x;
            if (version == "avg_pool_2")
            {
                deserialize(nr, in);
                deserialize(nc, in);
                deserialize(stride_y, in);
                deserialize(stride_x, in);
                deserialize(item.padding_y_, in);
                deserialize(item.padding_x_, in);
            }
            else
            {
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::avg_pool_.");
            }

            if (item.padding_y_ != _padding_y) throw serialization_error("Wrong padding_y found while deserializing dlib::avg_pool_");
            if (item.padding_x_ != _padding_x) throw serialization_error("Wrong padding_x found while deserializing dlib::avg_pool_");
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
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            return out;
        }

        friend void to_xml(const avg_pool_& item, std::ostream& out)
        {
            out << "<avg_pool"
                << " nr='"<<_nr<<"'"
                << " nc='"<<_nc<<"'"
                << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<item.padding_y_<<"'"
                << " padding_x='"<<item.padding_x_<<"'"
                << "/>\n";
        }
    private:

        tt::pooling ap;
        resizable_tensor params;

        int padding_y_;
        int padding_x_;
    };

    template <
        long nr,
        long nc,
        int stride_y,
        int stride_x,
        typename SUBNET
        >
    using avg_pool = add_layer<avg_pool_<nr,nc,stride_y,stride_x>, SUBNET>;

    template <
        typename SUBNET
        >
    using avg_pool_everything = add_layer<avg_pool_<0,0,1,1>, SUBNET>;

// ----------------------------------------------------------------------------------------

    const double DEFAULT_LAYER_NORM_EPS = 1e-5;

    class layer_norm_
    {
    public:
        explicit layer_norm_(
            double eps_ = DEFAULT_LAYER_NORM_EPS
        ) :
            learning_rate_multiplier(1),
            weight_decay_multiplier(0),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(1),
            eps(eps_)
        {
        }

        double get_eps() const { return eps; }

        double get_learning_rate_multiplier () const  { return learning_rate_multiplier; }
        double get_weight_decay_multiplier () const   { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val)  { weight_decay_multiplier  = val; }

        double get_bias_learning_rate_multiplier () const  { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier () const   { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val)  { bias_weight_decay_multiplier  = val; }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            gamma = alias_tensor(1, sub.get_output().k(), sub.get_output().nr(), sub.get_output().nc());
            beta = gamma;

            params.set_size(gamma.size()+beta.size());

            gamma(params,0) = 1;
            beta(params,gamma.size()) = 0;
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());
            tt::layer_normalize(eps, output, means, invstds, sub.get_output(), g, b);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            auto g = gamma(params, 0);
            auto g_grad = gamma(params_grad, 0);
            auto b_grad = beta(params_grad, gamma.size());
            tt::layer_normalize_gradient(eps, gradient_input, means, invstds, sub.get_output(), g, sub.get_gradient_input(), g_grad, b_grad);
        }

        const tensor& get_layer_params() const { return params; };
        tensor& get_layer_params() { return params; };

        friend void serialize(const layer_norm_& item, std::ostream& out)
        {
            serialize("layer_norm_", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.beta, out);
            serialize(item.means, out);
            serialize(item.invstds, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.weight_decay_multiplier, out);
            serialize(item.bias_learning_rate_multiplier, out);
            serialize(item.bias_weight_decay_multiplier, out);
            serialize(item.eps, out);
        }

        friend void deserialize(layer_norm_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "layer_norm_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::layer_norm_.");
            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.beta, in);
            deserialize(item.means, in);
            deserialize(item.invstds, in);
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.weight_decay_multiplier, in);
            deserialize(item.bias_learning_rate_multiplier, in);
            deserialize(item.bias_weight_decay_multiplier, in);
            deserialize(item.eps, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const layer_norm_& item)
        {
            out << "layer_norm";
            out << " eps="<<item.eps;
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
            out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            return out;
        }

        friend void to_xml(const layer_norm_& item, std::ostream& out)
        {
            out << "<layer_norm";
            out << " eps='"<<item.eps<<"'";
            out << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'";
            out << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'";
            out << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'";
            out << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'";
            out << ">\n";
            out << mat(item.params);
            out << "</layer_norm>\n";
        }

    private:
        resizable_tensor params;
        alias_tensor gamma, beta;
        resizable_tensor means, invstds;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        double eps;
    };

    template <typename SUBNET>
    using layer_norm = add_layer<layer_norm_, SUBNET>;

// ----------------------------------------------------------------------------------------
    enum layer_mode
    {
        CONV_MODE = 0,
        FC_MODE = 1
    };

    const double DEFAULT_BATCH_NORM_EPS = 0.0001;

    template <
        layer_mode mode
        >
    class bn_
    {
    public:
        explicit bn_(
            unsigned long window_size,
            double eps_ = DEFAULT_BATCH_NORM_EPS
        ) : 
            num_updates(0), 
            running_stats_window_size(window_size),
            learning_rate_multiplier(1),
            weight_decay_multiplier(0),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(1),
            eps(eps_)
        {
            DLIB_CASSERT(window_size > 0, "The batch normalization running stats window size can't be 0.");
        }

        bn_() : bn_(100) {}

        layer_mode get_mode() const { return mode; }
        unsigned long get_running_stats_window_size () const { return running_stats_window_size; }
        void set_running_stats_window_size (unsigned long new_window_size ) 
        { 
            DLIB_CASSERT(new_window_size > 0, "The batch normalization running stats window size can't be 0.");
            running_stats_window_size = new_window_size; 
        }
        double get_eps() const { return eps; }

        double get_learning_rate_multiplier () const  { return learning_rate_multiplier; }
        double get_weight_decay_multiplier () const   { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val)  { weight_decay_multiplier  = val; }

        double get_bias_learning_rate_multiplier () const  { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier () const   { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val)  { bias_weight_decay_multiplier  = val; }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }


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
                ++num_updates;
                if (num_updates > running_stats_window_size)
                    num_updates = running_stats_window_size;

                if (mode == FC_MODE)
                    tt::batch_normalize(eps, output, means, invstds, decay, running_means, running_variances, sub.get_output(), g, b);
                else 
                    tt::batch_normalize_conv(eps, output, means, invstds, decay, running_means, running_variances, sub.get_output(), g, b);
            }
            else // we are running in testing mode so we just linearly scale the input tensor.
            {
                if (mode == FC_MODE)
                    tt::batch_normalize_inference(eps, output, sub.get_output(), g, b, running_means, running_variances);
                else
                    tt::batch_normalize_conv_inference(eps, output, sub.get_output(), g, b, running_means, running_variances);
            }
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            auto g = gamma(params,0);
            auto g_grad = gamma(params_grad, 0);
            auto b_grad = beta(params_grad, gamma.size());
            if (mode == FC_MODE)
                tt::batch_normalize_gradient(eps, gradient_input, means, invstds, sub.get_output(), g, sub.get_gradient_input(), g_grad, b_grad );
            else
                tt::batch_normalize_conv_gradient(eps, gradient_input, means, invstds, sub.get_output(), g, sub.get_gradient_input(), g_grad, b_grad );
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const bn_& item, std::ostream& out)
        {
            if (mode == CONV_MODE)
                serialize("bn_con2", out);
            else // if FC_MODE
                serialize("bn_fc2", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.beta, out);
            serialize(item.means, out);
            serialize(item.invstds, out);
            serialize(item.running_means, out);
            serialize(item.running_variances, out);
            serialize(item.num_updates, out);
            serialize(item.running_stats_window_size, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.weight_decay_multiplier, out);
            serialize(item.bias_learning_rate_multiplier, out);
            serialize(item.bias_weight_decay_multiplier, out);
            serialize(item.eps, out);
        }

        friend void deserialize(bn_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (mode == CONV_MODE) 
            {
                if (version != "bn_con2")
                    throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::bn_.");
            }
            else // must be in FC_MODE
            {
                if (version != "bn_fc2")
                    throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::bn_.");
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
            deserialize(item.learning_rate_multiplier, in);
            deserialize(item.weight_decay_multiplier, in);
            deserialize(item.bias_learning_rate_multiplier, in);
            deserialize(item.bias_weight_decay_multiplier, in);
            deserialize(item.eps, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const bn_& item)
        {
            if (mode == CONV_MODE)
                out << "bn_con  ";
            else
                out << "bn_fc   ";
            out << " eps="<<item.eps;
            out << " running_stats_window_size="<<item.running_stats_window_size;
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
            out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            return out;
        }

        friend void to_xml(const bn_& item, std::ostream& out)
        {
            if (mode==CONV_MODE)
                out << "<bn_con";
            else
                out << "<bn_fc";

            out << " eps='"<<item.eps<<"'";
            out << " running_stats_window_size='"<<item.running_stats_window_size<<"'";
            out << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'";
            out << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'";
            out << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'";
            out << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'";
            out << ">\n";

            out << mat(item.params);

            if (mode==CONV_MODE)
                out << "</bn_con>\n";
            else
                out << "</bn_fc>\n";
        }

    private:

        friend class affine_;

        resizable_tensor params;
        alias_tensor gamma, beta;
        resizable_tensor means, running_means;
        resizable_tensor invstds, running_variances;
        unsigned long num_updates;
        unsigned long running_stats_window_size;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        double eps;
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
        fc_(num_fc_outputs o) : num_outputs(o.num_outputs), num_inputs(0),
            learning_rate_multiplier(1),
            weight_decay_multiplier(1),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(0),
            use_bias(true)
        {}

        fc_() : fc_(num_fc_outputs(num_outputs_)) {}

        double get_learning_rate_multiplier () const  { return learning_rate_multiplier; }
        double get_weight_decay_multiplier () const   { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val)  { weight_decay_multiplier  = val; }

        double get_bias_learning_rate_multiplier () const  { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier () const   { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val)  { bias_weight_decay_multiplier  = val; }
        void disable_bias() { use_bias = false; }
        bool bias_is_disabled() const { return !use_bias; }

        unsigned long get_num_outputs (
        ) const { return num_outputs; }

        void set_num_outputs(long num) 
        {
            DLIB_CASSERT(num > 0);
            if (num != (long)num_outputs)
            {
                DLIB_CASSERT(get_layer_params().size() == 0, 
                    "You can't change the number of filters in fc_ if the parameter tensor has already been allocated.");
                num_outputs = num;
            }
        }

        fc_bias_mode get_bias_mode (
        ) const { return bias_mode; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            num_inputs = sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k();
            if (bias_mode == FC_HAS_BIAS && use_bias)
                params.set_size(num_inputs+1, num_outputs);
            else
                params.set_size(num_inputs, num_outputs);

            dlib::rand rnd(std::rand());
            randomize_parameters(params, num_inputs+num_outputs, rnd);

            weights = alias_tensor(num_inputs, num_outputs);

            if (bias_mode == FC_HAS_BIAS && use_bias)
            {
                biases = alias_tensor(1,num_outputs);
                // set the initial bias values to zero
                biases(params,weights.size()) = 0;
            }
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            DLIB_CASSERT((long)num_inputs == sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k(),
                "The size of the input tensor to this fc layer doesn't match the size the fc layer was trained with.");
            output.set_size(sub.get_output().num_samples(), num_outputs);

            auto w = weights(params, 0);
            tt::gemm(0,output, 1,sub.get_output(),false, w,false);
            if (bias_mode == FC_HAS_BIAS && use_bias)
            {
                auto b = biases(params, weights.size());
                tt::add(1,output,1,b);
            }
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            // no point computing the parameter gradients if they won't be used.
            if (learning_rate_multiplier != 0)
            {
                // compute the gradient of the weight parameters.  
                auto pw = weights(params_grad, 0);
                tt::gemm(0,pw, 1,sub.get_output(),true, gradient_input,false);

                if (bias_mode == FC_HAS_BIAS && use_bias)
                {
                    // compute the gradient of the bias parameters.  
                    auto pb = biases(params_grad, weights.size());
                    tt::assign_bias_gradient(pb, gradient_input);
                }
            }

            // compute the gradient for the data
            auto w = weights(params, 0);
            tt::gemm(1,sub.get_gradient_input(), 1,gradient_input,false, w,true);
        }

        alias_tensor_instance get_weights()
        {
            return weights(params, 0);
        }

        alias_tensor_const_instance get_weights() const
        {
            return weights(params, 0);
        }

        alias_tensor_instance get_biases()
        {
            static_assert(bias_mode == FC_HAS_BIAS, "This fc_ layer doesn't have a bias vector "
                "to be retrieved, as per template parameter 'bias_mode'.");
            return biases(params, weights.size());
        }

        alias_tensor_const_instance get_biases() const
        {
            static_assert(bias_mode == FC_HAS_BIAS, "This fc_ layer doesn't have a bias vector "
                "to be retrieved, as per template parameter 'bias_mode'.");
            return biases(params, weights.size());
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const fc_& item, std::ostream& out)
        {
            serialize("fc_3", out);
            serialize(item.num_outputs, out);
            serialize(item.num_inputs, out);
            serialize(item.params, out);
            serialize(item.weights, out);
            serialize(item.biases, out);
            serialize((int)bias_mode, out);
            serialize(item.learning_rate_multiplier, out);
            serialize(item.weight_decay_multiplier, out);
            serialize(item.bias_learning_rate_multiplier, out);
            serialize(item.bias_weight_decay_multiplier, out);
            serialize(item.use_bias, out);
        }

        friend void deserialize(fc_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "fc_2" || version == "fc_3")
            {
                deserialize(item.num_outputs, in);
                deserialize(item.num_inputs, in);
                deserialize(item.params, in);
                deserialize(item.weights, in);
                deserialize(item.biases, in);
                int bmode = 0;
                deserialize(bmode, in);
                if (bias_mode != (fc_bias_mode)bmode) throw serialization_error("Wrong fc_bias_mode found while deserializing dlib::fc_");
                deserialize(item.learning_rate_multiplier, in);
                deserialize(item.weight_decay_multiplier, in);
                deserialize(item.bias_learning_rate_multiplier, in);
                deserialize(item.bias_weight_decay_multiplier, in);
                if (version == "fc_3")
                {
                    deserialize(item.use_bias, in);
                }
            }
            else
            {
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::fc_.");
            }
        }

        friend std::ostream& operator<<(std::ostream& out, const fc_& item)
        {
            if (bias_mode == FC_HAS_BIAS)
            {
                out << "fc\t ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
                out << " learning_rate_mult="<<item.learning_rate_multiplier;
                out << " weight_decay_mult="<<item.weight_decay_multiplier;
                if (item.use_bias)
                {
                    out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                    out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
                }
                else
                {
                    out << " use_bias=false";
                }
            }
            else
            {
                out << "fc_no_bias ("
                    << "num_outputs="<<item.num_outputs
                    << ")";
                out << " learning_rate_mult="<<item.learning_rate_multiplier;
                out << " weight_decay_mult="<<item.weight_decay_multiplier;
            }
            return out;
        }

        friend void to_xml(const fc_& item, std::ostream& out)
        {
            if (bias_mode==FC_HAS_BIAS)
            {
                out << "<fc"
                    << " num_outputs='"<<item.num_outputs<<"'"
                    << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                    << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'"
                    << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                    << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'"
                    << " use_bias='"<<(item.use_bias?"true":"false")<<"'>\n";
                out << mat(item.params);
                out << "</fc>\n";
            }
            else
            {
                out << "<fc_no_bias"
                    << " num_outputs='"<<item.num_outputs<<"'"
                    << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                    << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'";
                out << ">\n";
                out << mat(item.params);
                out << "</fc_no_bias>\n";
            }
        }

    private:

        unsigned long num_outputs;
        unsigned long num_inputs;
        resizable_tensor params;
        alias_tensor weights, biases;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;
        bool use_bias;
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
            drop_rate(drop_rate_),
            rnd(std::rand())
        {
            DLIB_CASSERT(0 <= drop_rate && drop_rate <= 1);
        }

        // We have to add a copy constructor and assignment operator because the rnd object
        // is non-copyable.
        dropout_(
            const dropout_& item
        ) : drop_rate(item.drop_rate), mask(item.mask), rnd(std::rand())
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
            tt::multiply(false, output, input, mask);
        } 

        void backward_inplace(
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& /*params_grad*/
        )
        {
            if (is_same_object(gradient_input, data_grad))
                tt::multiply(false, data_grad, mask, gradient_input);
            else
                tt::multiply(true, data_grad, mask, gradient_input);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

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

        void clean(
        ) 
        {
            mask.clear();
        }

        friend std::ostream& operator<<(std::ostream& out, const dropout_& item)
        {
            out << "dropout\t ("
                << "drop_rate="<<item.drop_rate
                << ")";
            return out;
        }

        friend void to_xml(const dropout_& item, std::ostream& out)
        {
            out << "<dropout"
                << " drop_rate='"<<item.drop_rate<<"'";
            out << "/>\n";
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
            tt::affine_transform(output, input, val);
        } 

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        void backward_inplace(
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& /*params_grad*/
        )
        {
            if (is_same_object(gradient_input, data_grad))
                tt::affine_transform(data_grad, gradient_input, val);
            else
                tt::affine_transform(data_grad, data_grad, gradient_input, 1, val);
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

        friend void to_xml(const multiply_& item, std::ostream& out)
        {
            out << "<multiply"
                << " val='"<<item.val<<"'";
            out << "/>\n";
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

            g = pointwise_divide(mat(sg), sqrt(mat(item.running_variances)+item.get_eps()));
            b = mat(sb) - pointwise_multiply(mat(g), mat(item.running_means));
        }

        layer_mode get_mode() const { return mode; }

        void disable()
        {
            params.clear();
            disabled = true;
        }

        bool is_disabled() const { return disabled; }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            if (disabled)
                return;

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
            if (disabled)
                return;

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
            if (disabled)
                return;

            auto g = gamma(params,0);
            auto b = beta(params,gamma.size());

            // We are computing the gradient of dot(gradient_input, computed_output*g + b)
            if (mode == FC_MODE)
            {
                if (is_same_object(gradient_input, data_grad))
                    tt::multiply(false, data_grad, gradient_input, g);
                else
                    tt::multiply(true, data_grad, gradient_input, g);
            }
            else
            {
                if (is_same_object(gradient_input, data_grad))
                    tt::multiply_conv(false, data_grad, gradient_input, g);
                else
                    tt::multiply_conv(true, data_grad, gradient_input, g);
            }
        }

        alias_tensor_instance get_gamma() { return gamma(params, 0); };
        alias_tensor_const_instance get_gamma() const { return gamma(params, 0); };

        alias_tensor_instance get_beta() { return beta(params, gamma.size()); };
        alias_tensor_const_instance get_beta() const { return beta(params, gamma.size()); };

        const tensor& get_layer_params() const { return empty_params; }
        tensor& get_layer_params() { return empty_params; }

        friend void serialize(const affine_& item, std::ostream& out)
        {
            serialize("affine_2", out);
            serialize(item.params, out);
            serialize(item.gamma, out);
            serialize(item.beta, out);
            serialize((int)item.mode, out);
            serialize(item.disabled, out);
        }

        friend void deserialize(affine_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "bn_con2")
            {
                // Since we can build an affine_ from a bn_ we check if that's what is in
                // the stream and if so then just convert it right here.
                unserialize sin(version, in);
                bn_<CONV_MODE> temp;
                deserialize(temp, sin);
                item = temp;
                return;
            }
            else if (version == "bn_fc2")
            {
                // Since we can build an affine_ from a bn_ we check if that's what is in
                // the stream and if so then just convert it right here.
                unserialize sin(version, in);
                bn_<FC_MODE> temp;
                deserialize(temp, sin);
                item = temp;
                return;
            }

            if (version != "affine_" && version != "affine_2")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::affine_.");
            deserialize(item.params, in);
            deserialize(item.gamma, in);
            deserialize(item.beta, in);
            int mode;
            deserialize(mode, in);
            item.mode = (layer_mode)mode;
            if (version == "affine_2")
                deserialize(item.disabled, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const affine_& item)
        {
            out << "affine";
            if (item.disabled)
                out << "\t (disabled)";
            return out;
        }

        friend void to_xml(const affine_& item, std::ostream& out)
        {
            if (item.mode==CONV_MODE)
                out << "<affine_con";
            else
                out << "<affine_fc";
            if (item.disabled)
                out << " disabled='"<< std::boolalpha << item.disabled << "'";
            out << ">\n";
            out << mat(item.params);

            if (item.mode==CONV_MODE)
                out << "</affine_con>\n";
            else
                out << "</affine_fc>\n";
        }

    private:
        resizable_tensor params, empty_params; 
        alias_tensor gamma, beta;
        layer_mode mode;
        bool disabled = false;
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
        const static unsigned long id = tag_id<tag>::id;

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
            auto&& t1 = sub.get_output();
            auto&& t2 = layer<tag>(sub).get_output();
            output.set_size(std::max(t1.num_samples(),t2.num_samples()),
                            std::max(t1.k(),t2.k()),
                            std::max(t1.nr(),t2.nr()),
                            std::max(t1.nc(),t2.nc()));
            tt::add(output, t1, t2);
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

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        friend void serialize(const add_prev_& /*item*/, std::ostream& out)
        {
            serialize("add_prev_", out);
        }

        friend void deserialize(add_prev_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "add_prev_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::add_prev_.");
        }
        friend std::ostream& operator<<(std::ostream& out, const add_prev_& /*item*/)
        {
            out << "add_prev"<<id;
            return out;
        }

        friend void to_xml(const add_prev_& /*item*/, std::ostream& out)
        {
            out << "<add_prev tag='"<<id<<"'/>\n";
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

    template <
        template<typename> class tag
        >
    class mult_prev_
    {
    public:
        const static unsigned long id = tag_id<tag>::id;

        mult_prev_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto&& t1 = sub.get_output();
            auto&& t2 = layer<tag>(sub).get_output();
            output.set_size(std::max(t1.num_samples(),t2.num_samples()),
                            std::max(t1.k(),t2.k()),
                            std::max(t1.nr(),t2.nr()),
                            std::max(t1.nc(),t2.nc()));
            tt::multiply_zero_padded(false, output, t1, t2);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto&& t1 = sub.get_output();
            auto&& t2 = layer<tag>(sub).get_output();
            // The gradient just flows backwards to the two layers that forward()
            // multiplied together.
            tt::multiply_zero_padded(true, sub.get_gradient_input(), t2, gradient_input);
            tt::multiply_zero_padded(true, layer<tag>(sub).get_gradient_input(), t1, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        friend void serialize(const mult_prev_& /*item*/, std::ostream& out)
        {
            serialize("mult_prev_", out);
        }

        friend void deserialize(mult_prev_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "mult_prev_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::mult_prev_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const mult_prev_& /*item*/)
        {
            out << "mult_prev"<<id;
            return out;
        }

        friend void to_xml(const mult_prev_& /*item*/, std::ostream& out)
        {
            out << "<mult_prev tag='"<<id<<"'/>\n";
        }

    private:
        resizable_tensor params;
    };

    template <
        template<typename> class tag,
        typename SUBNET
        >
    using mult_prev = add_layer<mult_prev_<tag>, SUBNET>;

    template <typename SUBNET> using mult_prev1  = mult_prev<tag1, SUBNET>;
    template <typename SUBNET> using mult_prev2  = mult_prev<tag2, SUBNET>;
    template <typename SUBNET> using mult_prev3  = mult_prev<tag3, SUBNET>;
    template <typename SUBNET> using mult_prev4  = mult_prev<tag4, SUBNET>;
    template <typename SUBNET> using mult_prev5  = mult_prev<tag5, SUBNET>;
    template <typename SUBNET> using mult_prev6  = mult_prev<tag6, SUBNET>;
    template <typename SUBNET> using mult_prev7  = mult_prev<tag7, SUBNET>;
    template <typename SUBNET> using mult_prev8  = mult_prev<tag8, SUBNET>;
    template <typename SUBNET> using mult_prev9  = mult_prev<tag9, SUBNET>;
    template <typename SUBNET> using mult_prev10 = mult_prev<tag10, SUBNET>;

    using mult_prev1_  = mult_prev_<tag1>;
    using mult_prev2_  = mult_prev_<tag2>;
    using mult_prev3_  = mult_prev_<tag3>;
    using mult_prev4_  = mult_prev_<tag4>;
    using mult_prev5_  = mult_prev_<tag5>;
    using mult_prev6_  = mult_prev_<tag6>;
    using mult_prev7_  = mult_prev_<tag7>;
    using mult_prev8_  = mult_prev_<tag8>;
    using mult_prev9_  = mult_prev_<tag9>;
    using mult_prev10_ = mult_prev_<tag10>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class tag
        >
    class resize_prev_to_tagged_
    {
    public:
        const static unsigned long id = tag_id<tag>::id;

        resize_prev_to_tagged_()
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto& prev = sub.get_output();
            auto& tagged = layer<tag>(sub).get_output();

            DLIB_CASSERT(prev.num_samples() == tagged.num_samples());

            output.set_size(prev.num_samples(),
                            prev.k(),
                            tagged.nr(),
                            tagged.nc());

            if (prev.nr() == tagged.nr() && prev.nc() == tagged.nc())
            {
                tt::copy_tensor(false, output, 0, prev, 0, prev.k());
            }
            else
            {
                tt::resize_bilinear(output, prev);
            }
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto& prev = sub.get_gradient_input();

            DLIB_CASSERT(prev.k() == gradient_input.k());
            DLIB_CASSERT(prev.num_samples() == gradient_input.num_samples());

            if (prev.nr() == gradient_input.nr() && prev.nc() == gradient_input.nc())
            {
                tt::copy_tensor(true, prev, 0, gradient_input, 0, prev.k());
            }
            else
            {
                tt::resize_bilinear_gradient(prev, gradient_input);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        friend void serialize(const resize_prev_to_tagged_& /*item*/, std::ostream& out)
        {
            serialize("resize_prev_to_tagged_", out);
        }

        friend void deserialize(resize_prev_to_tagged_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "resize_prev_to_tagged_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::resize_prev_to_tagged_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const resize_prev_to_tagged_& /*item*/)
        {
            out << "resize_prev_to_tagged"<<id;
            return out;
        }

        friend void to_xml(const resize_prev_to_tagged_& /*item*/, std::ostream& out)
        {
            out << "<resize_prev_to_tagged tag='"<<id<<"'/>\n";
        }

    private:
        resizable_tensor params;
    };

    template <
        template<typename> class tag,
        typename SUBNET
        >
    using resize_prev_to_tagged = add_layer<resize_prev_to_tagged_<tag>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class tag
        >
    class scale_
    {
    public:
        const static unsigned long id = tag_id<tag>::id;

        scale_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto&& scales = sub.get_output();
            auto&& src = layer<tag>(sub).get_output();
            DLIB_CASSERT(scales.num_samples() == src.num_samples() &&
                         scales.k()           == src.k() &&
                         scales.nr()          == 1 &&
                         scales.nc()          == 1, 
                         "scales.k(): " << scales.k() <<
                         "\nsrc.k(): " << src.k() 
                         );

            output.copy_size(src);
            tt::scale_channels(false, output, src, scales);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto&& scales = sub.get_output();
            auto&& src = layer<tag>(sub).get_output();
            // The gradient just flows backwards to the two layers that forward()
            // read from.
            tt::scale_channels(true, layer<tag>(sub).get_gradient_input(), gradient_input, scales);

            if (reshape_src.num_samples() != src.num_samples())
            {
                reshape_scales = alias_tensor(src.num_samples()*src.k());
                reshape_src = alias_tensor(src.num_samples()*src.k(),src.nr()*src.nc());
            }

            auto&& scales_grad = sub.get_gradient_input();
            auto sgrad = reshape_scales(scales_grad);
            tt::dot_prods(true, sgrad, reshape_src(src), reshape_src(gradient_input));
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const scale_& item, std::ostream& out)
        {
            serialize("scale_", out);
            serialize(item.reshape_scales, out);
            serialize(item.reshape_src, out);
        }

        friend void deserialize(scale_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "scale_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::scale_.");
            deserialize(item.reshape_scales, in);
            deserialize(item.reshape_src, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const scale_& /*item*/)
        {
            out << "scale"<<id;
            return out;
        }

        friend void to_xml(const scale_& /*item*/, std::ostream& out)
        {
            out << "<scale tag='"<<id<<"'/>\n";
        }

    private:
        alias_tensor reshape_scales;
        alias_tensor reshape_src;
        resizable_tensor params;
    };

    template <
        template<typename> class tag,
        typename SUBNET
        >
    using scale = add_layer<scale_<tag>, SUBNET>;

    template <typename SUBNET> using scale1  = scale<tag1, SUBNET>;
    template <typename SUBNET> using scale2  = scale<tag2, SUBNET>;
    template <typename SUBNET> using scale3  = scale<tag3, SUBNET>;
    template <typename SUBNET> using scale4  = scale<tag4, SUBNET>;
    template <typename SUBNET> using scale5  = scale<tag5, SUBNET>;
    template <typename SUBNET> using scale6  = scale<tag6, SUBNET>;
    template <typename SUBNET> using scale7  = scale<tag7, SUBNET>;
    template <typename SUBNET> using scale8  = scale<tag8, SUBNET>;
    template <typename SUBNET> using scale9  = scale<tag9, SUBNET>;
    template <typename SUBNET> using scale10 = scale<tag10, SUBNET>;

    using scale1_  = scale_<tag1>;
    using scale2_  = scale_<tag2>;
    using scale3_  = scale_<tag3>;
    using scale4_  = scale_<tag4>;
    using scale5_  = scale_<tag5>;
    using scale6_  = scale_<tag6>;
    using scale7_  = scale_<tag7>;
    using scale8_  = scale_<tag8>;
    using scale9_  = scale_<tag9>;
    using scale10_ = scale_<tag10>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class tag
        >
    class scale_prev_
    {
    public:
        const static unsigned long id = tag_id<tag>::id;

        scale_prev_()
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            auto&& src = sub.get_output();
            auto&& scales = layer<tag>(sub).get_output();
            DLIB_CASSERT(scales.num_samples() == src.num_samples() &&
                         scales.k()           == src.k() &&
                         scales.nr()          == 1 &&
                         scales.nc()          == 1,
                         "scales.k(): " << scales.k() <<
                         "\nsrc.k(): " << src.k()
                         );

            output.copy_size(src);
            tt::scale_channels(false, output, src, scales);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto&& src = sub.get_output();
            auto&& scales = layer<tag>(sub).get_output();
            tt::scale_channels(true, sub.get_gradient_input(), gradient_input, scales);

            if (reshape_src.num_samples() != src.num_samples())
            {
                reshape_scales = alias_tensor(src.num_samples()*src.k());
                reshape_src = alias_tensor(src.num_samples()*src.k(),src.nr()*src.nc());
            }

            auto&& scales_grad = layer<tag>(sub).get_gradient_input();
            auto sgrad = reshape_scales(scales_grad);
            tt::dot_prods(true, sgrad, reshape_src(src), reshape_src(gradient_input));
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        friend void serialize(const scale_prev_& item, std::ostream& out)
        {
            serialize("scale_prev_", out);
            serialize(item.reshape_scales, out);
            serialize(item.reshape_src, out);
        }

        friend void deserialize(scale_prev_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "scale_prev_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::scale_prev_.");
            deserialize(item.reshape_scales, in);
            deserialize(item.reshape_src, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const scale_prev_& /*item*/)
        {
            out << "scale_prev"<<id;
            return out;
        }

        friend void to_xml(const scale_prev_& /*item*/, std::ostream& out)
        {
            out << "<scale_prev tag='"<<id<<"'/>\n";
        }

    private:
        alias_tensor reshape_scales;
        alias_tensor reshape_src;
        resizable_tensor params;
    };

    template <
        template<typename> class tag,
        typename SUBNET
        >
    using scale_prev = add_layer<scale_prev_<tag>, SUBNET>;

    template <typename SUBNET> using scale_prev1  = scale_prev<tag1, SUBNET>;
    template <typename SUBNET> using scale_prev2  = scale_prev<tag2, SUBNET>;
    template <typename SUBNET> using scale_prev3  = scale_prev<tag3, SUBNET>;
    template <typename SUBNET> using scale_prev4  = scale_prev<tag4, SUBNET>;
    template <typename SUBNET> using scale_prev5  = scale_prev<tag5, SUBNET>;
    template <typename SUBNET> using scale_prev6  = scale_prev<tag6, SUBNET>;
    template <typename SUBNET> using scale_prev7  = scale_prev<tag7, SUBNET>;
    template <typename SUBNET> using scale_prev8  = scale_prev<tag8, SUBNET>;
    template <typename SUBNET> using scale_prev9  = scale_prev<tag9, SUBNET>;
    template <typename SUBNET> using scale_prev10 = scale_prev<tag10, SUBNET>;

    using scale_prev1_  = scale_prev_<tag1>;
    using scale_prev2_  = scale_prev_<tag2>;
    using scale_prev3_  = scale_prev_<tag3>;
    using scale_prev4_  = scale_prev_<tag4>;
    using scale_prev5_  = scale_prev_<tag5>;
    using scale_prev6_  = scale_prev_<tag6>;
    using scale_prev7_  = scale_prev_<tag7>;
    using scale_prev8_  = scale_prev_<tag8>;
    using scale_prev9_  = scale_prev_<tag9>;
    using scale_prev10_ = scale_prev_<tag10>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:
        relu_()
        {
        }

        void disable()
        {
            params.clear();
            disabled = true;
        }

        bool is_disabled() const { return disabled; }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            if (disabled)
                return;

            tt::relu(output, input);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& 
        )
        {
            if (disabled)
                return;

            tt::relu_gradient(data_grad, computed_output, gradient_input);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const relu_& item, std::ostream& out)
        {
            serialize("relu_2", out);
            serialize(item.disabled, out);
        }

        friend void deserialize(relu_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version == "relu_2")
            {
                deserialize(item.disabled, in);
                return;
            }
            if (version != "relu_" && version != "relu_2")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::relu_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const relu_& item)
        {
            out << "relu";
            if (item.disabled)
            {
                out << "\t (disabled)";
            }
            return out;
        }

        friend void to_xml(const relu_& item, std::ostream& out)
        {
            out << "<relu";
            if (item.disabled)
            {
                out << " disabled='"<< std::boolalpha << item.disabled << "'";
            }
            out << "/>\n";
        }

    private:
        resizable_tensor params;
        bool disabled = false;
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

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

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

        friend void to_xml(const prelu_& item, std::ostream& out)
        {
            out << "<prelu initial_param_value='"<<item.initial_param_value<<"'>\n";
            out << mat(item.params);
            out << "</prelu>\n";
        }

    private:
        resizable_tensor params;
        float initial_param_value;
    };

    template <typename SUBNET>
    using prelu = add_layer<prelu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class leaky_relu_
    {
    public:
        explicit leaky_relu_(
            float alpha_ = 0.01f
        ) : alpha(alpha_)
        {
        }

        float get_alpha(
        ) const {
            return alpha;
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::leaky_relu(output, input, alpha);
        }

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input,
            tensor& data_grad,
            tensor&
        )
        {
            tt::leaky_relu_gradient(data_grad, computed_output, gradient_input, alpha);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const leaky_relu_& item, std::ostream& out)
        {
            serialize("leaky_relu_", out);
            serialize(item.alpha, out);
        }

        friend void deserialize(leaky_relu_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "leaky_relu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::leaky_relu_.");
            deserialize(item.alpha, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const leaky_relu_& item)
        {
            out << "leaky_relu\t("
                << "alpha=" << item.alpha
                << ")";
            return out;
        }

        friend void to_xml(const leaky_relu_& item, std::ostream& out)
        {
            out << "<leaky_relu alpha='"<< item.alpha << "'/>\n";
        }

    private:
        resizable_tensor params;
        float alpha;
    };

    template <typename SUBNET>
    using leaky_relu = add_layer<leaky_relu_, SUBNET>;

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

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const sig_& /*item*/, std::ostream& out)
        {
            serialize("sig_", out);
        }

        friend void deserialize(sig_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "sig_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::sig_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const sig_& /*item*/)
        {
            out << "sig";
            return out;
        }

        friend void to_xml(const sig_& /*item*/, std::ostream& out)
        {
            out << "<sig/>\n";
        }


    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using sig = add_layer<sig_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class mish_
    {
    public:
        mish_()
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(
            const SUBNET& sub,
            resizable_tensor& data_output
        )
        {
            data_output.copy_size(sub.get_output());
            tt::mish(data_output, sub.get_output());
        }

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor&
        )
        {
            tt::mish_gradient(sub.get_gradient_input(), sub.get_output(), gradient_input);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const mish_& /*item*/, std::ostream& out)
        {
            serialize("mish_", out);
        }

        friend void deserialize(mish_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "mish_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::mish_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const mish_& /*item*/)
        {
            out << "mish";
            return out;
        }

        friend void to_xml(const mish_& /*item*/, std::ostream& out)
        {
            out << "<mish/>\n";
        }


    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using mish = add_layer<mish_, SUBNET>;

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

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

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

        friend void serialize(const htan_& /*item*/, std::ostream& out)
        {
            serialize("htan_", out);
        }

        friend void deserialize(htan_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "htan_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::htan_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const htan_& /*item*/)
        {
            out << "htan";
            return out;
        }

        friend void to_xml(const htan_& /*item*/, std::ostream& out)
        {
            out << "<htan/>\n";
        }


    private:
        resizable_tensor params;
    };


    template <typename SUBNET>
    using htan = add_layer<htan_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class clipped_relu_
    {
    public:
        clipped_relu_(
            const float ceiling_ = 6.0f
        ) : ceiling(ceiling_)
        {
        }

        float get_ceiling(
        ) const {
            return ceiling;
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::clipped_relu(output, input, ceiling);
        }

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input,
            tensor& data_grad,
            tensor&
        )
        {
            tt::clipped_relu_gradient(data_grad, computed_output, gradient_input, ceiling);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const clipped_relu_& item, std::ostream& out)
        {
            serialize("clipped_relu_", out);
            serialize(item.ceiling, out);
        }

        friend void deserialize(clipped_relu_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "clipped_relu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::clipped_relu_.");
            deserialize(item.ceiling, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const clipped_relu_& item)
        {
            out << "clipped_relu\t("
                << "ceiling=" << item.ceiling
                << ")";
            return out;
        }

        friend void to_xml(const clipped_relu_& item, std::ostream& out)
        {
            out << "<clipped_relu ceiling='" << item.ceiling << "'/>\n";
        }


    private:
        resizable_tensor params;
        float ceiling;
    };

    template <typename SUBNET>
    using clipped_relu = add_layer<clipped_relu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class elu_
    {
    public:
        elu_(
            const float alpha_ = 1.0f
        ) : alpha(alpha_)
        {
        }

        float get_alpha(
        ) const {
            return alpha;
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::elu(output, input, alpha);
        }

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input,
            tensor& data_grad,
            tensor&
        )
        {
            tt::elu_gradient(data_grad, computed_output, gradient_input, alpha);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const elu_& item, std::ostream& out)
        {
            serialize("elu_", out);
            serialize(item.alpha, out);
        }

        friend void deserialize(elu_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "elu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::elu_.");
            deserialize(item.alpha, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const elu_& item)
        {
            out << "elu\t ("
                << "alpha=" << item.alpha
                << ")";
            return out;
        }

        friend void to_xml(const elu_& item, std::ostream& out)
        {
            out << "<elu alpha='" << item.alpha << "'/>\n";
        }


    private:
        resizable_tensor params;
        float alpha;
    };

    template <typename SUBNET>
    using elu = add_layer<elu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class gelu_
    {
    public:
        gelu_()
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(
            const SUBNET& sub,
            resizable_tensor& data_output
        )
        {
            data_output.copy_size(sub.get_output());
            tt::gelu(data_output, sub.get_output());
        }

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor&
        )
        {
            tt::gelu_gradient(sub.get_gradient_input(), sub.get_output(), gradient_input);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const gelu_& /*item*/, std::ostream& out)
        {
            serialize("gelu_", out);
        }

        friend void deserialize(gelu_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "gelu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::gelu_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const gelu_& /*item*/)
        {
            out << "gelu";
            return out;
        }

        friend void to_xml(const gelu_& /*item*/, std::ostream& out)
        {
            out << "<gelu/>\n";
        }


    private:
        resizable_tensor params;
    };

    template <typename SUBNET>
    using gelu = add_layer<gelu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class smelu_
    {
    public:
        explicit smelu_(
            float beta_ = 1
        ) : beta(beta_)
        {
        }

        float get_beta(
        ) const {
            return beta;
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::smelu(output, input, beta);
        }

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input,
            tensor& data_grad,
            tensor&
        )
        {
            tt::smelu_gradient(data_grad, computed_output, gradient_input, beta);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const smelu_& item, std::ostream& out)
        {
            serialize("smelu_", out);
            serialize(item.beta, out);
        }

        friend void deserialize(smelu_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "smelu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::smelu_.");
            deserialize(item.beta, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const smelu_& item)
        {
            out << "smelu\t ("
                << "beta=" << item.beta
                << ")";
            return out;
        }

        friend void to_xml(const smelu_& item, std::ostream& out)
        {
            out << "<smelu beta='"<< item.beta << "'/>\n";
        }

    private:
        resizable_tensor params;
        float beta;
    };

    template <typename SUBNET>
    using smelu = add_layer<smelu_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class silu_
    {
    public:
        silu_(
        )
        {
        }

        template <typename SUBNET>
        void setup(const SUBNET& /*sub*/)
        {
        }

        template <typename SUBNET>
        void forward(
            const SUBNET& sub,
            resizable_tensor& data_ouput)
        {
            data_ouput.copy_size(sub.get_output());
            tt::silu(data_ouput, sub.get_output());
        }

        template <typename SUBNET>
        void backward(
            const tensor& gradient_input,
            SUBNET& sub,
            tensor&
        )
        {
            tt::silu_gradient(sub.get_gradient_input(), sub.get_output(), gradient_input);
        }

        inline dpoint map_input_to_output (const dpoint& p) const { return p; }
        inline dpoint map_output_to_input (const dpoint& p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const silu_& /*item*/, std::ostream& out)
        {
            serialize("silu_", out);
        }

        friend void deserialize(silu_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "silu_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::silu_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const silu_& /*item*/)
        {
            out << "silu";
            return out;
        }

        friend void to_xml(const silu_& /*item*/, std::ostream& out)
        {
            out << "<silu/>\n";
        }

    private:
        resizable_tensor params;
    };

    template <typename SUBNET>
    using silu = add_layer<silu_, SUBNET>;

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

        friend void serialize(const softmax_& /*item*/, std::ostream& out)
        {
            serialize("softmax_", out);
        }

        friend void deserialize(softmax_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "softmax_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::softmax_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const softmax_& /*item*/)
        {
            out << "softmax";
            return out;
        }

        friend void to_xml(const softmax_& /*item*/, std::ostream& out)
        {
            out << "<softmax/>\n";
        }

    private:
        resizable_tensor params;
    };

    template <typename SUBNET>
    using softmax = add_layer<softmax_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class softmax_all_
    {
    public:
        softmax_all_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::softmax_all(output, input);
        } 

        void backward_inplace(
            const tensor& computed_output,
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& 
        )
        {
            tt::softmax_all_gradient(data_grad, computed_output, gradient_input);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const softmax_all_& /*item*/, std::ostream& out)
        {
            serialize("softmax_all_", out);
        }

        friend void deserialize(softmax_all_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "softmax_all_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::softmax_all_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const softmax_all_& /*item*/)
        {
            out << "softmax_all";
            return out;
        }

        friend void to_xml(const softmax_all_& /*item*/, std::ostream& out)
        {
            out << "<softmax_all/>\n";
        }

    private:
        resizable_tensor params;
    };

    template <typename SUBNET>
    using softmax_all = add_layer<softmax_all_, SUBNET>;

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <template<typename> class TAG_TYPE, template<typename> class... TAG_TYPES>
        struct concat_helper_impl{

            constexpr static size_t tag_count() {return 1 + concat_helper_impl<TAG_TYPES...>::tag_count();}
            static void list_tags(std::ostream& out)
            {
                out << tag_id<TAG_TYPE>::id << (tag_count() > 1 ? "," : "");
                concat_helper_impl<TAG_TYPES...>::list_tags(out);
            }

            template<typename SUBNET>
            static void resize_out(resizable_tensor& out, const SUBNET& sub, long sum_k)
            {
                auto& t = layer<TAG_TYPE>(sub).get_output();
                concat_helper_impl<TAG_TYPES...>::resize_out(out, sub, sum_k + t.k());
            }
            template<typename SUBNET>
            static void concat(tensor& out, const SUBNET& sub, size_t k_offset)
            {
                auto& t = layer<TAG_TYPE>(sub).get_output();
                tt::copy_tensor(false, out, k_offset, t, 0, t.k());
                k_offset += t.k();
                concat_helper_impl<TAG_TYPES...>::concat(out, sub, k_offset);
            }
            template<typename SUBNET>
            static void split(const tensor& input, SUBNET& sub, size_t k_offset)
            {
                auto& t = layer<TAG_TYPE>(sub).get_gradient_input();
                tt::copy_tensor(true, t, 0, input, k_offset, t.k());
                k_offset += t.k();
                concat_helper_impl<TAG_TYPES...>::split(input, sub, k_offset);
            }
        };
        template <template<typename> class TAG_TYPE>
        struct concat_helper_impl<TAG_TYPE>{
            constexpr static size_t tag_count() {return 1;}
            static void list_tags(std::ostream& out) 
            { 
                out << tag_id<TAG_TYPE>::id;
            }

            template<typename SUBNET>
            static void resize_out(resizable_tensor& out, const SUBNET& sub, long sum_k)
            {
                auto& t = layer<TAG_TYPE>(sub).get_output();
                out.set_size(t.num_samples(), t.k() + sum_k, t.nr(), t.nc());
            }
            template<typename SUBNET>
            static void concat(tensor& out, const SUBNET& sub, size_t k_offset)
            {
                auto& t = layer<TAG_TYPE>(sub).get_output();
                tt::copy_tensor(false, out, k_offset, t, 0, t.k());
            }
            template<typename SUBNET>
            static void split(const tensor& input, SUBNET& sub, size_t k_offset)
            {
                auto& t = layer<TAG_TYPE>(sub).get_gradient_input();
                tt::copy_tensor(true, t, 0, input, k_offset, t.k());
            }
        };
    }
    // concat layer
    template<
        template<typename> class... TAG_TYPES
        >
    class concat_
    {
        static void list_tags(std::ostream& out) { impl::concat_helper_impl<TAG_TYPES...>::list_tags(out);};

    public:
        constexpr static size_t tag_count() {return impl::concat_helper_impl<TAG_TYPES...>::tag_count();};

        template <typename SUBNET>
        void setup (const SUBNET&)
        {
            // do nothing
        }
        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            // the total depth of result is the sum of depths from all tags
            impl::concat_helper_impl<TAG_TYPES...>::resize_out(output, sub, 0);

            // copy output from each tag into different part result
            impl::concat_helper_impl<TAG_TYPES...>::concat(output, sub, 0);
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor&)
        {
            // Gradient is split into parts for each tag layer
            impl::concat_helper_impl<TAG_TYPES...>::split(gradient_input, sub, 0);
        }

        dpoint map_input_to_output(dpoint p) const { return p; }
        dpoint map_output_to_input(dpoint p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const concat_& /*item*/, std::ostream& out)
        {
            serialize("concat_", out);
            size_t count = tag_count();
            serialize(count, out);
        }

        friend void deserialize(concat_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "concat_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::concat_.");
            size_t count_tags;
            deserialize(count_tags, in);
            if (count_tags != tag_count())
                throw serialization_error("Invalid count of tags "+ std::to_string(count_tags) +", expecting " +
                                          std::to_string(tag_count()) +
                                                  " found while deserializing dlib::concat_.");
        }

        friend std::ostream& operator<<(std::ostream& out, const concat_& /*item*/)
        {
            out << "concat\t (";
            list_tags(out);
            out << ")";
            return out;
        }

        friend void to_xml(const concat_& /*item*/, std::ostream& out)
        {
            out << "<concat tags='";
            list_tags(out);
            out << "'/>\n";
        }

    private:
        resizable_tensor params; // unused
    };


    // concat layer definitions
    template <template<typename> class TAG1,
            template<typename> class TAG2,
            typename SUBNET>
    using concat2 = add_layer<concat_<TAG1, TAG2>, SUBNET>;

    template <template<typename> class TAG1,
            template<typename> class TAG2,
            template<typename> class TAG3,
            typename SUBNET>
    using concat3 = add_layer<concat_<TAG1, TAG2, TAG3>, SUBNET>;

    template <template<typename> class TAG1,
            template<typename> class TAG2,
            template<typename> class TAG3,
            template<typename> class TAG4,
            typename SUBNET>
    using concat4 = add_layer<concat_<TAG1, TAG2, TAG3, TAG4>, SUBNET>;

    template <template<typename> class TAG1,
            template<typename> class TAG2,
            template<typename> class TAG3,
            template<typename> class TAG4,
            template<typename> class TAG5,
            typename SUBNET>
    using concat5 = add_layer<concat_<TAG1, TAG2, TAG3, TAG4, TAG5>, SUBNET>;

    // inception layer will use tags internally. If user will use tags too, some conflicts
    // possible to exclude them, here are new tags specially for inceptions
    template <typename SUBNET> using itag0  = add_tag_layer< 1000 + 0, SUBNET>;
    template <typename SUBNET> using itag1  = add_tag_layer< 1000 + 1, SUBNET>;
    template <typename SUBNET> using itag2  = add_tag_layer< 1000 + 2, SUBNET>;
    template <typename SUBNET> using itag3  = add_tag_layer< 1000 + 3, SUBNET>;
    template <typename SUBNET> using itag4  = add_tag_layer< 1000 + 4, SUBNET>;
    template <typename SUBNET> using itag5  = add_tag_layer< 1000 + 5, SUBNET>;
    // skip to inception input
    template <typename SUBNET> using iskip  = add_skip_layer< itag0, SUBNET>;

    // here are some templates to be used for creating inception layer groups
    template <template<typename>class B1,
            template<typename>class B2,
            typename SUBNET>
    using inception2 = concat2<itag1, itag2, itag1<B1<iskip< itag2<B2< itag0<SUBNET>>>>>>>;

    template <template<typename>class B1,
            template<typename>class B2,
            template<typename>class B3,
            typename SUBNET>
    using inception3 = concat3<itag1, itag2, itag3, itag1<B1<iskip< itag2<B2<iskip< itag3<B3<  itag0<SUBNET>>>>>>>>>>;

    template <template<typename>class B1,
            template<typename>class B2,
            template<typename>class B3,
            template<typename>class B4,
            typename SUBNET>
    using inception4 = concat4<itag1, itag2, itag3, itag4,
                itag1<B1<iskip< itag2<B2<iskip< itag3<B3<iskip<  itag4<B4<  itag0<SUBNET>>>>>>>>>>>>>;

    template <template<typename>class B1,
            template<typename>class B2,
            template<typename>class B3,
            template<typename>class B4,
            template<typename>class B5,
            typename SUBNET>
    using inception5 = concat5<itag1, itag2, itag3, itag4, itag5,
                itag1<B1<iskip< itag2<B2<iskip< itag3<B3<iskip<  itag4<B4<iskip<  itag5<B5<  itag0<SUBNET>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    const double DEFAULT_L2_NORM_EPS = 1e-5;

    class l2normalize_
    {
    public:
        explicit l2normalize_(
            double eps_ = DEFAULT_L2_NORM_EPS
        ) : 
            eps(eps_)
        {
        }

        double get_eps() const { return eps; }

        template <typename SUBNET>
        void setup (const SUBNET& /*sub*/)
        {
        }

        void forward_inplace(const tensor& input, tensor& output)
        {
            tt::inverse_norms(norm, input, eps);
            tt::scale_rows(output, input, norm);
        } 

        void backward_inplace(
            const tensor& computed_output, 
            const tensor& gradient_input, 
            tensor& data_grad, 
            tensor& /*params_grad*/
        )
        {
            if (is_same_object(gradient_input, data_grad))
            {
                tt::dot_prods(temp, gradient_input, computed_output);
                tt::scale_rows2(0, data_grad, gradient_input, computed_output, temp, norm);
            }
            else
            {
                tt::dot_prods(temp, gradient_input, computed_output);
                tt::scale_rows2(1, data_grad, gradient_input, computed_output, temp, norm);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const l2normalize_& item, std::ostream& out)
        {
            serialize("l2normalize_", out);
            serialize(item.eps, out);
        }

        friend void deserialize(l2normalize_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "l2normalize_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::l2normalize_.");
            deserialize(item.eps, in);
        }

        friend std::ostream& operator<<(std::ostream& out, const l2normalize_& item)
        {
            out << "l2normalize";
            out << " eps="<<item.eps;
            return out;
        }

        friend void to_xml(const l2normalize_& item, std::ostream& out)
        {
            out << "<l2normalize";
            out << " eps='"<<item.eps<<"'";
            out << "/>\n";
        }
    private:
        double eps;

        resizable_tensor params; // unused
        // Here only to avoid reallocation and as a cache between forward/backward
        // functions.  
        resizable_tensor norm;
        resizable_tensor temp;
    };

    template <typename SUBNET>
    using l2normalize = add_layer<l2normalize_, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        long _offset,
        long _k,
        long _nr,
        long _nc
        >
    class extract_
    {
        static_assert(_offset >= 0, "The offset must be >= 0.");
        static_assert(_k > 0,  "The number of channels must be > 0.");
        static_assert(_nr > 0, "The number of rows must be > 0.");
        static_assert(_nc > 0, "The number of columns must be > 0.");
    public:
        extract_(
        )  
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            DLIB_CASSERT((long)sub.get_output().size() >= sub.get_output().num_samples()*(_offset+_k*_nr*_nc), 
                "The tensor we are trying to extract from the input tensor is too big to fit into the input tensor.");

            aout = alias_tensor(sub.get_output().num_samples(), _k*_nr*_nc);
            ain = alias_tensor(sub.get_output().num_samples(),  sub.get_output().size()/sub.get_output().num_samples());
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            if (aout.num_samples() != sub.get_output().num_samples())
            {
                aout = alias_tensor(sub.get_output().num_samples(), _k*_nr*_nc);
                ain = alias_tensor(sub.get_output().num_samples(),  sub.get_output().size()/sub.get_output().num_samples());
            }

            output.set_size(sub.get_output().num_samples(), _k, _nr, _nc);
            auto out = aout(output,0);
            auto in = ain(sub.get_output(),0);
            tt::copy_tensor(false, out, 0, in, _offset, _k*_nr*_nc);
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            auto out = ain(sub.get_gradient_input(),0);
            auto in = aout(gradient_input,0);
            tt::copy_tensor(true, out, _offset, in, 0, _k*_nr*_nc);
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const extract_& /*item*/, std::ostream& out)
        {
            serialize("extract_", out);
            serialize(_offset, out);
            serialize(_k, out);
            serialize(_nr, out);
            serialize(_nc, out);
        }

        friend void deserialize(extract_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "extract_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::extract_.");

            long offset;
            long k;
            long nr;
            long nc;
            deserialize(offset, in);
            deserialize(k, in);
            deserialize(nr, in);
            deserialize(nc, in);

            if (offset != _offset) throw serialization_error("Wrong offset found while deserializing dlib::extract_");
            if (k != _k)   throw serialization_error("Wrong k found while deserializing dlib::extract_");
            if (nr != _nr) throw serialization_error("Wrong nr found while deserializing dlib::extract_");
            if (nc != _nc) throw serialization_error("Wrong nc found while deserializing dlib::extract_");
        }

        friend std::ostream& operator<<(std::ostream& out, const extract_& /*item*/)
        {
            out << "extract\t ("
                << "offset="<<_offset
                << ", k="<<_k
                << ", nr="<<_nr
                << ", nc="<<_nc
                << ")";
            return out;
        }

        friend void to_xml(const extract_& /*item*/, std::ostream& out)
        {
            out << "<extract";
            out << " offset='"<<_offset<<"'";
            out << " k='"<<_k<<"'";
            out << " nr='"<<_nr<<"'";
            out << " nc='"<<_nc<<"'";
            out << "/>\n";
        }
    private:
        alias_tensor aout, ain;

        resizable_tensor params; // unused
    };

    template <
        long offset,
        long k,
        long nr,
        long nc,
        typename SUBNET
        >
    using extract = add_layer<extract_<offset,k,nr,nc>, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <long long row_stride = 2, long long col_stride = 2>
    class reorg_
    {
        static_assert(row_stride >= 1, "The row_stride must be >= 1");
        static_assert(row_stride >= 1, "The col_stride must be >= 1");

    public:
        reorg_(
        )
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            DLIB_CASSERT(sub.get_output().nr() % row_stride == 0);
            DLIB_CASSERT(sub.get_output().nc() % col_stride == 0);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            output.set_size(
                sub.get_output().num_samples(),
                sub.get_output().k() * col_stride * row_stride,
                sub.get_output().nr() / row_stride,
                sub.get_output().nc() / col_stride
            );
            tt::reorg(output, row_stride, col_stride, sub.get_output());
        }

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& /*params_grad*/)
        {
            tt::reorg_gradient(sub.get_gradient_input(), row_stride, col_stride, gradient_input);
        }

        inline dpoint map_input_to_output (dpoint p) const
        {
            p.x() = p.x() / col_stride;
            p.y() = p.y() / row_stride;
            return p;
        }
        inline dpoint map_output_to_input (dpoint p) const
        {
            p.x() = p.x() * col_stride;
            p.y() = p.y() * row_stride;
            return p;
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const reorg_& /*item*/, std::ostream& out)
        {
            serialize("reorg_", out);
            serialize(row_stride, out);
            serialize(col_stride, out);
        }

        friend void deserialize(reorg_& /*item*/, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "reorg_")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::reorg_.");
            long long rs;
            long long cs;
            deserialize(rs, in);
            deserialize(cs, in);
            if (rs != row_stride) throw serialization_error("Wrong row_stride found while deserializing dlib::reorg_");
            if (cs != col_stride) throw serialization_error("Wrong col_stride found while deserializing dlib::reorg_");
        }

        friend std::ostream& operator<<(std::ostream& out, const reorg_& /*item*/)
        {
            out << "reorg\t ("
                << "row_stride=" << row_stride
                << ", col_stride=" << col_stride
                << ")";
            return out;
        }

        friend void to_xml(const reorg_ /*item*/, std::ostream& out)
        {
            out << "<reorg";
            out << " row_stride='" << row_stride << "'";
            out << " col_stride='" << col_stride << "'";
            out << "/>\n";
        }

    private:
        resizable_tensor params; // unused

    };

    template <typename SUBNET>
    using reorg = add_layer<reorg_<2, 2>, SUBNET>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_LAYERS_H_


