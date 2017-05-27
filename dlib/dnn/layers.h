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
#include "utilities.h"
#include <sstream>


namespace dlib
{

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
    class con_
    {
    public:

        static_assert(_num_filters > 0, "The number of filters must be > 0");
        static_assert(_nr > 0, "The number of rows in a filter must be > 0");
        static_assert(_nc > 0, "The number of columns in a filter must be > 0");
        static_assert(_stride_y > 0, "The filter stride must be > 0");
        static_assert(_stride_x > 0, "The filter stride must be > 0");
        static_assert(0 <= _padding_y && _padding_y < _nr, "The padding must be smaller than the filter size.");
        static_assert(0 <= _padding_x && _padding_x < _nc, "The padding must be smaller than the filter size.");

        con_(
        ) : 
            learning_rate_multiplier(1),
            weight_decay_multiplier(1),
            bias_learning_rate_multiplier(1),
            bias_weight_decay_multiplier(0),
            padding_y_(_padding_y),
            padding_x_(_padding_x)
        {}

        long num_filters() const { return _num_filters; }
        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long stride_y() const { return _stride_y; }
        long stride_x() const { return _stride_x; }
        long padding_y() const { return padding_y_; }
        long padding_x() const { return padding_x_; }

        double get_learning_rate_multiplier () const  { return learning_rate_multiplier; }
        double get_weight_decay_multiplier () const   { return weight_decay_multiplier; }
        void set_learning_rate_multiplier(double val) { learning_rate_multiplier = val; }
        void set_weight_decay_multiplier(double val)  { weight_decay_multiplier  = val; }

        double get_bias_learning_rate_multiplier () const  { return bias_learning_rate_multiplier; }
        double get_bias_weight_decay_multiplier () const   { return bias_weight_decay_multiplier; }
        void set_bias_learning_rate_multiplier(double val) { bias_learning_rate_multiplier = val; }
        void set_bias_weight_decay_multiplier(double val)  { bias_weight_decay_multiplier  = val; }

        inline point map_input_to_output (
            point p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline point map_output_to_input (
            point p
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
            padding_y_(item.padding_y_),
            padding_x_(item.padding_x_)
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
                _stride_x,
                padding_y_,
                padding_x_
                );

            tt::add(1,output,1,biases(params,filters.size()));
        } 

        template <typename SUBNET>
        void backward(const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            conv.get_gradient_for_data (gradient_input, filters(params,0), sub.get_gradient_input());
            // no point computing the parameter gradients if they won't be used.
            if (learning_rate_multiplier != 0)
            {
                auto filt = filters(params_grad,0);
                conv.get_gradient_for_filters (gradient_input, sub.get_output(), filt);
                auto b = biases(params_grad, filters.size());
                tt::assign_conv_bias_gradient(b, gradient_input);
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const con_& item, std::ostream& out)
        {
            serialize("con_4", out);
            serialize(item.params, out);
            serialize(_num_filters, out);
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
        }

        friend void deserialize(con_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            long num_filters;
            long nr;
            long nc;
            int stride_y;
            int stride_x;
            if (version == "con_4")
            {
                deserialize(item.params, in);
                deserialize(num_filters, in);
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
                if (num_filters != _num_filters) 
                {
                    std::ostringstream sout;
                    sout << "Wrong num_filters found while deserializing dlib::con_" << std::endl;
                    sout << "expected " << _num_filters << " but found " << num_filters << std::endl;
                    throw serialization_error(sout.str());
                }

                if (nr != _nr) throw serialization_error("Wrong nr found while deserializing dlib::con_");
                if (nc != _nc) throw serialization_error("Wrong nc found while deserializing dlib::con_");
                if (stride_y != _stride_y) throw serialization_error("Wrong stride_y found while deserializing dlib::con_");
                if (stride_x != _stride_x) throw serialization_error("Wrong stride_x found while deserializing dlib::con_");
            }
            else
            {
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::con_.");
            }
        }


        friend std::ostream& operator<<(std::ostream& out, const con_& item)
        {
            out << "con\t ("
                << "num_filters="<<_num_filters
                << ", nr="<<_nr
                << ", nc="<<_nc
                << ", stride_y="<<_stride_y
                << ", stride_x="<<_stride_x
                << ", padding_y="<<item.padding_y_
                << ", padding_x="<<item.padding_x_
                << ")";
            out << " learning_rate_mult="<<item.learning_rate_multiplier;
            out << " weight_decay_mult="<<item.weight_decay_multiplier;
            out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
            out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
            return out;
        }

        friend void to_xml(const con_& item, std::ostream& out)
        {
            out << "<con"
                << " num_filters='"<<_num_filters<<"'"
                << " nr='"<<_nr<<"'"
                << " nc='"<<_nc<<"'"
                << " stride_y='"<<_stride_y<<"'"
                << " stride_x='"<<_stride_x<<"'"
                << " padding_y='"<<item.padding_y_<<"'"
                << " padding_x='"<<item.padding_x_<<"'"
                << " learning_rate_mult='"<<item.learning_rate_multiplier<<"'"
                << " weight_decay_mult='"<<item.weight_decay_multiplier<<"'"
                << " bias_learning_rate_mult='"<<item.bias_learning_rate_multiplier<<"'"
                << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'>\n";
            out << mat(item.params);
            out << "</con>";
        }

    private:

        resizable_tensor params;
        alias_tensor filters, biases;

        tt::tensor_conv conv;
        double learning_rate_multiplier;
        double weight_decay_multiplier;
        double bias_learning_rate_multiplier;
        double bias_weight_decay_multiplier;

        // These are here only because older versions of con (which you might encounter
        // serialized to disk) used different padding settings.
        int padding_y_;
        int padding_x_;

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

        inline point map_input_to_output (
            point p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline point map_output_to_input (
            point p
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

        inline point map_input_to_output (
            point p
        ) const
        {
            p.x() = (p.x()+padding_x()-nc()/2)/stride_x();
            p.y() = (p.y()+padding_y()-nr()/2)/stride_y();
            return p;
        }

        inline point map_output_to_input (
            point p
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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }


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

    namespace impl
    {
        class visitor_bn_running_stats_window_size
        {
        public:

            visitor_bn_running_stats_window_size(unsigned long new_window_size_) : new_window_size(new_window_size_) {}

            template <typename T>
            void set_window_size(T&) const
            {
                // ignore other layer detail types
            }

            template < layer_mode mode >
            void set_window_size(bn_<mode>& l) const
            {
                l.set_running_stats_window_size(new_window_size);
            }

            template<typename input_layer_type>
            void operator()(size_t , input_layer_type& )  const
            {
                // ignore other layers
            }

            template <typename T, typename U, typename E>
            void operator()(size_t , add_layer<T,U,E>& l)  const
            {
                set_window_size(l.layer_details());
            }

        private:

            unsigned long new_window_size;
        };
    }

    template <typename net_type>
    void set_all_bn_running_stats_window_sizes (
        net_type& net,
        unsigned long new_window_size
    )
    {
        visit_layers(net, impl::visitor_bn_running_stats_window_size(new_window_size));
    }

// ----------------------------------------------------------------------------------------
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
            bias_weight_decay_multiplier(0)
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
            // no point computing the parameter gradients if they won't be used.
            if (learning_rate_multiplier != 0)
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
            serialize("fc_2", out);
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
        }

        friend void deserialize(fc_& item, std::istream& in)
        {
            std::string version;
            deserialize(version, in);
            if (version != "fc_2")
                throw serialization_error("Unexpected version '"+version+"' found while deserializing dlib::fc_.");

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
                out << " bias_learning_rate_mult="<<item.bias_learning_rate_multiplier;
                out << " bias_weight_decay_mult="<<item.bias_weight_decay_multiplier;
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
                    << " bias_weight_decay_mult='"<<item.bias_weight_decay_multiplier<<"'";
                out << ">\n";
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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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

            g = pointwise_multiply(mat(sg), 1.0f/sqrt(mat(item.running_variances)+item.get_eps()));
            b = mat(sb) - pointwise_multiply(mat(g), mat(item.running_means));
        }

        layer_mode get_mode() const { return mode; }

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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

        friend void to_xml(const affine_& item, std::ostream& out)
        {
            if (item.mode==CONV_MODE)
                out << "<affine_con>\n";
            else
                out << "<affine_fc>\n";

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
            out << "add_prev"<<id;
            return out;
        }

        friend void to_xml(const add_prev_& item, std::ostream& out)
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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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

        friend void to_xml(const relu_& /*item*/, std::ostream& out)
        {
            out << "<relu/>\n";
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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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

        inline point map_input_to_output (const point& p) const { return p; }
        inline point map_output_to_input (const point& p) const { return p; }

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
                tt::copy_tensor(out, k_offset, t, 0, t.k());
                k_offset += t.k();
                concat_helper_impl<TAG_TYPES...>::concat(out, sub, k_offset);
            }
            template<typename SUBNET>
            static void split(const tensor& input, SUBNET& sub, size_t k_offset)
            {
                auto& t = layer<TAG_TYPE>(sub).get_gradient_input();
                tt::copy_tensor(t, 0, input, k_offset, t.k());
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
                tt::copy_tensor(out, k_offset, t, 0, t.k());
            }
            template<typename SUBNET>
            static void split(const tensor& input, SUBNET& sub, size_t k_offset)
            {
                auto& t = layer<TAG_TYPE>(sub).get_gradient_input();
                tt::copy_tensor(t, 0, input, k_offset, t.k());
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

        point map_input_to_output(point p) const { return p; }
        point map_output_to_input(point p) const { return p; }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const concat_& item, std::ostream& out)
        {
            serialize("concat_", out);
            size_t count = tag_count();
            serialize(count, out);
        }

        friend void deserialize(concat_& item, std::istream& in)
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

        friend std::ostream& operator<<(std::ostream& out, const concat_& item)
        {
            out << "concat\t (";
            list_tags(out);
            out << ")";
            return out;
        }

        friend void to_xml(const concat_& item, std::ostream& out)
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

}

#endif // DLIB_DNn_LAYERS_H_


