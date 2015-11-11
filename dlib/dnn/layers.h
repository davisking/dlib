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


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class con_
    {
    public:
        con_()
        {}

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            // TODO
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            // TODO
        } 

        template <typename SUBNET>
        void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            // TODO
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

    private:

        resizable_tensor params;
    };

    template <typename SUBNET>
    using con = add_layer<con_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class fc_
    {
    public:
        fc_() : num_outputs(1), num_inputs(0)
        {
        }

        explicit fc_(
            unsigned long num_outputs_
        ) : num_outputs(num_outputs_), num_inputs(0)
        {
        }

        unsigned long get_num_outputs (
        ) const { return num_outputs; }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
            num_inputs = sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k();
            params.set_size(num_inputs, num_outputs);

            dlib::rand rnd("fc_"+cast_to_string(num_outputs));
            randomize_parameters(params, num_inputs+num_outputs, rnd);
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            output.set_size(sub.get_output().num_samples(), num_outputs);

            output = mat(sub.get_output())*mat(params);
        } 

        template <typename SUBNET>
        void backward(const tensor& , const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            // compute the gradient of the parameters.  
            params_grad = trans(mat(sub.get_output()))*mat(gradient_input);

            // compute the gradient for the data
            sub.get_gradient_input() += mat(gradient_input)*trans(mat(params));
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

        friend void serialize(const fc_& item, std::ostream& out)
        {
            serialize("fc_", out);
            serialize(item.num_outputs, out);
            serialize(item.num_inputs, out);
            serialize(item.params, out);
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
        }

    private:

        unsigned long num_outputs;
        unsigned long num_inputs;
        resizable_tensor params;
    };


    template <typename SUBNET>
    using fc = add_layer<fc_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:
        relu_() 
        {
        }

        template <typename SUBNET>
        void setup (const SUBNET& sub)
        {
        }

        template <typename SUBNET>
        void forward(const SUBNET& sub, resizable_tensor& output)
        {
            output.copy_size(sub.get_output());
            output = lowerbound(mat(sub.get_output()), 0);
        } 

        template <typename SUBNET>
        void backward(const tensor&, const tensor& gradient_input, SUBNET& sub, tensor& params_grad)
        {
            const float* grad = gradient_input.host();
            const float* in = sub.get_output().host();
            float* out = sub.get_gradient_input().host();
            for (unsigned long i = 0; i < sub.get_output().size(); ++i)
            {
                if (in[i] > 0)
                    out[i] = grad[i];
                else
                    out[i] = 0;
            }

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

}

#endif // DLIB_DNn_LAYERS_H_


