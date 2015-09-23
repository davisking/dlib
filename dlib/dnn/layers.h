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

        template <typename SUB_NET>
        void setup (const SUB_NET& sub)
        {
            // TODO
        }

        template <typename SUB_NET>
        void forward(const SUB_NET& sub, resizable_tensor& output)
        {
            // TODO
        } 

        template <typename SUB_NET>
        void backward(const tensor& gradient_input, SUB_NET& sub, tensor& params_grad)
        {
            // TODO
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

    private:

        resizable_tensor params;
    };

    template <typename SUB_NET>
    using con = add_layer<con_, SUB_NET>;

// ----------------------------------------------------------------------------------------

    class fc_
    {
    public:
        fc_() : num_outputs(1)
        {
            rnd.set_seed("fc_" + cast_to_string(num_outputs));
        }

        explicit fc_(unsigned long num_outputs_)
        {
            num_outputs = num_outputs_;
            rnd.set_seed("fc_" + cast_to_string(num_outputs));
        }

        unsigned long get_num_outputs (
        ) const { return num_outputs; }

        template <typename SUB_NET>
        void setup (const SUB_NET& sub)
        {
            num_inputs = sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k();
            params.set_size(num_inputs, num_outputs);

            std::cout << "fc_::setup() " << params.size() << std::endl;

            randomize_parameters(params, num_inputs+num_outputs, rnd);
        }

        template <typename SUB_NET>
        void forward(const SUB_NET& sub, resizable_tensor& output)
        {
            output.set_size(sub.get_output().num_samples(), num_outputs);

            output = mat(sub.get_output())*mat(params);
        } 

        template <typename SUB_NET>
        void backward(const tensor& gradient_input, SUB_NET& sub, tensor& params_grad)
        {
            // d1*W*p1 + d2*W*p2
            // total gradient = [d1*W; d2*W; d3*W; ...] == D*W


            // compute the gradient of the parameters.  
            params_grad += trans(mat(sub.get_output()))*mat(gradient_input);

            // compute the gradient for the data
            sub.get_gradient_input() += mat(gradient_input)*trans(mat(params));
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

    private:

        unsigned long num_outputs;
        unsigned long num_inputs;
        resizable_tensor params;
        dlib::rand rnd;
    };


    template <typename SUB_NET>
    using fc = add_layer<fc_, SUB_NET>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:
        relu_() 
        {
        }

        template <typename SUB_NET>
        void setup (const SUB_NET& sub)
        {
        }

        template <typename SUB_NET>
        void forward(const SUB_NET& sub, resizable_tensor& output)
        {
            output.copy_size(sub.get_output());
            output = lowerbound(mat(sub.get_output()), 0);
        } 

        template <typename SUB_NET>
        void backward(const tensor& gradient_input, SUB_NET& sub, tensor& params_grad)
        {
            const float* grad = gradient_input.host();
            const float* in = sub.get_output().host();
            float* out = sub.get_gradient_input().host();
            for (unsigned long i = 0; i < sub.get_output().size(); ++i)
            {
                if (in[i] > 0)
                    out[i] += grad[i];
            }

        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

    private:

        resizable_tensor params;
    };


    template <typename SUB_NET>
    using relu = add_layer<relu_, SUB_NET>;

// ----------------------------------------------------------------------------------------

    class multiply_
    {
    public:
        multiply_() 
        {
        }


        template <typename SUB_NET>
        void setup (const SUB_NET& sub)
        {
            num_inputs = sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k();
            params.set_size(1, num_inputs);

            std::cout << "multiply_::setup() " << params.size() << std::endl;

            const int num_outputs = num_inputs;

            randomize_parameters(params, num_inputs+num_outputs, rnd);
        }

        template <typename SUB_NET>
        void forward(const SUB_NET& sub, resizable_tensor& output)
        {
            DLIB_CASSERT( sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k() == params.size(), "");
            DLIB_CASSERT( sub.get_output().nr()*sub.get_output().nc()*sub.get_output().k() == num_inputs, "");

            output.copy_size(sub.get_output());
            auto indata = sub.get_output().host();
            auto outdata = output.host();
            auto paramdata = params.host();
            for (int i = 0; i < sub.get_output().num_samples(); ++i)
            {
                for (int j = 0; j < num_inputs; ++j)
                {
                    *outdata++ = *indata++ * paramdata[j];
                }
            }
        } 

        template <typename SUB_NET>
        void backward(const tensor& gradient_input, SUB_NET& sub, tensor& params_grad)
        {
            params_grad += sum_rows(pointwise_multiply(mat(sub.get_output()),mat(gradient_input)));

            for (long i = 0; i < gradient_input.num_samples(); ++i)
            {
                sub.get_gradient_input().add_to_sample(i, 
                    pointwise_multiply(rowm(mat(gradient_input),i), mat(params)));
            }
        }

        const tensor& get_layer_params() const { return params; }
        tensor& get_layer_params() { return params; }

    private:

        int num_inputs;
        resizable_tensor params;
        dlib::rand rnd;
    };

    template <typename SUB_NET>
    using multiply = add_layer<multiply_, SUB_NET>;

// ----------------------------------------------------------------------------------------

}

#endif // #define DLIB_DNn_LAYERS_H_


