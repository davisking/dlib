// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_LAYERS_ABSTRACT_H_
#ifdef DLIB_DNn_LAYERS_ABSTRACT_H_

#include "tensor_abstract.h"
#include "core_abstract.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class SUB_NET 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS

                By "Sub net" we mean the part of the network closer to the input.  Whenever
                you get a SUB_NET it will always have computed its outputs and they will be
                available in get_output().

        !*/

    public:

        const tensor& get_output(
        ) const;

        tensor& get_gradient_input(
        );

        const NEXT_SUB_NET& sub_net(
        ) const;

        NEXT_SUB_NET& sub_net(
        );
    };

// ----------------------------------------------------------------------------------------

    class EXAMPLE_LAYER_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                Each layer in a deep neural network can be thought of as a function,
                f(data,parameters), that takes in a data tensor, some parameters, and
                produces an output tensor.  You create an entire deep network by composing
                these functions.  Importantly, you are able to use a wide range of
                different functions to accommodate whatever task you are trying to accomplish.
                Dlib includes a number of common layer types but if you want to define your
                own then you simply implement a class with the same interface as EXAMPLE_LAYER_.

        !*/

    public:

        EXAMPLE_LAYER_(
        );
        /*!
            ensures
                - Default constructs this object.  This function is not required to do
                  anything in particular but it is required that layer objects be default
                  constructable. 
        !*/

        template <typename SUB_NET>
        void setup (
            const SUB_NET& sub
        );
        /*!
            requires
                - SUB_NET implements the SUB_NET interface defined at the top of this file.
            ensures
                - performs any necessary initial memory allocations and/or sets parameters
                  to their initial values prior to learning.  Therefore, calling setup
                  destroys any previously learned parameters.
        !*/

        template <typename SUB_NET>
        void forward(
            const SUB_NET& sub, 
            resizable_tensor& output
        );
        /*!
            requires
                - SUB_NET implements the SUB_NET interface defined at the top of this file.
                - setup() has been called.
            ensures
                - Runs the output of the sub-network through this layer and stores the
                  output into #output.  In particular, forward() can use any of the outputs
                  in sub (e.g. sub.get_output(), sub.sub_net().get_output(), etc.) to
                  compute whatever it wants.
                - #output.num_samples() == sub.get_output().num_samples()
        !*/

        template <typename SUB_NET>
        void backward(
            const tensor& gradient_input, 
            SUB_NET& sub, 
            tensor& params_grad
        );
        /*!
            requires
                - SUB_NET implements the SUB_NET interface defined at the top of this file.
                - setup() has been called.
                - gradient_input has the same dimensions as the output of forward(sub,output).
                - have_same_dimensions(sub.get_gradient_input(), sub.get_output()) == true
                - have_same_dimensions(params_grad, get_layer_params()) == true
            ensures
                - This function outputs the gradients of this layer with respect to the
                  input data from sub and also with respect to this layer's parameters.
                  These gradients are stored into #sub and #params_grad, respectively. To be
                  precise, the gradients are taken of a function f(sub,get_layer_params())
                  which is defined thusly:   
                    - let OUT be the output of forward(sub,OUT).
                    - let f(sub,get_layer_params()) == dot(OUT, gradient_input)
                  Then we define the following gradient vectors: 
                    - PARAMETER_GRADIENT == gradient of f(sub,get_layer_params()) with
                      respect to get_layer_params(). 
                    - for all valid I:
                        - DATA_GRADIENT_I == gradient of f(sub,get_layer_params()) with
                          respect to layer<I>(sub).get_output() (recall that forward() can
                          draw inputs from the immediate sub layer, sub.sub_net(), or
                          any earlier layer.  So you must consider the gradients with
                          respect to all inputs drawn from sub)
                  Finally, backward() adds these gradients into the output by performing:
                    - params_grad += PARAMETER_GRADIENT
                    - for all valid I:
                        - layer<I>(sub).get_gradient_input() += DATA_GRADIENT_I
        !*/

        const tensor& get_layer_params(
        ) const; 
        /*!
            ensures
                - returns the parameters that define the behavior of forward().
        !*/

        tensor& get_layer_params(
        ); 
        /*!
            ensures
                - returns the parameters that define the behavior of forward().
        !*/

    };

    // For each layer you define, always define an add_layer template so that layers can be
    // easily composed.  Moreover, the convention is that the layer class ends with an _
    // while the add_layer template has the same name but without the trailing _.
    template <typename SUB_NET>
    using EXAMPLE_LAYER = add_layer<EXAMPLE_LAYER_, SUB_NET>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class fc_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_LAYER_ interface defined above.
                In particular, it defines a fully connected layer that takes an input
                tensor and multiplies it by a weight matrix and outputs the results.
        !*/

    public:
        fc_(
        );
        /*!
            ensures
                - #get_num_outputs() == 1
        !*/

        explicit fc_(
            unsigned long num_outputs
        );
        /*!
            ensures
                - #get_num_outputs() == num_outputs
        !*/

        unsigned long get_num_outputs (
        ) const; 
        /*!
            ensures
                - This layer outputs column vectors that contain get_num_outputs()
                  elements. That is, the output tensor T from forward() will be such that:
                    - T.num_samples() == however many samples were given to forward().
                    - T.nr() == get_num_outputs()
                    - The rest of the dimensions of T will be 1.
        !*/

        template <typename SUB_NET> void setup (const SUB_NET& sub);
        template <typename SUB_NET> void forward(const SUB_NET& sub, resizable_tensor& output);
        template <typename SUB_NET> void backward(const tensor& gradient_input, SUB_NET& sub, tensor& params_grad);
        const tensor& get_layer_params() const; 
        tensor& get_layer_params(); 
        /*!
            These functions are implemented as described in the EXAMPLE_LAYER_ interface.
        !*/
    };


    template <typename SUB_NET>
    using fc = add_layer<fc_, SUB_NET>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
    public:

        relu_(
        );

        template <typename SUB_NET> void setup (const SUB_NET& sub);
        template <typename SUB_NET> void forward(const SUB_NET& sub, resizable_tensor& output);
        template <typename SUB_NET> void backward(const tensor& gradient_input, SUB_NET& sub, tensor& params_grad);
        const tensor& get_layer_params() const; 
        tensor& get_layer_params(); 
        /*!
            These functions are implemented as described in the EXAMPLE_LAYER_ interface.
        !*/
    };


    template <typename SUB_NET>
    using relu = add_layer<relu_, SUB_NET>;

// ----------------------------------------------------------------------------------------

}

#endif // #define DLIB_DNn_LAYERS_H_

