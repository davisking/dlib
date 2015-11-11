// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_LAYERS_ABSTRACT_H_
#ifdef DLIB_DNn_LAYERS_ABSTRACT_H_

#include "tensor_abstract.h"
#include "core_abstract.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    class SUBNET 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object represents a deep neural network.  In particular, it is
                the simplified interface through which layer objects interact with their
                subnetworks.  A layer's two important tasks are to (1) take outputs from its
                subnetwork and forward propagate them though itself and (2) to backwards
                propagate an error gradient through itself and onto its subnetwork.
                The idea of a subnetwork is illustrated in the following diagram:

                  +---------------------------------------------------------+
                  | loss <-- layer1 <-- layer2 <-- ... <-- layern <-- input |
                  +---------------------------------------------------------+
                                      ^                            ^
                                      \__ subnetwork for layer1 __/

                Therefore, by "subnetwork" we mean the part of the network closer to the
                input.  
        !*/

    public:
        // You aren't allowed to copy subnetworks from inside a layer.
        SUBNET(const SUBNET&) = delete;
        SUBNET& operator=(const SUBNET&) = delete;

        const tensor& get_output(
        ) const;
        /*!
            ensures
                - returns the output of this subnetwork.  This is the data that the next
                  layer in the network will take as input.
                - have_same_dimensions(#get_gradient_input(), get_output()) == true
        !*/

        tensor& get_gradient_input(
        );
        /*!
            ensures
                - returns the error gradient for this subnetwork.  That is, this is the
                  error gradient that this network will use to update itself.  Therefore,
                  when performing back propagation, layers that sit on top of this
                  subnetwork write their back propagated error gradients into
                  get_gradient_input().  Or to put it another way, during back propagation,
                  layers take the contents of their get_gradient_input() and back propagate
                  it through themselves and store the results into their subnetwork's
                  get_gradient_input().
        !*/

        const NEXT_SUBNET& subnet(
        ) const;
        /*!
            ensures
                - returns the subnetwork of *this network.  With respect to the diagram
                  above, if *this was layer1 then subnet() would return the network that
                  begins with layer2.
        !*/

        NEXT_SUBNET& subnet(
        );
        /*!
            ensures
                - returns the subnetwork of *this network.  With respect to the diagram
                  above, if *this was layer1 then subnet() would return the network that
                  begins with layer2.
        !*/
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
                different functions to accommodate the task you are trying to accomplish.
                Dlib includes a number of common layer types but if you want to define your
                own then you simply implement a class with the same interface as
                EXAMPLE_LAYER_.

                Note that there is no dlib::EXAMPLE_LAYER_ type.  It is shown here purely
                to document the interface that a layer object must implement.
        !*/

    public:

        EXAMPLE_LAYER_(
        );
        /*!
            ensures
                - Default constructs this object.  This function is not required to do
                  anything in particular but it must exist, that is, it is required that
                  layer objects be default constructable. 
        !*/

        EXAMPLE_LAYER_ (
            const EXAMPLE_LAYER_& item
        );
        /*!
            ensures
                - EXAMPLE_LAYER_ objects are copy constructable
        !*/

        EXAMPLE_LAYER_(
            const some_other_layer_type& item
        );
        /*!
            ensures
                - Constructs this object from item.  This form of constructor is optional
                  but it allows you to provide a conversion from one layer type to another.
                  For example, the following code is valid only if my_layer2 can be
                  constructed from my_layer1:
                    relu<fc<my_layer1<fc<input<matrix<float>>>>>> my_dnn1;
                    relu<fc<my_layer2<fc<input<matrix<float>>>>>> my_dnn2(my_dnn1);
                  This kind of pattern is useful if you want to use one type of layer
                  during training but a different type of layer during testing since it
                  allows you to easily convert between related deep neural network types.  
        !*/

        template <typename SUBNET>
        void setup (
            const SUBNET& sub
        );
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of this file.
            ensures
                - performs any necessary initial memory allocations and/or sets parameters
                  to their initial values prior to learning.  Therefore, calling setup
                  destroys any previously learned parameters.  Also, typically setup()
                  would look at the dimensions of the outputs of sub and configure the
                  number of parameters in *this accordingly.
        !*/

        template <typename SUBNET>
        void forward(
            const SUBNET& sub, 
            resizable_tensor& output
        );
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of this file.
                - setup() has been called.
            ensures
                - Runs the output of the subnetwork through this layer and stores the
                  output into #output.  In particular, forward() can use any of the outputs
                  in sub (e.g. sub.get_output(), sub.subnet().get_output(), etc.) to
                  compute whatever it wants.
        !*/

        template <typename SUBNET>
        void backward(
            const tensor& computed_output,
            const tensor& gradient_input, 
            SUBNET& sub, 
            tensor& params_grad
        );
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of this file.
                - setup() has been called.
                - computed_output is the tensor resulting from calling forward(sub,computed_output).
                  Moreover, this was the most recent call to forward().  This means that
                  backward() is allowed to cache intermediate results computed during
                  forward() and use them for the backward computation.
                - have_same_dimensions(gradient_input, computed_output)
                - have_same_dimensions(sub.get_gradient_input(), sub.get_output()) == true
                - have_same_dimensions(params_grad, get_layer_params()) == true
            ensures
                - This function outputs the gradients of this layer with respect to the
                  input data from sub and also with respect to this layer's parameters.
                  These gradients are stored into #sub and #params_grad, respectively. To be
                  precise, the gradients are taken of a function f(sub,get_layer_params())
                  which is defined thusly:   
                    - Recalling that computed_output is a function of sub and get_layer_params() 
                      since it is the result of calling forward(sub,computed_output):
                      let f(sub,get_layer_params()) == dot(computed_output, gradient_input)
                  Then we define the following gradient vectors: 
                    - PARAMETER_GRADIENT == gradient of f(sub,get_layer_params()) with
                      respect to get_layer_params(). 
                    - for all valid I:
                        - DATA_GRADIENT_I == gradient of f(sub,get_layer_params()) with
                          respect to layer<I>(sub).get_output() (recall that forward() can
                          draw inputs from the immediate sub layer, sub.subnet(), or
                          any earlier layer.  So you must consider the gradients with
                          respect to all inputs drawn from sub)
                  Finally, backward() outputs these gradients by performing:
                    - params_grad = PARAMETER_GRADIENT 
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

    void serialize(const EXAMPLE_LAYER_& item, std::ostream& out);
    void deserialize(EXAMPLE_LAYER_& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    // For each layer you define, always define an add_layer template so that layers can be
    // easily composed.  Moreover, the convention is that the layer class ends with an _
    // while the add_layer template has the same name but without the trailing _.
    template <typename SUBNET>
    using EXAMPLE_LAYER = add_layer<EXAMPLE_LAYER_, SUBNET>;

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
                    - T.k() == get_num_outputs()
                    - The rest of the dimensions of T will be 1.
        !*/

        template <typename SUBNET> void setup (const SUBNET& sub);
        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
        template <typename SUBNET> void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        const tensor& get_layer_params() const; 
        tensor& get_layer_params(); 
        /*!
            These functions are implemented as described in the EXAMPLE_LAYER_ interface.
        !*/
    };

    void serialize(const fc_& item, std::ostream& out);
    void deserialize(fc_& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    template <typename SUBNET>
    using fc = add_layer<fc_, SUBNET>;

// ----------------------------------------------------------------------------------------

    class relu_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is an implementation of the EXAMPLE_LAYER_ interface defined above.
                In particular, it defines a rectified linear layer.  Therefore, it passes
                its inputs though the function f(x)=max(x,0) where f() is applied pointwise
                across the input tensor.
                
        !*/

    public:

        relu_(
        );

        template <typename SUBNET> void setup (const SUBNET& sub);
        template <typename SUBNET> void forward(const SUBNET& sub, resizable_tensor& output);
        template <typename SUBNET> void backward(const tensor& computed_output, const tensor& gradient_input, SUBNET& sub, tensor& params_grad);
        const tensor& get_layer_params() const; 
        tensor& get_layer_params(); 
        /*!
            These functions are implemented as described in the EXAMPLE_LAYER_ interface.
        !*/
    };

    void serialize(const relu_& item, std::ostream& out);
    void deserialize(relu_& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    template <typename SUBNET>
    using relu = add_layer<relu_, SUBNET>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_LAYERS_ABSTRACT_H_

