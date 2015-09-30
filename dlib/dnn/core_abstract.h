// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_CORE_ABSTRACT_H_
#ifdef DLIB_DNn_CORE_ABSTRACT_H_

#include "tensor_abstract.h"
#include "solvers_abstract.h"
#include <memory>
#include <type_traits>
#include "../rand.h"


namespace dlib
{

// ----------------------------------------------------------------------------------------

    void randomize_parameters (
        tensor& params,
        unsigned long num_inputs_and_outputs,
        dlib::rand& rnd
    );
    /*!
        ensures
            - This function assigns random values into params based on the given random
              number generator.  In particular, it uses the parameter initialization method
              of formula 16 from the paper "Understanding the difficulty of training deep
              feedforward neural networks" by Xavier Glorot and Yoshua Bengio.
            - It is assumed that the total number of inputs and outputs from the layer is
              num_inputs_and_outputs.  That is, you should set num_inputs_and_outputs to
              the sum of the dimensionalities of the vectors going into and out of the
              layer that uses params as its parameters.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T, 
        size_t N
        >
    class sstack
    {
        /*!
            REQUIREMENTS ON T
                - T is default and copy constructable.

            REQUIREMENTS ON N
                - N > 0

            WHAT THIS OBJECT REPRESENTS
                This is a basic stack of T objects.  It holds N of the objects and is
                entirely allocated on the stack rather than on the heap.
        !*/

    public:
        typedef T value_type;
        const static size_t num_elements = N;

        sstack(
        );
        /*!
            ensures
                - #size() == N
                - All elements of this stack are default constructed.
        !*/

        sstack(
            const T& item
        );
        /*!
            ensures
                - #size() == N
                - Initializes all N elements in this stack with the given item.  E.g.
                  top()==item, pop().top()==item, pop().pop().top()==item, etc.
        !*/

        const T& top(
        ) const;
        /*!
            ensures
                - returns the top element of the stack.
        !*/

        T& top(
        );
        /*!
            ensures
                - returns the top element of the stack.  
        !*/


        size_t size(
        ) const;
        /*!
            ensures
                - returns the number of elements in this stack.  In particular, the number
                  returned is always N.
        !*/

        const sstack<T,N-1>& pop(
        ) const;
        /*!
            requires
                - size() > 1
            ensures
                - returns a reference to the sub-stack S such that:
                    - S.size() == size()-1.
                    - S.top() is the next element in the stack.
        !*/

        sstack<T,N-1>& pop(
        ); 
        /*!
            requires
                - size() > 1
            ensures
                - returns a reference to the sub-stack S such that:
                    - S.size() == size()-1.
                    - S.top() is the next element in the stack.
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        typename LAYER_DETAILS, 
        typename SUBNET
        >
    class add_layer
    {
        /*!
            REQUIREMENTS ON LAYER_DETAILS
                - Must be a type that implements the EXAMPLE_LAYER_ interface defined in
                  layers_abstract.h

            REQUIREMENTS ON SUBNET
                - One of the following must be true:
                    - SUBNET implements the EXAMPLE_INPUT_LAYER interface defined in
                      input_abstract.h.
                    - SUBNET is an add_layer object.
                    - SUBNET is an add_tag_layer object.
                    - SUBNET is an add_skip_layer object.

            WHAT THIS OBJECT REPRESENTS
                This object represents a deep neural network.  In particular, it is a tool
                for adding another layer on top of the neural network of type SUBNET, which
                is specified as a template argument.  The specific layer added is defined
                by the LAYER_DETAILS details template argument.
        !*/

    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;
        // If SUBNET is an input layer then num_layers == 1, otherwise it has the
        // definition shown here:
        const static size_t num_layers = subnet_type::num_layers + 1;

        add_layer(
        );
        /*!
            ensures
                - default constructs all the layers in this network.
        !*/

        add_layer(const add_layer&) = default;
        add_layer(add_layer&&) = default;
        add_layer& operator=(add_layer&&) = default;
        add_layer& operator=(const add_layer&) = default;
        /*!
            ensures
                - this object is copyable and movable.
        !*/

        template <typename T, typename U>
        add_layer(
            const add_layer<T,U>& item
        );
        /*!
            ensures
                - This constructor allows you to copy neural network objects from one to
                  another as long as their corresponding layers can be constructed from
                  each other.
                - #layer_details() == layer_details_type(item.layer_details())
                - #subnet()        == subnet_type(item.subnet())
        !*/

        template <typename ...T>
        add_layer(
            const layer_details_type& layer_det, 
            T&& ...args
        );
        /*!
            ensures
                - #layer_details() == layer_details_type(layer_det)
                - #subnet()        == subnet_type(args)
        !*/

        template <typename ...T>
        add_layer(
            layer_details_type&& layer_det, 
            T&& ...args
        );
        /*!
            ensures
                - #layer_details() == layer_details_type(layer_det)
                - #subnet()        == subnet_type(args)
        !*/

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const;
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
            ensures
                - Converts the iterator range into a tensor and stores it into #data.
                - #data.num_samples() == distance(ibegin,iend)*sample_expansion_factor. 
                - The data in the ith sample of #data corresponds to the input_type object
                  *(ibegin+i/sample_expansion_factor).
                - Invokes data.async_copy_to_device() so that the data begins transferring
                  to the GPU device, if present.
                - This function is implemented by calling the to_tensor() routine defined
                  at the input layer of this network.  
        !*/

        const subnet_type& subnet(
        ) const; 
        /*!
            ensures
                - returns the immediate subnetwork of *this network.  
        !*/

        subnet_type& subnet(
        );
        /*!
            ensures
                - returns the immediate subnetwork of *this network.  
        !*/

        const layer_details_type& layer_details(
        ) const; 
        /*!
            ensures
                - returns the layer_details_type instance that defines the behavior of the
                  layer at the top of this network.  I.e. returns the layer details that
                  defines the behavior of the layer nearest to the network output rather
                  than the input layer.
        !*/

        layer_details_type& layer_details(
        );
        /*!
            ensures
                - returns the layer_details_type instance that defines the behavior of the
                  layer at the top of this network.  I.e. returns the layer details that
                  defines the behavior of the layer nearest to the network output rather
                  than the input layer.
        !*/

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        );
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
            ensures
                - runs [ibegin,iend) through the network and returns the results.
                  In particular, this function performs:
                    to_tensor(ibegin,iend,temp_tensor);
                    return forward(temp_tensor);
                - The return value from this function is also available in #get_output().
                  i.e. this function returns #get_output().
                - #get_output().num_samples() == std::distance(ibegin,iend)*sample_expansion_factor.
                - have_same_dimensions(#get_gradient_input(), #get_output()) == true.
                - All elements of #get_gradient_input() are set to 0. 
                  i.e. calling this function clears out #get_gradient_input() and ensures
                  it has the same dimensions as the most recent output.
        !*/

        const tensor& operator() (
            const input_type& x
        );
        /*!
            ensures
                - runs a single x through the network and returns the output.
                  I.e. returns (*this)(&x, &x+1);
        !*/

        const tensor& forward(
            const tensor& x
        );
        /*!
            requires
                - x.num_samples()%sample_expansion_factor == 0
                - x.num_samples() > 0
            ensures
                - Runs x through the network and returns the results.  In particular, this
                  function performs the equivalent of:
                    subnet().forward(x);
                    if (this is the first time forward() has been called) then
                        layer_details().setup(subnet());
                    layer_details().forward(subnet(), get_output());
                - The return value from this function is also available in #get_output().
                  i.e. this function returns #get_output().
                - #get_output().num_samples() == x.num_samples().
                - have_same_dimensions(#get_gradient_input(), #get_output()) == true
                - All elements of #get_gradient_input() are set to 0. 
                  i.e. calling this function clears out #get_gradient_input() and ensures
                  it has the same dimensions as the most recent output.
        !*/

        const tensor& get_output(
        ) const;
        /*!
            ensures
                - returns the output for the last tensor that was run through the network.
                  If nothing has been run through the network yet then returns an empty
                  tensor. 
        !*/

        tensor& get_gradient_input(
        );
        /*!
            ensures
                - returns the error gradient for this network.  That is, this is the error
                  gradient that this network will use to update itself when update() is
                  called.  Therefore, when performing back propagation, layers that sit on
                  top of this network layer write their back propagated error gradients
                  into get_gradient_input().  Or to put it another way, during back
                  propagation, layers take the contents of their get_gradient_input() and
                  back propagate it through themselves and store the results into their
                  subnetwork's get_gradient_input().

                  This means you should consider get_gradient_input() as an input to the
                  update() method.  
        !*/

        template <typename solver_type>
        void update(
            const tensor& x, 
            sstack<solver_type,num_layers>& solvers
        );
        /*!
            requires
                - forward(x) was called to forward propagate x though the network.
                - x.num_samples() == get_output().num_samples()
                - get_gradient_input() has been set equal to the gradient of this network's
                  output with respect to some loss function.
                - This instance of solvers has only ever been used with this network.  That
                  is, if you want to call update() on some other neural network object then
                  you must not reuse the same solvers object.
            ensures
                - Back propagates the error gradient, get_gradient_input(), through this
                  network and uses the provided solvers to update the network parameters.
        !*/

        void clean(
        );
        /*!
            ensures
                - Causes the network to forget about everything but its parameters.  
                  That is, for each layer we will have:
                    - get_output().num_samples() == 0
                    - get_gradient_input().num_samples() == 0
                  However, running new input data though this network will still have the
                  same output it would have had regardless of any calls to clean().
                  The purpose of clean() is to compact the network object prior to saving
                  it to disk so that it takes up less space and the IO is quicker.
        !*/

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class no_label_type;

    template <
        typename LOSS_DETAILS, 
        typename SUBNET
        >
    class add_loss_layer
    {
        /*!
            REQUIREMENTS ON LOSS_DETAILS 
                - Must be a type that implements the EXAMPLE_LOSS_LAYER_ interface defined
                  in loss_abstract.h
                - LOSS_DETAILS::sample_expansion_factor == SUBNET::sample_expansion_factor
                  i.e. The loss layer and input layer must agree on the sample_expansion_factor.

            REQUIREMENTS ON SUBNET
                - One of the following must be true:
                    - SUBNET is an add_layer object.
                    - SUBNET is an add_tag_layer object.
                    - SUBNET is an add_skip_layer object.

            WHAT THIS OBJECT REPRESENTS
                This object represents a deep neural network.  In particular, it is a tool
                for adding a loss layer on top of the neural network of type SUBNET, which
                is specified as a template argument.  The specific layer added is defined
                by the LOSS_DETAILS details template argument.  Importantly, a loss layer
                is the last layer in a deep neural network.  So once it is added you can't
                add any other layers of any type.
        !*/

    public:
        typedef LOSS_DETAILS loss_details_type;
        typedef SUBNET subnet_type;
        typedef typename subnet_type::input_type input_type;
        // Note that the loss layer doesn't count as an additional layer since it doesn't
        // have any learnable parameters.  
        const static size_t num_layers = subnet_type::num_layers;
        const static unsigned int sample_expansion_factor = subnet_type::sample_expansion_factor;
        // If LOSS_DETAILS is an unsupervised loss then label_type==no_label_type.
        // Otherwise it is defined as follows:
        typedef typename LOSS_DETAILS::label_type label_type;



        add_loss_layer() = default;
        /*!
            ensures
                - default constructs all the layers in this network.
        !*/

        add_loss_layer(const add_loss_layer&) = default;
        add_loss_layer(add_loss_layer&&) = default;
        add_loss_layer& operator=(add_loss_layer&&) = default;
        add_loss_layer& operator=(const add_loss_layer&) = default;
        /*!
            ensures
                - this object is copyable and movable.
        !*/

        template <typename T, typename U>
        add_loss_layer(
            const add_loss_layer<T,U>& item
        );
        /*!
            ensures
                - This constructor allows you to copy neural network objects from one to
                  another as long as their corresponding layers can be constructed from
                  each other.
                - #loss_details() == loss_details_type(item.loss_details())
                - #subnet()       == subnet_type(item.subnet())
        !*/

        template <typename ...T>
        add_loss_layer(
            const LOSS_DETAILS& layer_det, 
            T&& ...args
        ); 
        /*!
            ensures
                - #loss_details() == loss_details_type(layer_det)
                - #subnet()       == subnet_type(args)
        !*/

        template <typename ...T>
        add_loss_layer(
            LOSS_DETAILS&& layer_det, 
            T&& ...args
        );
        /*!
            ensures
                - #loss_details() == loss_details_type(layer_det)
                - #subnet()       == subnet_type(args)
        !*/

        template <typename ...T>
        add_loss_layer(
            T ...args
        ); 
        /*!
            ensures
                - #loss_details() == loss_details_type()
                - #subnet()       == subnet_type(args)
        !*/

        const subnet_type& subnet(
        ) const; 
        /*!
            ensures
                - returns the immediate subnetwork of *this network.  
        !*/

        subnet_type& subnet(
        ); 
        /*!
            ensures
                - returns the immediate subnetwork of *this network.  
        !*/

        const loss_details_type& loss_details(
        ) const; 
        /*!
            ensures
                - returns the loss_details_type instance that defines the behavior of the
                  loss layer used by this network.
        !*/

        loss_details_type& loss_details(
        ); 
        /*!
            ensures
                - returns the loss_details_type instance that defines the behavior of the
                  loss layer used by this network.
        !*/

        template <typename input_iterator>
        void to_tensor (
            input_iterator ibegin,
            input_iterator iend,
            resizable_tensor& data
        ) const;
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
            ensures
                - Converts the iterator range into a tensor and stores it into #data.
                - #data.num_samples() == distance(ibegin,iend)*sample_expansion_factor. 
                - The data in the ith sample of #data corresponds to the input_type object
                  *(ibegin+i/sample_expansion_factor).
                - Invokes data.async_copy_to_device() so that the data begins transferring
                  to the GPU device, if present.
                - This function is implemented by calling the to_tensor() routine defined
                  at the input layer of this network.  
        !*/

    // -------------

        template <typename output_iterator>
        void operator() (
            const tensor& x, 
            output_iterator obegin
        );
        /*!
            requires
                - x.num_samples()%sample_expansion_factor == 0
                - x.num_samples() > 0
                - obegin == iterator pointing to the start of a range of
                  x.num_samples()/sample_expansion_factor label_type elements.
            ensures
                - runs x through the network and writes the output to the range at obegin.
                - loss_details().to_label() is used to write the network output into
                  obegin.
        !*/

        template <typename input_iterator, typename label_iterator>
        void operator() (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator obegin
        );
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
                - obegin == iterator pointing to the start of a range of
                  std::distance(ibegin,iend) label_type elements.
            ensures
                - runs [ibegin,iend) through the network and writes the output to the range
                  at obegin.
                - loss_details().to_label() is used to write the network output into
                  obegin.
        !*/

    // -------------

        const label_type& operator() (
            const input_type& x
        );
        /*!
            ensures
                - runs a single object, x, through the network and returns the output.
                - loss_details().to_label() is used to convert the network output into a
                  label_type.
        !*/

    // -------------

        template <typename label_iterator>
        double compute_loss (
            const tensor& x,
            label_iterator lbegin 
        );
        /*!
            requires
                - x.num_samples()%sample_expansion_factor == 0
                - x.num_samples() > 0
                - lbegin == iterator pointing to the start of a range of
                  x.num_samples()/sample_expansion_factor label_type elements.
            ensures
                - runs x through the network, compares the output to the expected output
                  pointed to by lbegin, and returns the resulting loss. 
                - for all valid k:
                    - the expected label of the kth sample in x is *(lbegin+k/sample_expansion_factor).
                - This function does not update the network parameters.
        !*/

        template <typename input_iterator, typename label_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin 
        );
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
                - lbegin == iterator pointing to the start of a range of
                  std::distance(ibegin,iend) label_type elements.
            ensures
                - runs [ibegin,iend) through the network, compares the output to the
                  expected output pointed to by lbegin, and returns the resulting loss. 
                - for all valid k:
                    - the expected label of *(ibegin+k) is *(lbegin+k).
                - This function does not update the network parameters.
        !*/

    // -------------

        double compute_loss (
            const tensor& x
        );
        /*!
            requires
                - LOSS_DETAILS is an unsupervised loss.  i.e. label_type==no_label_type.
                - x.num_samples()%sample_expansion_factor == 0
                - x.num_samples() > 0
            ensures
                - runs x through the network and returns the resulting loss. 
                - This function does not update the network parameters.
        !*/

        template <typename input_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend,
        );
        /*!
            requires
                - LOSS_DETAILS is an unsupervised loss.  i.e. label_type==no_label_type.
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
            ensures
                - runs [ibegin,iend) through the network and returns the resulting loss. 
                - This function does not update the network parameters.
        !*/

    // -------------

        template <typename label_iterator, typename solver_type>
        double update (
            const tensor& x,
            label_iterator lbegin,
            sstack<solver_type,num_layers>& solvers
        );
        /*!
            requires
                - x.num_samples()%sample_expansion_factor == 0
                - x.num_samples() > 0
                - lbegin == iterator pointing to the start of a range of
                  x.num_samples()/sample_expansion_factor label_type elements.
                - This instance of solvers has only ever been used with this network.  That
                  is, if you want to call update() on some other neural network object then
                  you must not reuse the same solvers object.
            ensures
                - runs x through the network, compares the output to the expected output
                  pointed to by lbegin, and updates the network parameters via
                  backpropagation.
                - for all valid k:
                    - the expected label of the kth sample in x is *(lbegin+k/sample_expansion_factor).
                - The provided solvers are used to update the parameters in each layer of
                  the network.
                - returns compute_loss(x,lbegin)
        !*/

        template <typename input_iterator, typename label_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin,
            sstack<solver_type,num_layers>& solvers
        );
        /*!
            requires
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
                - lbegin == iterator pointing to the start of a range of
                  std::distance(ibegin,iend) label_type elements.
                - This instance of solvers has only ever been used with this network.  That
                  is, if you want to call update() on some other neural network object then
                  you must not reuse the same solvers object.
            ensures
                - runs [ibegin,iend) through the network, compares the output to the
                  expected output pointed to by lbegin, and updates the network parameters
                  via backpropagation.
                - for all valid k:
                    - the expected label of *(ibegin+k) is *(lbegin+k).
                - The provided solvers are used to update the parameters in each layer of
                  the network.
                - returns compute_loss(ibegin,iend,lbegin)
        !*/

    // -------------

        template <typename solver_type>
        double update (
            const tensor& x,
            sstack<solver_type,num_layers>& solvers
        );
        /*!
            requires
                - LOSS_DETAILS is an unsupervised loss.  i.e. label_type==no_label_type.
                - x.num_samples()%sample_expansion_factor == 0
                - x.num_samples() > 0
                - This instance of solvers has only ever been used with this network.  That
                  is, if you want to call update() on some other neural network object then
                  you must not reuse the same solvers object.
            ensures
                - runs x through the network and updates the network parameters by
                  back-propagating the loss gradient through the network.
                - The provided solvers are used to update the parameters in each layer of
                  the network.
                - returns compute_loss(x)
        !*/

        template <typename input_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            sstack<solver_type,num_layers>& solvers
        );
        /*!
            requires
                - LOSS_DETAILS is an unsupervised loss.  i.e. label_type==no_label_type.
                - [ibegin, iend) is an iterator range over input_type objects.
                - std::distance(ibegin,iend) > 0
                - This instance of solvers has only ever been used with this network.  That
                  is, if you want to call update() on some other neural network object then
                  you must not reuse the same solvers object.
            ensures
                - runs [ibegin,iend) through the network and updates the network parameters
                  by back-propagating the loss gradient through the network.
                - The provided solvers are used to update the parameters in each layer of
                  the network.
                - returns compute_loss(ibegin,iend)
        !*/

    // -------------

        void clean (
        );
        /*!
            ensures
                - Causes the network to forget about everything but its parameters.  
                - invokes subnet().clean()
        !*/
    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        unsigned long ID, 
        typename SUBNET
        >
    class add_tag_layer
    {
        /*!
            REQUIREMENTS ON SUBNET
                - One of the following must be true:
                    - SUBNET implements the EXAMPLE_INPUT_LAYER interface defined in
                      input_abstract.h.
                    - SUBNET is an add_layer object.
                    - SUBNET is an add_tag_layer object.
                    - SUBNET is an add_skip_layer object.

            WHAT THIS OBJECT REPRESENTS
                This object adds a new layer to a deep neural network.  However, this layer
                simply performs the identity transform.  This means it is a no-op and its
                presence does not change the behavior of the network.  It exists solely to
                be used by add_skip_layer to reference a particular part of a network.

                Also, this object provides an interface identical to the one defined by the
                add_layer object.
        !*/
    };

    template <typename SUBNET> using tag1  = add_tag_layer< 1, SUBNET>;
    template <typename SUBNET> using tag2  = add_tag_layer< 2, SUBNET>;
    template <typename SUBNET> using tag3  = add_tag_layer< 3, SUBNET>;
    template <typename SUBNET> using tag4  = add_tag_layer< 4, SUBNET>;
    template <typename SUBNET> using tag5  = add_tag_layer< 5, SUBNET>;
    template <typename SUBNET> using tag6  = add_tag_layer< 6, SUBNET>;
    template <typename SUBNET> using tag7  = add_tag_layer< 7, SUBNET>;
    template <typename SUBNET> using tag8  = add_tag_layer< 8, SUBNET>;
    template <typename SUBNET> using tag9  = add_tag_layer< 9, SUBNET>;
    template <typename SUBNET> using tag10 = add_tag_layer<10, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class TAG_TYPE, 
        typename SUBNET
        >
    class add_skip_layer
    {
        /*!
            REQUIREMENTS ON SUBNET
                - One of the following must be true:
                    - SUBNET is an add_layer object.
                    - SUBNET is an add_tag_layer object.
                    - SUBNET is an add_skip_layer object.

            WHAT THIS OBJECT REPRESENTS
                This object adds a new layer to a deep neural network which draws its
                inputs from layer<TAG_TYPE>(subnet()) and performs the identity transform.

                Also, this object provides an interface identical to the one defined by the
                add_layer object.
        !*/
    };

    template <typename SUBNET> using skip1  = add_skip_layer< tag1, SUBNET>;
    template <typename SUBNET> using skip2  = add_skip_layer< tag2, SUBNET>;
    template <typename SUBNET> using skip3  = add_skip_layer< tag3, SUBNET>;
    template <typename SUBNET> using skip4  = add_skip_layer< tag4, SUBNET>;
    template <typename SUBNET> using skip5  = add_skip_layer< tag5, SUBNET>;
    template <typename SUBNET> using skip6  = add_skip_layer< tag6, SUBNET>;
    template <typename SUBNET> using skip7  = add_skip_layer< tag7, SUBNET>;
    template <typename SUBNET> using skip8  = add_skip_layer< tag8, SUBNET>;
    template <typename SUBNET> using skip9  = add_skip_layer< tag9, SUBNET>;
    template <typename SUBNET> using skip10 = add_skip_layer<tag10, SUBNET>;

// ----------------------------------------------------------------------------------------

    template <
        unsigned int i, 
        typename net_type
        >
    auto& layer (
        net_type& n
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or add_tag_layer.
        ensures
            - This function chains together i calls to n.subnet() and returns the
              result.  So for example:
                - if (i == 0)
                    - returns n
                - else if (i == 1)
                    - returns n.subnet()
                - else if (i == 2)
                    - returns n.subnet().subnet()
                - else if (i == 3)
                    - returns n.subnet().subnet().subnet()
                - else
                    - etc.
    !*/

    template <
        template<typename> class Match, 
        typename net_type 
        >
    auto& layer (
        net_type& n
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or add_tag_layer.
        ensures
            - returns the first layer in n that is of type Match.  E.g. if net_type is
              fc<relu<fc<input<sample_type>>>> then calling layer<relu>(n) would return
              layer<1>(n), that is, a reference to the relu layer.
    !*/

    template <
        template<typename> class Match, 
        unsigned int i, 
        typename net_type
        >
    auto& layer (
        net_type& n
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or add_tag_layer.
        ensures
            - returns layer<i>(layer<Match>(n))
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename layer_details_type
        >
    void test_layer (
        layer_details_type l
    );
    /*!
        ensures
            - Checks if l correctly implements the EXAMPLE_LAYER_ interface defined in
              layers_abstract.h.  Importantly, it computes numerical approximations to the
              gradients and compares them to the outputs of the layer.  
    !*/

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename net_type, 
        typename solver_type = sgd
        >
    class dnn_trainer
    {
        /*!
            REQUIREMENTS ON net_type
                - net_type is an add_loss_layer object.

            REQUIREMENTS ON solver_type
                - solver_type is an implementation of the EXAMPLE_SOLVER interface defined
                  in solvers_abstract.h

            WHAT THIS OBJECT REPRESENTS

        !*/

    public:

        typedef typename net_type::label_type label_type;
        typedef typename net_type::input_type input_type;

        dnn_trainer(
        );

        explicit dnn_trainer(
            const net_type& net
        );

        dnn_trainer(
            const net_type& net, 
            const solver_type& solver
        ); 

        const net_type& get_net (
        ) const; 

        void set_net (
            const net_type& net
        ); 

        void set_solver (
            const solver_type& solver_
        );

        const sstack<solver_type,net_type::num_layers>& get_solvers (
        ) const; 

        sstack<solver_type,net_type::num_layers>& get_solvers (
        ); 

        const net_type& train (
            const std::vector<input_type>& data,
            const std::vector<label_type>& labels 
        ); 
        /*!
            requires
                - data.size() == labels.size()
                - TODO: the net has a supervised loss layer.
        !*/

        const net_type& train (
            const std::vector<input_type>& data
        );
        /*!
            requires 
                - TODO: the net has an unsupervised loss layer.
            ensures
                - trains an auto-encoder
        !*/

    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_CORE_ABSTRACT_H_ DLIB_DNn_CORE_H_

