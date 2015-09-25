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
                entirely allocated on the stack.
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
                    - Initializes all N elements in this stack with the given item.
                      E.g. top()==item, pop().top()==item, pop().pop().top()==item, etc.
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
                    - returns the number of elements in this stack.  In particular, the
                      number returned is always N.
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
        typename SUB_NET
        >
    class add_layer
    {
        /*!
            REQUIREMENTS ON LAYER_DETAILS
                - Must be a type that implements the EXAMPLE_LAYER_ interface defined in
                  layers_abstract.h

            REQUIREMENTS ON SUB_NET
                - One of the following must be true:
                    - SUB_NET implements the EXAMPLE_INPUT_LAYER interface defined in
                      input_abstract.h.
                    - SUB_NET is an add_layer object.
                    - SUB_NET is an add_tag_layer object.
                    - SUB_NET is an add_skip_layer object.

            WHAT THIS OBJECT REPRESENTS
                Stacks a new layer, defined by LAYER_DETAILS, on top of SUB_NET type.
        !*/

    public:
        typedef LAYER_DETAILS layer_details_type;
        typedef SUB_NET sub_net_type;
        typedef typename sub_net_type::input_type input_type;
        const static unsigned int sample_expansion_factor = sub_net_type::sample_expansion_factor;
        // If SUB_NET is an input layer then num_layers == 1, otherwise it has the
        // definition shown here:
        const static size_t num_layers = sub_net_type::num_layers + 1;

        add_layer(
        );

        add_layer(const add_layer&) = default;
        add_layer(add_layer&&) = default;
        add_layer& operator=(add_layer&&) = default;
        add_layer& operator=(const add_layer&) = default;


        // Allow copying networks from one to another as long as their corresponding 
        // layers can be constructed from each other.
        template <typename T, typename U>
        add_layer(
            const add_layer<T,U>& item
        );
        /*!
            ensures
                - #layer_details() == layer_details_type(item.layer_details())
                - #sub_net()       == sub_net_type(item.sub_net())
        !*/

        template <typename ...T>
        add_layer(
            const LAYER_DETAILS& layer_det, 
            T&& ...args
        );
        /*!
            ensures
                - #layer_details() == layer_details_type(layer_det)
                - #sub_net() == sub_net_type(args)
        !*/

        template <typename ...T>
        add_layer(
            LAYER_DETAILS&& layer_det, 
            T&& ...args
        );
        /*!
            ensures
                - #layer_details() == layer_details_type(layer_det)
                - #sub_net() == sub_net_type(args)
        !*/

        template <typename input_iterator>
        void to_tensor (
            input_iterator begin,
            input_iterator end,
            resizable_tensor& data
        ) const;
        /*!
            requires
                - [begin, end) is an iterator range over input_type objects.
            ensures
                - Converts the iterator range into a tensor and stores it into #data.
                - #data.num_samples() == distance(begin,end)*sample_expansion_factor. 
                - Invokes data.async_copy_to_device() so that the data begins transferring
                  to the device.
                - Ultimately this function just calls sub_net().sub_net()...sub_net().to_tensor(begin,end,data).
        !*/

        template <typename input_iterator>
        const tensor& operator() (
            input_iterator ibegin,
            input_iterator iend
        );
        /*!
            ensures
                - runs [ibegin,iend) through the network and returns the results.
                  In particular, this function performs:
                    to_tensor(ibegin,iend,temp_tensor);
                    return forward(temp_tensor);
                - The return value from this function is also available in #get_output().
                - have_same_dimensions(#get_gradient_input(), #get_output()) == true
                - All elements of #get_gradient_input() are set to 0. 
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
            ensures
                - Runs x through the network and returns the results.  In particular, this
                  function performs the equivalent of:
                    sub_net().forward(x);
                    if (this is the first time forward() has been called) then
                        layer_details().setup(sub_net());
                    layer_details().forward(sub_net(), get_output());
                - The return value from this function is also available in #get_output().
                - have_same_dimensions(#get_gradient_input(), #get_output()) == true
                - All elements of #get_gradient_input() are set to 0. 
        !*/
        {
            sub_network.forward(x);
            const dimpl::sub_net_wrapper<sub_net_type> wsub(sub_network);
            if (!this_layer_setup_called)
            {
                details.setup(wsub);
                this_layer_setup_called = true;
            }
            details.forward(wsub, cached_output);
            gradient_input_is_stale = true;
            return get_output();
        }

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
                - 
        !*/
        { 
            if (gradient_input_is_stale)
            {
                gradient_input_is_stale = false;
                x_grad.copy_size(get_output());
                x_grad = 0;
            }
            return x_grad; 
        }

        template <typename solver_type>
        void update(const tensor& x, sstack<solver_type,num_layers>& solvers)
        /*!
            requires
                - forward(x) was called to forward propagate x though the network.
                - x.num_samples() == get_gradient_input().num_samples()
                - get_gradient_input() == the gradient of the network with respect
                  to some loss.
        !*/
        {
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub_network);
            params_grad.copy_size(details.get_layer_params());
            params_grad = 0;
            details.backward(get_gradient_input(), wsub, static_cast<tensor&>(params_grad));
            // Don't try to adjust the parameters if this layer doesn't have any.
            if (params_grad.size() != 0)
                solvers.top()(details, static_cast<const tensor&>(params_grad));
            sub_network.update(x, solvers.pop());
        }

        const sub_net_type& sub_net() const { return sub_network; }
        sub_net_type& sub_net() { return sub_network; }

        const layer_details_type& layer_details() const { return details; } 
        layer_details_type& layer_details() { return details; } 

        void clean(
        );

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class no_label_type;

    template <
        typename LOSS_DETAILS, 
        typename SUB_NET
        >
    class add_loss_layer
    {
        /*!
            REQUIREMENTS ON LOSS_DETAILS 
                - Must be a type that implements the EXAMPLE_LAYER_ interface defined in
                  layers_abstract.h
                - LOSS_DETAILS::sample_expansion_factor == SUB_NET::sample_expansion_factor
                  i.e. The loss layer and input layer must agree on the sample_expansion_factor.

            REQUIREMENTS ON SUB_NET
                - One of the following must be true:
                    - SUB_NET is an add_layer object.
                    - SUB_NET is an add_tag_layer object.
                    - SUB_NET is an add_skip_layer object.

            WHAT THIS OBJECT REPRESENTS
                - Adds a loss layer, defined by LOSS_DETAILS, on top of SUB_NET.
        !*/

    public:
        typedef LOSS_DETAILS loss_details_type;
        typedef SUB_NET sub_net_type;
        typedef typename sub_net_type::input_type input_type;
        // Note that the loss layer doesn't count as an additional layer. 
        const static size_t num_layers = sub_net_type::num_layers;
        const static unsigned int sample_expansion_factor = sub_net_type::sample_expansion_factor;
        // If LOSS_DETAILS is an unsupervised loss then label_type==no_label_type.
        // Otherwise it is defined as follows:
        typedef typename LOSS_DETAILS::label_type label_type;

        static_assert(sample_expansion_factor == LOSS_DETAILS::sample_expansion_factor,
            "The loss layer and input layer must agree on the sample_expansion_factor.");


        add_loss_layer() = default;
        add_loss_layer(const add_loss_layer&) = default;
        add_loss_layer(add_loss_layer&&) = default;
        add_loss_layer& operator=(add_loss_layer&&) = default;
        add_loss_layer& operator=(const add_loss_layer&) = default;

        template <typename T, typename U>
        add_loss_layer(
            const add_loss_layer<T,U>& item
        ) : 
            loss(item.loss_details()),
            sub(item.sub_net())
        {}

        template <typename ...T>
        add_loss_layer(
            const LOSS_DETAILS& layer_det, 
            T&& ...args
        ) : 
            loss(layer_det), 
            sub(std::forward<T>(args)...)
        {
        }

        template <typename ...T>
        add_loss_layer(
            LOSS_DETAILS&& layer_det, 
            T&& ...args
        ) : 
            loss(std::move(layer_det)), 
            sub(std::forward<T>(args)...)
        {
        }

        template <typename ...T>
        add_loss_layer(
            T ...args
        ) : 
            sub(std::move(args)...)
        {
        }

        template <typename input_iterator, typename output_iterator>
        void operator() (
            input_iterator ibegin,
            input_iterator iend,
            output_iterator obegin
        )
        /*!
            requires
                - obegin == iterator pointing to the start of a range of distance(ibegin,iend)
                  elements.
            ensures
                - runs [ibegin,iend) through the network and writes the output to the range at obegin.
        !*/
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            loss.to_label(sub, obegin);
        }


        const label_type& operator() (const input_type& x)
        /*!
            ensures
                - runs a single x through the network and returns the output.
        !*/
        {
            (*this)(&x, &x+1, &temp_label);
            return temp_label;
        }


        template <typename input_iterator, typename label_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin 
        )
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub);
            return loss.compute_loss(temp_tensor, lbegin, wsub);
        }

        template <typename input_iterator>
        double compute_loss (
            input_iterator ibegin,
            input_iterator iend,
        );

        template <typename input_iterator, typename label_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            label_iterator lbegin,
            sstack<solver_type,num_layers>& solvers
        )
        {
            sub.to_tensor(ibegin,iend,temp_tensor);
            sub.forward(temp_tensor);
            dimpl::sub_net_wrapper<sub_net_type> wsub(sub);
            double l = loss.compute_loss(temp_tensor, lbegin, wsub);
            sub.update(temp_tensor, solvers);
            return l;
        }

        template <typename input_iterator, typename solver_type>
        double update (
            input_iterator ibegin,
            input_iterator iend,
            sstack<solver_type,num_layers>& solvers
        );

        const sub_net_type& sub_net() const { return sub; }
        sub_net_type& sub_net() { return sub; }
        const loss_details_type& loss_details() const { return loss; }
        loss_details_type& loss_details() { return loss; }

        void clean (
        )
        /*!
            ensures
                - Causes the network to forget about everything but its parameters.  
                  That is, for each layer we will have:
                    - get_output().num_samples() == 0
                    - get_gradient_input().num_samples() == 0
                  However, running new input data though this network will still have the
                  same output it would have had regardless of any calls to clean().
                  Finally, the purpose of clean() is to compact the network object prior to
                  saving it to disk so that it takes up less space and the IO is quicker.
        !*/
        {
            temp_tensor.clear();
            sub.clear();
        }

    private:

        loss_details_type loss;
        sub_net_type sub;

        // These two objects don't logically contribute to the state of this object.  They
        // are here to prevent them from being reallocated over and over.
        label_type temp_label;
        resizable_tensor temp_tensor;
    };


    template <typename T, typename U>
    struct is_layer_type<add_loss_layer<T,U>> : std::true_type {};

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        unsigned long ID, 
        typename SUB_NET
        >
    class add_tag_layer
    {
        /*!
            REQUIREMENTS ON SUB_NET

            WHAT THIS OBJECT REPRESENTS
                This object draws its inputs from sub_net() and performs the identity
                transform.  This means it is a no-op and its presence does not change
                the behavior of the network.  It exists solely to be used by add_skip_layer
                to reference a particular part of a network.

        !*/
    };

    template <typename SUB_NET> using tag1  = add_tag_layer< 1, SUB_NET>;
    template <typename SUB_NET> using tag2  = add_tag_layer< 2, SUB_NET>;
    template <typename SUB_NET> using tag3  = add_tag_layer< 3, SUB_NET>;
    template <typename SUB_NET> using tag4  = add_tag_layer< 4, SUB_NET>;
    template <typename SUB_NET> using tag5  = add_tag_layer< 5, SUB_NET>;
    template <typename SUB_NET> using tag6  = add_tag_layer< 6, SUB_NET>;
    template <typename SUB_NET> using tag7  = add_tag_layer< 7, SUB_NET>;
    template <typename SUB_NET> using tag8  = add_tag_layer< 8, SUB_NET>;
    template <typename SUB_NET> using tag9  = add_tag_layer< 9, SUB_NET>;
    template <typename SUB_NET> using tag10 = add_tag_layer<10, SUB_NET>;

// ----------------------------------------------------------------------------------------

    template <
        template<typename> class TAG_TYPE, 
        typename SUB_NET
        >
    class add_skip_layer
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object draws its inputs from layer<TAG_TYPE>(sub_net())
                and performs the identity transform.
        !*/
    };

    template <typename SUB_NET> using skip1  = add_skip_layer< tag1, SUB_NET>;
    template <typename SUB_NET> using skip2  = add_skip_layer< tag2, SUB_NET>;
    template <typename SUB_NET> using skip3  = add_skip_layer< tag3, SUB_NET>;
    template <typename SUB_NET> using skip4  = add_skip_layer< tag4, SUB_NET>;
    template <typename SUB_NET> using skip5  = add_skip_layer< tag5, SUB_NET>;
    template <typename SUB_NET> using skip6  = add_skip_layer< tag6, SUB_NET>;
    template <typename SUB_NET> using skip7  = add_skip_layer< tag7, SUB_NET>;
    template <typename SUB_NET> using skip8  = add_skip_layer< tag8, SUB_NET>;
    template <typename SUB_NET> using skip9  = add_skip_layer< tag9, SUB_NET>;
    template <typename SUB_NET> using skip10 = add_skip_layer<tag10, SUB_NET>;

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
            - This function chains together i calls to n.sub_net() and returns the
              result.  So for example:
                - if (i == 0)
                    - returns n
                - else if (i == 1)
                    - returns n.sub_net()
                - else if (i == 2)
                    - returns n.sub_net().sub_net()
                - else if (i == 3)
                    - returns n.sub_net().sub_net().sub_net()
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
        requires
            - l implements the EXAMPLE_LAYER_ interface defined in layers_abstract.h
        ensures
            - tests l for compliance against the EXAMPLE_LAYER_ interface spec.
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

