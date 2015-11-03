// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_LOSS_ABSTRACT_H_
#ifdef DLIB_DNn_LOSS_ABSTRACT_H_

#include "core_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class EXAMPLE_LOSS_LAYER_
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                A loss layer is the final layer in a deep neural network.  It computes the
                task loss.  That is, it computes a number that tells us how well the
                network is performing on some task, such as predicting a binary label.  

                You can use one of the loss layers that comes with dlib (defined below).
                But importantly, you are able to define your own loss layers to suit your
                needs.  You do this by creating a class that defines an interface matching
                the one described by this EXAMPLE_LOSS_LAYER_ class.  Note that there is no
                dlib::EXAMPLE_LOSS_LAYER_ type.  It is shown here purely to document the
                interface that a loss layer must implement.

                A loss layer can optionally provide a to_label() method that converts the
                output of a network into a user defined type.  If to_label() is not
                provided then the operator() methods of add_loss_layer will not be
                available, but otherwise everything will function as normal.

                Finally, note that there are two broad flavors of loss layer, supervised
                and unsupervised.  The EXAMPLE_LOSS_LAYER_ as shown here is a supervised
                layer.  To make an unsupervised loss you simply leave out the label_type
                typedef, to_label(), and the truth iterator argument to compute_loss().
        !*/

    public:

        // sample_expansion_factor must be > 0
        const static unsigned int sample_expansion_factor;
        typedef whatever_type_you_use_for_labels label_type;

        EXAMPLE_LOSS_LAYER_ (
            const EXAMPLE_LOSS_LAYER_& item
        );
        /*!
            ensures
                - EXAMPLE_LOSS_LAYER_ objects are copy constructable
        !*/

        // Implementing to_label() is optional.
        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of
                  layers_abstract.h.
                - input_tensor was given as input to the network sub and the outputs are
                  now visible in layer<i>(sub).get_output(), for all valid i.
                - input_tensor.num_samples() > 0
                - input_tensor.num_samples()%sample_expansion_factor == 0.
                - iter == an iterator pointing to the beginning of a range of
                  input_tensor.num_samples()/sample_expansion_factor elements.  Moreover,
                  they must be label_type elements.
            ensures
                - Converts the output of the provided network to label_type objects and
                  stores the results into the range indicated by iter.  In particular, for
                  all valid i, it will be the case that:
                    *(iter+i/sample_expansion_factor) is the element corresponding to the
                    output of sub for the ith sample in input_tensor.
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            requires
                - SUBNET implements the SUBNET interface defined at the top of
                  layers_abstract.h.
                - input_tensor was given as input to the network sub and the outputs are
                  now visible in layer<i>(sub).get_output(), for all valid i.
                - input_tensor.num_samples() > 0
                - input_tensor.num_samples()%sample_expansion_factor == 0.
                - for all valid i:
                    - layer<i>(sub).get_gradient_input() has the same dimensions as
                      layer<i>(sub).get_output().
                - truth == an iterator pointing to the beginning of a range of
                  input_tensor.num_samples()/sample_expansion_factor elements.  Moreover,
                  they must be label_type elements.
                - for all valid i:
                    - *(truth+i/sample_expansion_factor) is the label of the ith sample in
                      input_tensor.
            ensures
                - This function computes a loss function that describes how well the output
                  of sub matches the expected labels given by truth.  Let's write the loss
                  function as L(input_tensor, truth, sub).  
                - Then compute_loss() computes the gradient of L() with respect to the
                  outputs in sub.  Specifically, compute_loss() adds the gradients into sub
                  by performing the following tensor additions, for all valid i: 
                    - layer<i>(sub).get_gradient_input() += the gradient of
                      L(input_tensor,truth,sub) with respect to layer<i>(sub).get_output().
                - returns L(input_tensor,truth,sub)
        !*/
    };

    void serialize(const EXAMPLE_LOSS_LAYER_& item, std::ostream& out);
    void deserialize(EXAMPLE_LOSS_LAYER_& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    // For each loss layer you define, always define an add_loss_layer template so that
    // layers can be easily composed.  Moreover, the convention is that the layer class
    // ends with an _ while the add_loss_layer template has the same name but without the
    // trailing _.
    template <typename SUBNET>
    using EXAMPLE_LOSS_LAYER = add_loss_layer<EXAMPLE_LOSS_LAYER_, SUBNET>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    class loss_binary_hinge_ 
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object implements the loss layer interface defined above by
                EXAMPLE_LOSS_LAYER_.  In particular, you use this loss to perform binary
                classification with the hinge loss.  Therefore, the possible outputs/labels
                when using this loss are +1 and -1.
        !*/
    public:

        const static unsigned int sample_expansion_factor = 1;
        typedef float label_type;

        template <
            typename SUB_TYPE,
            typename label_iterator
            >
        void to_label (
            const tensor& input_tensor,
            const SUB_TYPE& sub,
            label_iterator iter
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
        !*/

        template <
            typename const_label_iterator,
            typename SUBNET
            >
        double compute_loss (
            const tensor& input_tensor,
            const_label_iterator truth, 
            SUBNET& sub
        ) const;
        /*!
            This function has the same interface as EXAMPLE_LOSS_LAYER_::to_label() except
            it has the additional calling requirements that: 
                - sub.get_output().nr() == 1
                - sub.get_output().nc() == 1
                - sub.get_output().k() == 1
                - all values pointed to by truth are +1 or -1.
        !*/

    };

    void serialize(const loss_binary_hinge_& item, std::ostream& out);
    void deserialize(loss_binary_hinge_& item, std::istream& in);
    /*!
        provides serialization support  
    !*/

    template <typename SUBNET>
    using loss_binary_hinge = add_loss_layer<loss_binary_hinge_, SUBNET>;

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_LOSS_ABSTRACT_H_

