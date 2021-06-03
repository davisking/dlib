// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_UTILITIES_ABSTRACT_H_
#ifdef DLIB_DNn_UTILITIES_ABSTRACT_H_

#include "core_abstract.h"
#include "../geometry/vector_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    double log1pexp(
        double x
    );
    /*!
        ensures
            - returns log(1+exp(x))
              (except computes it using a numerically accurate method)

        NOTE: For technical reasons, it is defined in misc.h.
    !*/

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

    template <typename net_type>
    void net_to_xml (
        const net_type& net,
        std::ostream& out
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
            - All layers in the net must provide to_xml() functions.
        ensures
            - Prints the given neural network object as an XML document to the given output
              stream.
    !*/

    template <typename net_type>
    void net_to_xml (
        const net_type& net,
        const std::string& filename
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
            - All layers in the net must provide to_xml() functions.
        ensures
            - This function is just like the above net_to_xml(), except it writes to a file
              rather than an ostream.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    dpoint input_tensor_to_output_tensor(
        const net_type& net,
        dpoint p 
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_skip_layer, or add_tag_layer.
            - All layers in the net must provide map_input_to_output() functions.
        ensures
            - Given a dpoint (i.e. a row,column coordinate) in the input tensor given to
              net, this function returns the corresponding dpoint in the output tensor
              net.get_output().  This kind of mapping is useful when working with fully
              convolutional networks as you will often want to know what parts of the
              output feature maps correspond to what parts of the input.
            - If the network contains skip layers then any layers skipped over by the skip
              layer are ignored for the purpose of computing this coordinate mapping.  That
              is, if you walk the network from the output layer to the input layer, where
              each time you encounter a skip layer you jump to the layer indicated by the
              skip layer, you will visit exactly the layers in the network involved in the
              input_tensor_to_output_tensor() calculation. This behavior is useful since it
              allows you to compute some auxiliary DNN as a separate branch of computation
              that is separate from the main network's job of running some kind of fully
              convolutional network over an image.  For instance, you might want to have a
              branch in your network that computes some global image level
              summarization/feature.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    dpoint output_tensor_to_input_tensor(
        const net_type& net,
        dpoint p  
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_skip_layer, or add_tag_layer.
            - All layers in the net must provide map_output_to_input() functions.
        ensures
            - This function provides the reverse mapping of input_tensor_to_output_tensor().
              That is, given a dpoint in net.get_output(), what is the corresponding dpoint
              in the input tensor?
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    inline size_t count_parameters(
        const net_type& net
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
        ensures
            - Returns the number of allocated parameters in the network. E.g. if the network has not
              been trained then, since nothing has been allocated yet, it will return 0.
    !*/

// ----------------------------------------------------------------------------------------

    template<typename net_type>
    void set_all_learning_rate_multipliers(
        net_type& net,
        double learning_rate_multiplier
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
            - learning_rate_multiplier >= 0
        ensures
            - Sets all learning_rate_multipliers and bias_learning_rate_multipliers in net
              to learning_rate_multiplier.
    !*/

// ----------------------------------------------------------------------------------------

    template <size_t begin, size_t end, typename net_type>
    void set_learning_rate_multipliers_range(
        net_type& net,
        double learning_rate_multiplier
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
            - learning_rate_multiplier >= 0
            - begin <= end <= net_type::num_layers
        ensures
            - Loops over the layers in the range [begin,end) in net and calls
              set_learning_rate_multiplier on them with the value of
              learning_rate_multiplier.
    !*/

// ----------------------------------------------------------------------------------------
}

#endif // DLIB_DNn_UTILITIES_ABSTRACT_H_ 


