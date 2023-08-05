// Copyright (C) 2022  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_VISITORS_ABSTRACT_H_
#ifdef DLIB_DNn_VISITORS_ABSTRACT_H_

#include "input.h"
#include "layers.h"
#include "loss.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    void set_all_bn_running_stats_window_sizes (
        const net_type& net,
        unsigned long new_window_size
    );
    /*!
        requires
            - new_window_size > 0
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
        ensures
            - Sets the get_running_stats_window_size() field of all bn_ layers in net to
              new_window_size.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    void disable_duplicative_biases (
        const net_type& net
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
        ensures
            - Disables bias for all bn_ and layer_norm_ inputs.
            - Sets the get_bias_learning_rate_multiplier() and get_bias_weight_decay_multiplier()
              to zero of all bn_ and layer_norm_ inputs.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    void fuse_layers (
        net_type& net
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
            - net has been properly allocated, that is: count_parameters(net) > 0.
        ensures
            - Disables all the affine_ layers that have a convolution as an input.
            - Updates the convolution weights beneath the affine_ layers to produce the same
              output as with the affine_ layers enabled.
            - Disables all the relu_ layers that have a convolution as input.
            - Disables all the relu_ layers that have an affine_ layer as input, with a
              convolution as input.
            - Updates the convolution to apply a relu activation function, to produce the same
              output as with the relu_ layer enabled.
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

    template <typename net_type>
    void net_to_dot (
        const net_type& net,
        std::ostream& out
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
        ensures
            - Prints the given neural network object as an DOT document to the given output
              stream.
            - The contents of #out can be used by the dot program from Graphviz to export
              the network diagram to any supported format.
    !*/

    template <typename net_type>
    void net_to_dot (
        const net_type& net,
        const std::string& filename
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_loss_layer, add_skip_layer, or
              add_tag_layer.
        ensures
            - This function is just like the above net_to_dot(), except it writes to a file
              rather than an ostream.
    !*/
}

#endif // DLIB_DNn_VISITORS_ABSTRACT_H_
