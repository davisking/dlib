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
    point input_tensor_to_output_tensor(
        const net_type& net,
        point p 
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_skip_layer, or add_tag_layer.
            - All layers in the net must provide map_input_to_output() functions.
        ensures
            - Given a point (i.e. a row,column coordinate) in the input tensor given to
              net, this function returns the corresponding point in the output tensor
              net.get_output().  This kind of mapping is useful when working with fully
              convolutional networks as you will often want to know what parts of the
              output feature maps correspond to what parts of the input.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename net_type>
    point output_tensor_to_input_tensor(
        const net_type& net,
        point p  
    );
    /*!
        requires
            - net_type is an object of type add_layer, add_skip_layer, or add_tag_layer.
            - All layers in the net must provide map_output_to_input() functions.
        ensures
            - This function provides the reverse mapping of input_tensor_to_output_tensor().
              That is, given a point in net.get_output(), what is the corresponding point
              in the input tensor?
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_ABSTRACT_H_ 


