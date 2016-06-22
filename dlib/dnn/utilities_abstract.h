// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_UTILITIES_ABSTRACT_H_
#ifdef DLIB_DNn_UTILITIES_ABSTRACT_H_

#include "core_abstract.h"

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

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DNn_UTILITIES_ABSTRACT_H_ 


