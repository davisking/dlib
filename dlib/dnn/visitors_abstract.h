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
