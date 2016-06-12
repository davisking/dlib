// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DNn_UTILITIES_ABSTRACT_H_
#ifdef DLIB_DNn_UTILITIES_ABSTRACT_H_

#include "core_abstract.h"

namespace dlib
{

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


