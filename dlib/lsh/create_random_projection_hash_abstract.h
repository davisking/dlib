// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CREATE_RANDOM_PROJECTION_HAsH_ABSTRACT_H__
#ifdef DLIB_CREATE_RANDOM_PROJECTION_HAsH_ABSTRACT_H__

#include "projection_hash_abstract.h"
#include "../rand.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename vector_type>
    projection_hash create_random_projection_hash (
        const vector_type& v,
        const int bits
    );

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CREATE_RANDOM_PROJECTION_HAsH_ABSTRACT_H__


