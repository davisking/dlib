// Copyright (C) 2020  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DNn_MISC_h
#define DLIB_DNn_MISC_h

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <typename label_type>
    struct weighted_label
    {
        weighted_label()
        {}

        weighted_label(label_type label, float weight = 1.f)
            : label(label), weight(weight)
        {}

        label_type label{};
        float weight = 1.f;
    };

}

#endif // DLIB_DNn_MISC_h

