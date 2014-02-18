// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CROSS_VALIDATE_TRACK_ASSOCIATION_TrAINER_ABSTRACT_H__
#ifdef DLIB_CROSS_VALIDATE_TRACK_ASSOCIATION_TrAINER_ABSTRACT_H__

#include "structural_track_association_trainer_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename track_association_function,
        typename detection_type,
        typename detection_id_type
        >
    double test_track_association_function (
        const track_association_function& assoc,
        const std::vector<std::vector<std::vector<std::pair<detection_type,detection_id_type> > > >& samples
    );
    /*!
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename trainer_type,
        typename detection_type,
        typename detection_id_type
        >
    double cross_validate_track_association_trainer (
        const trainer_type& trainer,
        const std::vector<std::vector<std::vector<std::pair<detection_type,detection_id_type> > > >& samples,
        const long folds
    );

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_CROSS_VALIDATE_TRACK_ASSOCIATION_TrAINER_ABSTRACT_H__


