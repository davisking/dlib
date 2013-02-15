// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_AVERAGE_PREcISION_ABSTRACT_H__
#ifdef DLIB_AVERAGE_PREcISION_ABSTRACT_H__

#include <vector>

namespace dlib
{
    double average_precision (
        const std::vector<bool>& items,
        unsigned long missing_relevant_items = 0
    );
    /*!
        ensures
            - Interprets items as a list of relevant and non-relevant items in a response
              from an information retrieval system.  In particular, items with a true value
              are relevant and false items are non-relevant.  This function then returns
              the average precision of the ranking of the given items.  For, example, the
              ranking [true, true, true, true, false] would have an average precision of 1.
              On the other hand, the ranking of [true false false true] would have an
              average precision of 0.75 (because the first true has a precision of 1 and
              the second true has a precision of 0.5, giving an average of 0.75).
            - As a special case, if item contains no true elements then the average
              precision is considered to be 1.
            - This function will add in missing_relevant_items number of items with a
              precision of zero into the average value returned.  For example, the average
              precision of the ranking [true, true] if there are 2 missing relevant items
              is 0.5.
    !*/

}

#endif // DLIB_AVERAGE_PREcISION_H__


