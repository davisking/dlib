// Copyright (C) 2013  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AVERAGE_PREcISION_H__
#define DLIB_AVERAGE_PREcISION_H__

#include "average_precision_abstract.h"
#include <vector>


namespace dlib
{
    namespace impl
    {
        inline bool get_bool_part (
            const bool& b
        ) { return b; }

        template <typename T>
        bool get_bool_part(const std::pair<T,bool>& item) { return item.second; }
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename alloc>
    double average_precision (
        const std::vector<T,alloc>& items,
        unsigned long missing_relevant_items = 0
    )
    {
        using namespace dlib::impl;
        double precision_sum = 0;
        double relevant_count = 0;
        for (unsigned long i = 0; i < items.size(); ++i)
        {
            if (get_bool_part(items[i]))
            {
                ++relevant_count;
                precision_sum += relevant_count / (i+1);
            }
        }

        relevant_count += missing_relevant_items;

        if (relevant_count != 0)
            return precision_sum/relevant_count;
        else
            return 1; 
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_AVERAGE_PREcISION_H__

