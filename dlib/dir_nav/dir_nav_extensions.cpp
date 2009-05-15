// Copyright (C) 2009  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_DIR_NAV_EXTENSIONs_CPP_
#define DLIB_DIR_NAV_EXTENSIONs_CPP_

#include "dir_nav_extensions.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace implementation_details
    {
        void get_all_sub_dirs (
            const directory& top_of_tree,
            unsigned long max_depth,
            std::vector<directory>& result,
            std::vector<directory>& temp
        )
        {
            if (max_depth > 0)
            {
                top_of_tree.get_dirs(temp);
                const unsigned long start = result.size();
                result.insert(result.end(), temp.begin(), temp.end());
                const unsigned long end = start + temp.size();

                for (unsigned long i = start; i < end; ++i)
                {
                    get_all_sub_dirs(result[i], max_depth-1, result, temp);
                }
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DIR_NAV_EXTENSIONs_CPP_



