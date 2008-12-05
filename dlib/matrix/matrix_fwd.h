// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_MATRIx_FWD
#define DLIB_MATRIx_FWD

#include "../memory_manager.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct default_matrix_layout;

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        long num_rows = 0,
        long num_cols = 0,
        typename mem_manager = memory_manager<char>::kernel_1a,
        typename layout = default_matrix_layout 
        >
    class matrix; 

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_MATRIx_FWD

