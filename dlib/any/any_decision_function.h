// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_AnY_DECISION_FUNCTION_Hh_
#define DLIB_AnY_DECISION_FUNCTION_Hh_

#include "any_decision_function_abstract.h"
#include "any_function.h"
#include "../algs.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <class sample_type, class result_type = double>
    using any_decision_function = any_function<result_type(const sample_type&)>;

// ----------------------------------------------------------------------------------------

}


#endif // DLIB_AnY_DECISION_FUNCTION_Hh_
