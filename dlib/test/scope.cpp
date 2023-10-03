// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/scope.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.scope");

// ---------------------------------------------------------------------------------------------------

    class scope_tester : public tester
    {
    public:
        scope_tester (
        ) :
            tester ("test_scope",
                    "Runs tests on the scope_exit and related objects")
        {}

        void perform_test (
        )
        {
        }
    } a;
}