// Copyright (C) 2023  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/optional.h>
#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;

    logger dlog("test.optional");

    class optional_tester : public tester
    {
    public:
        optional_tester (
        ) :
            tester ("optional",
                    "Runs tests on the optional object")
        {}

        void perform_test (
        )
        {
        }
    } a;
}