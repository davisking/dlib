// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/optimization.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../stl_checked.h"
#include "../array.h"

#include "tester.h"


namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.optimization");

    void optimization_test (
    )
    /*!
        ensures
            - runs tests on the optimization stuff compliance with the specs
    !*/
    {        

    }



    class optimization_tester : public tester
    {
    public:
        optimization_tester (
        ) :
            tester ("test_optimization",
                    "Runs tests on the optimization component.")
        {}

        void perform_test (
        )
        {
            optimization_test();
        }
    } a;

}


