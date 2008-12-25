// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/matrix.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <vector>
#include "../stl_checked.h"
#include "../array.h"
#include "../rand.h"

#include "tester.h"
#include <dlib/memory_manager_stateless.h>
#include <dlib/array2d.h>

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.matrix3");

    void matrix_test (
    )
    /*!
        ensures
            - runs tests on the matrix stuff compliance with the specs
    !*/
    {        
        typedef memory_manager_stateless<char>::kernel_2_2a MM;
        print_spinner();


        {
            matrix<long> m1(2,2), m2(2,2);

            m1 = 1, 2,
            3, 4;

            m2 = 4, 5,
            6, 7;


            DLIB_CASSERT(subm(tensor_product(m1,m2),range(0,1), range(0,1)) == 1*m2,"");
            DLIB_CASSERT(subm(tensor_product(m1,m2),range(0,1), range(2,3)) == 2*m2,"");
            DLIB_CASSERT(subm(tensor_product(m1,m2),range(2,3), range(0,1)) == 3*m2,"");
            DLIB_CASSERT(subm(tensor_product(m1,m2),range(2,3), range(2,3)) == 4*m2,"");
        }

    }






    class matrix_tester : public tester
    {
    public:
        matrix_tester (
        ) :
            tester ("test_matrix3",
                    "Runs tests on the matrix component.")
        {}

        void perform_test (
        )
        {
            matrix_test();
        }
    } a;

}


