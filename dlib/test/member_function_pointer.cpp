// Copyright (C) 2006  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>

#include <dlib/member_function_pointer.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace std;
    using namespace dlib;

    logger dlog("test.member_function_pointer");

    class mfp_test_helper
    {
    public:
        mfp_test_helper (
        ): i(-1) {}


        int i;


        void go0 (
        ) { i = 0; }
        void go1 (
            int
        ) { i = 1; }
        void go2 (
            int,int
        ) { i = 2; }
        void go3 (
            int,int,int
        ) { i = 3; }
        void go4 (
            int,int,int,int
        ) { i = 4; }
    };

    template <
        template  <typename P1 = void, typename P2 = void, typename P3 = void, typename P4 = void> class mfp
        >
    void member_function_pointer_kernel_test (
    )
    /*!
        requires
            - mfp is an implementation of member_function_pointer/member_function_pointer_kernel_abstract.h 
        ensures
            - runs tests on mfp for compliance with the specs
    !*/
    {        


        mfp_test_helper helper;

        mfp<> a0, b0;
        mfp<int> a1, b1;
        mfp<int,int> a2, b2;
        mfp<int,int,int> a3, b3;
        mfp<int,int,int,int> a4, b4;

        member_function_pointer_kernel_c<mfp<> > a0c, b0c;
        member_function_pointer_kernel_c<mfp<int> > a1c, b1c;
        member_function_pointer_kernel_c<mfp<int,int> > a2c, b2c;
        member_function_pointer_kernel_c<mfp<int,int,int> > a3c, b3c;
        member_function_pointer_kernel_c<mfp<int,int,int,int> > a4c, b4c;

        DLIB_CASSERT(a0c == b0c, "");
        DLIB_CASSERT(a1c == b1c, "");
        DLIB_CASSERT(a2c == b2c, "");
        DLIB_CASSERT(a3c == b3c, "");
        DLIB_CASSERT(a4c == b4c, "");
        DLIB_CASSERT((a0c != b0c) == false, "");
        DLIB_CASSERT((a1c != b1c) == false, "");
        DLIB_CASSERT((a2c != b2c) == false, "");
        DLIB_CASSERT((a3c != b3c) == false, "");
        DLIB_CASSERT((a4c != b4c) == false, "");

        DLIB_CASSERT(a0.is_set() == false,"");
        DLIB_CASSERT(b0.is_set() == false,"");
        DLIB_CASSERT(a0c.is_set() == false,"");
        DLIB_CASSERT(b0c.is_set() == false,"");

        DLIB_CASSERT(a1.is_set() == false,"");
        DLIB_CASSERT(b1.is_set() == false,"");
        DLIB_CASSERT(a1c.is_set() == false,"");
        DLIB_CASSERT(b1c.is_set() == false,"");

        DLIB_CASSERT(a2.is_set() == false,"");
        DLIB_CASSERT(b2.is_set() == false,"");
        DLIB_CASSERT(a2c.is_set() == false,"");
        DLIB_CASSERT(b2c.is_set() == false,"");

        DLIB_CASSERT(a3.is_set() == false,"");
        DLIB_CASSERT(b3.is_set() == false,"");
        DLIB_CASSERT(a3c.is_set() == false,"");
        DLIB_CASSERT(b3c.is_set() == false,"");

        DLIB_CASSERT(a4.is_set() == false,"");
        DLIB_CASSERT(b4.is_set() == false,"");
        DLIB_CASSERT(a4c.is_set() == false,"");
        DLIB_CASSERT(b4c.is_set() == false,"");

        a0.set(helper,&mfp_test_helper::go0);
        a0c.set(helper,&mfp_test_helper::go0);
        DLIB_CASSERT(a0.is_set() == true,"");
        DLIB_CASSERT(a0c.is_set() == true,"");
        DLIB_CASSERT(b0.is_set() == false,"");
        DLIB_CASSERT(b0c.is_set() == false,"");

        a0 = a0;
        DLIB_CASSERT(a0 == a0, "");
        DLIB_CASSERT(!(a0 != a0),"");
        DLIB_CASSERT(a0.is_set() == true,"");
        DLIB_CASSERT(a0c.is_set() == true,"");
        DLIB_CASSERT(b0.is_set() == false,"");
        DLIB_CASSERT(b0c.is_set() == false,"");

        swap(a0,b0);
        swap(a0c,b0c);
        DLIB_CASSERT(a0.is_set() == false,"");
        DLIB_CASSERT(a0c.is_set() == false,"");
        DLIB_CASSERT(b0.is_set() == true,"");
        DLIB_CASSERT(b0c.is_set() == true,"");

        a0 = b0;
        DLIB_CASSERT(a0 == a0, "");
        DLIB_CASSERT(a0 == b0, "");
        DLIB_CASSERT(!(a0 != b0),"");
        DLIB_CASSERT(a0.is_set() == true,"");
        DLIB_CASSERT(a0c.is_set() == false,"");
        DLIB_CASSERT(b0.is_set() == true,"");
        DLIB_CASSERT(b0c.is_set() == true,"");


        a0.clear();
        a0c.clear();
        b0.clear();
        b0c.clear();
        DLIB_CASSERT(a0.is_set() == false,"");
        DLIB_CASSERT(a0c.is_set() == false,"");
        DLIB_CASSERT(b0.is_set() == false,"");
        DLIB_CASSERT(b0c.is_set() == false,"");


        a1.set(helper,&mfp_test_helper::go1);
        a1c.set(helper,&mfp_test_helper::go1);
        DLIB_CASSERT(a1.is_set() == true,"");
        DLIB_CASSERT(a1c.is_set() == true,"");
        DLIB_CASSERT(b1.is_set() == false,"");
        DLIB_CASSERT(b1c.is_set() == false,"");
        swap(a1,b1);
        swap(a1c,b1c);
        DLIB_CASSERT(a1.is_set() == false,"");
        DLIB_CASSERT(a1c.is_set() == false,"");
        DLIB_CASSERT(b1.is_set() == true,"");
        DLIB_CASSERT(b1c.is_set() == true,"");


        a1 = b1;
        DLIB_CASSERT(a1 == a1, "");
        DLIB_CASSERT(a1 == b1, "");
        DLIB_CASSERT(!(a1 != b1),"");
        DLIB_CASSERT(a1.is_set() == true,"");
        DLIB_CASSERT(a1c.is_set() == false,"");
        DLIB_CASSERT(b1.is_set() == true,"");
        DLIB_CASSERT(b1c.is_set() == true,"");


        a1.clear();
        a1c.clear();
        b1.clear();
        b1c.clear();
        DLIB_CASSERT(a1.is_set() == false,"");
        DLIB_CASSERT(a1c.is_set() == false,"");
        DLIB_CASSERT(b1.is_set() == false,"");
        DLIB_CASSERT(b1c.is_set() == false,"");


        a2.set(helper,&mfp_test_helper::go2);
        a2c.set(helper,&mfp_test_helper::go2);
        DLIB_CASSERT(a2.is_set() == true,"");
        DLIB_CASSERT(a2c.is_set() == true,"");
        DLIB_CASSERT(b2.is_set() == false,"");
        DLIB_CASSERT(b2c.is_set() == false,"");
        swap(a2,b2);
        swap(a2c,b2c);
        DLIB_CASSERT(a2.is_set() == false,"");
        DLIB_CASSERT(a2c.is_set() == false,"");
        DLIB_CASSERT(b2.is_set() == true,"");
        DLIB_CASSERT(b2c.is_set() == true,"");

        a2 = b2;
        DLIB_CASSERT(a2 == a2, "");
        DLIB_CASSERT(a2 == b2, "");
        DLIB_CASSERT(!(a2 != b2),"");
        DLIB_CASSERT(a2.is_set() == true,"");
        DLIB_CASSERT(a2c.is_set() == false,"");
        DLIB_CASSERT(b2.is_set() == true,"");
        DLIB_CASSERT(b2c.is_set() == true,"");

        a2.clear();
        a2c.clear();
        b2.clear();
        b2c.clear();
        DLIB_CASSERT(a2.is_set() == false,"");
        DLIB_CASSERT(a2c.is_set() == false,"");
        DLIB_CASSERT(b2.is_set() == false,"");
        DLIB_CASSERT(b2c.is_set() == false,"");


        a3.set(helper,&mfp_test_helper::go3);
        a3c.set(helper,&mfp_test_helper::go3);
        DLIB_CASSERT(a3.is_set() == true,"");
        DLIB_CASSERT(a3c.is_set() == true,"");
        DLIB_CASSERT(b3.is_set() == false,"");
        DLIB_CASSERT(b3c.is_set() == false,"");
        swap(a3,b3);
        swap(a3c,b3c);
        DLIB_CASSERT(a3.is_set() == false,"");
        DLIB_CASSERT(a3c.is_set() == false,"");
        DLIB_CASSERT(b3.is_set() == true,"");
        DLIB_CASSERT(b3c.is_set() == true,"");

        a3 = b3;
        DLIB_CASSERT(a3 == a3, "");
        DLIB_CASSERT(a3 == b3, "");
        DLIB_CASSERT(!(a3 != b3),"");
        DLIB_CASSERT(a3.is_set() == true,"");
        DLIB_CASSERT(a3c.is_set() == false,"");
        DLIB_CASSERT(b3.is_set() == true,"");
        DLIB_CASSERT(b3c.is_set() == true,"");


        a3.clear();
        a3c.clear();
        b3.clear();
        b3c.clear();
        DLIB_CASSERT(a3.is_set() == false,"");
        DLIB_CASSERT(a3c.is_set() == false,"");
        DLIB_CASSERT(b3.is_set() == false,"");
        DLIB_CASSERT(b3c.is_set() == false,"");


        a4.set(helper,&mfp_test_helper::go4);
        a4c.set(helper,&mfp_test_helper::go4);
        DLIB_CASSERT(a4.is_set() == true,"");
        DLIB_CASSERT(a4c.is_set() == true,"");
        DLIB_CASSERT(b4.is_set() == false,"");
        DLIB_CASSERT(b4c.is_set() == false,"");
        swap(a4,b4);
        swap(a4c,b4c);
        DLIB_CASSERT(a4.is_set() == false,"");
        DLIB_CASSERT(a4c.is_set() == false,"");
        DLIB_CASSERT(b4.is_set() == true,"");
        DLIB_CASSERT(b4c.is_set() == true,"");

        a4 = b4;
        a4 = b4;
        a4 = b4;
        a4 = b4;
        DLIB_CASSERT(a4 == a4, "");
        DLIB_CASSERT(a4 == b4, "");
        DLIB_CASSERT(!(a4 != b4),"");
        DLIB_CASSERT(a4.is_set() == true,"");
        DLIB_CASSERT(a4c.is_set() == false,"");
        DLIB_CASSERT(b4.is_set() == true,"");
        DLIB_CASSERT(b4c.is_set() == true,"");


        a4.clear();
        a4c.clear();
        b4.clear();
        b4c.clear();
        DLIB_CASSERT(a4.is_set() == false,"");
        DLIB_CASSERT(a4c.is_set() == false,"");
        DLIB_CASSERT(b4.is_set() == false,"");
        DLIB_CASSERT(b4c.is_set() == false,"");


        a0.set(helper,&mfp_test_helper::go0);
        a0c.set(helper,&mfp_test_helper::go0);
        b0 = a0; 
        b0c = a0c;
        helper.i = -1; 
        a0();
        DLIB_CASSERT(helper.i == 0,"");
        helper.i = -1; 
        b0();
        DLIB_CASSERT(helper.i == 0,"");
        helper.i = -1; 
        a0c();
        DLIB_CASSERT(helper.i == 0,"");
        helper.i = -1; 
        b0c();
        DLIB_CASSERT(helper.i == 0,"");


        a1.set(helper,&mfp_test_helper::go1);
        a1c.set(helper,&mfp_test_helper::go1);
        b1 = a1;
        b1c = a1c;
        helper.i = -1; 
        a1(0);
        DLIB_CASSERT(helper.i == 1,"");
        helper.i = -1; 
        b1(0);
        DLIB_CASSERT(helper.i == 1,"");
        helper.i = -1; 
        a1c(0);
        DLIB_CASSERT(helper.i == 1,"");
        helper.i = -1; 
        b1c(0);
        DLIB_CASSERT(helper.i == 1,"");


        a2.set(helper,&mfp_test_helper::go2);
        a2c.set(helper,&mfp_test_helper::go2);
        b2 = a2;
        b2c = a2c;
        helper.i = -1; 
        a2(0,0);
        DLIB_CASSERT(helper.i == 2,"");
        helper.i = -1; 
        b2(0,0);
        DLIB_CASSERT(helper.i == 2,"");
        helper.i = -1; 
        a2c(0,0);
        DLIB_CASSERT(helper.i == 2,"");
        helper.i = -1; 
        b2c(0,0);
        DLIB_CASSERT(helper.i == 2,"");


        a3.set(helper,&mfp_test_helper::go3);
        a3c.set(helper,&mfp_test_helper::go3);
        b3 = a3;
        b3c = a3c;
        helper.i = -1; 
        a3(0,0,0);
        DLIB_CASSERT(helper.i == 3,"");
        helper.i = -1; 
        b3(0,0,0);
        DLIB_CASSERT(helper.i == 3,"");
        helper.i = -1; 
        a3c(0,0,0);
        DLIB_CASSERT(helper.i == 3,"");
        helper.i = -1; 
        b3c(0,0,0);
        DLIB_CASSERT(helper.i == 3,"");


        a4.set(helper,&mfp_test_helper::go4);
        a4c.set(helper,&mfp_test_helper::go4);
        b4 = a4;
        b4c = a4c;
        helper.i = -1; 
        a4(0,0,0,0);
        DLIB_CASSERT(helper.i == 4,"");
        helper.i = -1; 
        b4(0,0,0,0);
        DLIB_CASSERT(helper.i == 4,"");
        helper.i = -1; 
        a4c(0,0,0,0);
        DLIB_CASSERT(helper.i == 4,"");
        helper.i = -1; 
        b4c(0,0,0,0);
        DLIB_CASSERT(helper.i == 4,"");



    }



    class member_function_pointer_tester : public tester
    {
    public:
        member_function_pointer_tester (
        ) :
            tester ("test_member_function_pointer",
                    "Runs tests on the member_function_pointer component.")
        {}

        void perform_test (
        )
        {
            member_function_pointer_kernel_test<member_function_pointer_kernel_1>();
        }
    } a;

}


