// Copyright (C) 2008  Davis E. King (davisking@users.sourceforge.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <dlib/geometry.h>
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/string.h>
#include <dlib/matrix.h>

#include "tester.h"

namespace  
{

    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.geometry");

    void geometry_test (
    )
    /*!
        ensures
            - runs tests on the geometry stuff compliance with the specs
    !*/
    {        
        print_spinner();

        point p1;
        point p2(2,3);

        DLIB_CASSERT(p1.x() == 0,"");
        DLIB_CASSERT(p1.y() == 0,"");
        DLIB_CASSERT(p2.x() == 2,"");
        DLIB_CASSERT(p2.y() == 3,"");

        DLIB_CASSERT((-p2).x() == -2,"");
        DLIB_CASSERT((-p2).y() == -3,"");


        p2 += p2;
        DLIB_CASSERT(p2.x() == 4,"");
        DLIB_CASSERT(p2.y() == 6,"");

        dlib::vector<double> v1 = point(1,0);
        dlib::vector<double> v2(0,0,1);

        p1 = v2.cross(v1);
        DLIB_CASSERT(p1 == point(0,1),"");
        DLIB_CASSERT(p1 != point(1,1),"");
        DLIB_CASSERT(p1 != point(1,0),"");

        p1 = point(2,3);
        rectangle rect1 = p1;
        DLIB_CASSERT(rect1.width() == 1,"");
        DLIB_CASSERT(rect1.height() == 1,"");
        p2 = point(1,1);

        rect1 += p2;
        DLIB_CASSERT(rect1.left() == 1,"");
        DLIB_CASSERT(rect1.top() == 1,"");
        DLIB_CASSERT(rect1.right() == 2,"");
        DLIB_CASSERT(rect1.bottom() == 3,"");

        DLIB_CASSERT(rect1.width() == 2,"");
        DLIB_CASSERT(rect1.height() == 3,"");

        // test the iostream << and >> operators (via string_cast and cast_to_string)
        DLIB_CASSERT(string_cast<point>(" (1, 2 )") == point(1,2),"");
        DLIB_CASSERT(string_cast<point>(" ( -1, 2 )") == point(-1,2),"");
        DLIB_CASSERT(string_cast<rectangle>(" [(1, 2 )(3,4)]") == rectangle(1,2,3,4),"");
        DLIB_CASSERT(string_cast<dlib::vector<double> >(" (1, 2 , 3.5)") == dlib::vector<double>(1,2,3.5),"");

        DLIB_CASSERT(string_cast<rectangle>(cast_to_string(rect1)) == rect1,"");
        DLIB_CASSERT(string_cast<point>(cast_to_string(p1)) == p1,"");
        DLIB_CASSERT(string_cast<dlib::vector<double> >(cast_to_string(v1)) == v1,"");

        rectangle rect2;

        // test the serialization code
        ostringstream sout;
        serialize(rect1,sout);
        serialize(p1,sout);
        serialize(v1,sout);
        serialize(rect1,sout);
        serialize(p1,sout);
        serialize(v1,sout);

        istringstream sin(sout.str());

        deserialize(rect2,sin);
        deserialize(p2,sin);
        deserialize(v2,sin);
        DLIB_CASSERT(rect2 == rect1,"");
        DLIB_CASSERT(p2 == p1,"");
        DLIB_CASSERT(v2 == v1,"");
        deserialize(rect2,sin);
        deserialize(p2,sin);
        deserialize(v2,sin);
        DLIB_CASSERT(rect2 == rect1,"");
        DLIB_CASSERT(p2 == p1,"");
        DLIB_CASSERT(v2 == v1,"");
        DLIB_CASSERT(sin,"");
        DLIB_CASSERT(sin.get() == EOF,"");


        v1.x() = 1;
        v1.y() = 2;
        v1.z() = 3;

        matrix<double> mv = v1;
        DLIB_CASSERT(mv.nr() == 3,"");
        DLIB_CASSERT(mv.nc() == 1,"");
        DLIB_CASSERT(mv(0) == 1,"");
        DLIB_CASSERT(mv(1) == 2,"");
        DLIB_CASSERT(mv(2) == 3,"");

        set_all_elements(mv,0);
        DLIB_CASSERT(mv(0) == 0,"");
        DLIB_CASSERT(mv(1) == 0,"");
        DLIB_CASSERT(mv(2) == 0,"");

        mv(0) = 5;
        mv(1) = 6;
        mv(2) = 7;

        v1 = mv;
        DLIB_CASSERT(v1.x() == 5,"");
        DLIB_CASSERT(v1.y() == 6,"");
        DLIB_CASSERT(v1.z() == 7,"");


        {
            dlib::vector<double,2> vd2;
            dlib::vector<double,3> vd3;
            dlib::vector<long,2> vl2;
            dlib::vector<long,3> vl3;

            vd2.x() = 2.3;
            vd2.y() = 4.7;

            vd3.z() = 9;

            vd3 = vd2;



            vl2 = vd3;
            vl3 = vd3;


            DLIB_CASSERT(vd2.z() == 0,"");
            DLIB_CASSERT(vd3.z() == 0,"");
            DLIB_CASSERT(vl2.z() == 0,"");
            DLIB_CASSERT(vl3.z() == 0,"");

            DLIB_CASSERT(vl2.x() == 2,"");
            DLIB_CASSERT(vl3.x() == 2,"");
            DLIB_CASSERT(vl2.y() == 5,"");
            DLIB_CASSERT(vl3.y() == 5,"");


            DLIB_CASSERT(abs(vd2.cross(vd3).dot(vd2)) < 1e-7,""); 
            DLIB_CASSERT(abs(vd3.cross(vd2).dot(vd2)) < 1e-7,""); 
            DLIB_CASSERT(abs(vd2.cross(vd3).dot(vd3)) < 1e-7,""); 
            DLIB_CASSERT(abs(vd3.cross(vd2).dot(vd3)) < 1e-7,""); 

            DLIB_CASSERT(abs(vl2.cross(vl3).dot(vl2)) == 0,""); 
            DLIB_CASSERT(abs(vl3.cross(vl2).dot(vl2)) == 0,""); 
            DLIB_CASSERT(abs(vl2.cross(vl3).dot(vl3)) == 0,""); 
            DLIB_CASSERT(abs(vl3.cross(vl2).dot(vl3)) == 0,""); 


            DLIB_CASSERT((vd2-vd3).length() < 1e-7,"");

            DLIB_CASSERT(vl2 == vl3,"");


            vl2.x() = 0;
            vl2.y() = 0;
            vl3 = vl2;

            vl2.x() = 4;
            vl3.y() = 3;

            DLIB_CASSERT(vl2.cross(vl3).length() == 12,"");
            DLIB_CASSERT(vl3.cross(vl2).length() == 12,"");


            matrix<double> m(3,3);
            m = 1,2,3,
                4,5,6,
                7,8,9;

            vd3.x() = 2;
            vd3.y() = 3;
            vd3.z() = 4;

            vd3 = m*vd3;

            DLIB_CASSERT(vd3.x() == 1*2 + 2*3 + 3*4,vd3.x() << " == " << (1*2 + 2*3 + 3*4));
            DLIB_CASSERT(vd3.y() == 4*2 + 5*3 + 6*4,"");
            DLIB_CASSERT(vd3.z() == 7*2 + 8*3 + 9*4,"");

            (vd3*2).dot(vd3);
            (vd2*2).dot(vd3);
            (vd3*2).dot(vd2);
            (vd2*2).dot(vd2);
            (2*vd3*2).dot(vd3);
            (2*vd2*2).dot(vd3);
            (2*vd3*2).dot(vd2);
            (2*vd2*2).dot(vd2);

            (vd2 + vd3).dot(vd2);
            (vd2 - vd3).dot(vd2);
            (vd2/2).dot(vd2);
            (vd3/2).dot(vd2);
        }

        {
            dlib::vector<double,2> vd2;
            dlib::vector<long,3> vl3;

            vl3.x() = 1;
            vl3.y() = 2;
            vl3.z() = 3;

            vd2.x() = 6.5;
            vd2.y() = 7.5;

            DLIB_CASSERT((vl3 + vd2).x() == 1+6.5,"");
            DLIB_CASSERT((vl3 + vd2).y() == 2+7.5,"");
            DLIB_CASSERT((vl3 + vd2).z() == 3+0,"");

            DLIB_CASSERT((vl3 - vd2).x() == 1-6.5,"");
            DLIB_CASSERT((vl3 - vd2).y() == 2-7.5,"");
            DLIB_CASSERT((vl3 - vd2).z() == 3-0,"");

        }

        {
            dlib::vector<double> v(3,4,5);
            DLIB_CASSERT((-v).x() == -3.0,"");
            DLIB_CASSERT((-v).y() == -4.0,"");
            DLIB_CASSERT((-v).z() == -5.0,"");
        }

        {
            rectangle rect;

            point tl(2,3);
            point tr(8,3);
            point bl(2,9);
            point br(8,9);

            rect += tl;
            rect += tr;
            rect += bl;
            rect += br;

            DLIB_CASSERT(rect.tl_corner() == tl, "");
            DLIB_CASSERT(rect.tr_corner() == tr, "");
            DLIB_CASSERT(rect.bl_corner() == bl, "");
            DLIB_CASSERT(rect.br_corner() == br, "");

        }

    }






    class geometry_tester : public tester
    {
    public:
        geometry_tester (
        ) :
            tester ("test_geometry",
                    "Runs tests on the geometry stuff.")
        {}

        void perform_test (
        )
        {
            geometry_test();
        }
    } a;

}



