// Copyright (C) 2007  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.


#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/pixel.h>
#include <dlib/matrix.h>
#include <dlib/image_io.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.pixel");

    // Compile time tests
    struct not_a_pixel_type{};

    static_assert(is_pixel_type<rgb_pixel>::value, "bad trait definition");
    static_assert(is_pixel_type<bgr_pixel>::value, "bad trait definition");
    static_assert(is_pixel_type<rgb_alpha_pixel>::value, "bad trait definition");
    static_assert(is_pixel_type<bgr_alpha_pixel>::value, "bad trait definition");
    static_assert(is_pixel_type<hsi_pixel>::value, "bad trait definition");
    static_assert(is_pixel_type<hsv_pixel>::value, "bad trait definition");
    static_assert(is_pixel_type<lab_pixel>::value, "bad trait definition");

    static_assert(is_pixel_type<char>::value,           "bad trait definition");
    static_assert(is_pixel_type<signed char>::value,    "bad trait definition");
    static_assert(is_pixel_type<unsigned char>::value,  "bad trait definition");
    static_assert(is_pixel_type<short>::value,          "bad trait definition");
    static_assert(is_pixel_type<unsigned short>::value, "bad trait definition");
    static_assert(is_pixel_type<int>::value,            "bad trait definition");
    static_assert(is_pixel_type<unsigned int>::value,   "bad trait definition");
    static_assert(is_pixel_type<long>::value,           "bad trait definition");
    static_assert(is_pixel_type<unsigned long>::value,  "bad trait definition");
    static_assert(is_pixel_type<int64>::value,          "bad trait definition");
    static_assert(is_pixel_type<uint64>::value,         "bad trait definition");

    static_assert(is_pixel_type<float>::value,                      "bad trait definition");
    static_assert(is_pixel_type<double>::value,                     "bad trait definition");
    static_assert(is_pixel_type<long double>::value,                "bad trait definition");
    static_assert(is_pixel_type<std::complex<float>>::value,        "bad trait definition");
    static_assert(is_pixel_type<std::complex<double>>::value,       "bad trait definition");
    static_assert(is_pixel_type<std::complex<long double>>::value,  "bad trait definition");

    static_assert(!is_pixel_type<not_a_pixel_type>::value, "bad trait definition");

    void pixel_test (
    )
    /*!
        ensures
            - runs tests on pixel objects and functions for compliance with the specs 
    !*/
    {        

        print_spinner();

        unsigned char p_gray;
        unsigned short p_gray16;
        long p_int;
        float p_float;
        signed char p_schar;
        rgb_pixel p_rgb,p_rgb2;
        hsi_pixel p_hsi, p_hsi2;
        hsv_pixel p_hsv, p_hsv2;
        rgb_alpha_pixel p_rgba;
        lab_pixel p_lab, p_lab2;

        assign_pixel(p_int, 0.0f);
        assign_pixel(p_float, 0.0f);
        assign_pixel(p_schar, 0);

        assign_pixel(p_gray, -2);
        assign_pixel(p_rgb,0);
        assign_pixel(p_hsi, -4);
        assign_pixel(p_hsv, -4);
        assign_pixel(p_rgba, p_int);
        assign_pixel(p_gray16,0);
        assign_pixel(p_lab,-400);

        DLIB_TEST(p_int == 0);
        DLIB_TEST(p_float == 0);
        DLIB_TEST(p_schar == 0);

        DLIB_TEST(p_gray == 0);
        DLIB_TEST(p_gray16 == 0);

        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 0);
        DLIB_TEST(p_rgb.blue == 0);

        DLIB_TEST(p_rgba.red == 0);
        DLIB_TEST(p_rgba.green == 0);
        DLIB_TEST(p_rgba.blue == 0);
        DLIB_TEST(p_rgba.alpha == 255);

        DLIB_TEST(p_hsi.h == 0);
        DLIB_TEST(p_hsi.s == 0);
        DLIB_TEST(p_hsi.i == 0);

        DLIB_TEST(p_hsv.h == 0);
        DLIB_TEST(p_hsv.s == 0);
        DLIB_TEST(p_hsv.v == 0);

        DLIB_TEST(p_lab.l == 0);
        DLIB_TEST(p_lab.a == 128);
        DLIB_TEST(p_lab.b == 128);

        assign_pixel(p_gray,10);
        assign_pixel(p_gray16,10);
        assign_pixel(p_rgb,10);
        assign_pixel(p_hsi,10);
        assign_pixel(p_hsv,10);
        assign_pixel(p_rgba,10);
        assign_pixel(p_lab,10);

        assign_pixel(p_int, -10);
        assign_pixel(p_float, -10);
        assign_pixel(p_schar, -10);

        DLIB_TEST(p_int == -10);
        DLIB_TEST(p_float == -10);
        DLIB_TEST(p_schar == -10);

        DLIB_TEST(p_gray == 10);
        DLIB_TEST(p_gray16 == 10);

        DLIB_TEST(p_rgb.red == 10);
        DLIB_TEST(p_rgb.green == 10);
        DLIB_TEST(p_rgb.blue == 10);

        DLIB_TEST(p_rgba.red == 10);
        DLIB_TEST(p_rgba.green == 10);
        DLIB_TEST(p_rgba.blue == 10);
        DLIB_TEST(p_rgba.alpha == 255);

        DLIB_TEST(p_hsi.h == 0);
        DLIB_TEST(p_hsi.s == 0);
        DLIB_TEST(p_hsi.i == 10);

        DLIB_TEST(p_hsv.h == 0);
        DLIB_TEST(p_hsv.s == 0);
        DLIB_TEST(p_hsv.v == 10);

        DLIB_TEST(p_lab.l == 10);
        DLIB_TEST(p_lab.a == 128);
        DLIB_TEST(p_lab.b == 128);

        assign_pixel(p_gray16,12345);
        DLIB_TEST(p_gray16 == 12345);

        assign_pixel(p_float,3.141);
        DLIB_TEST(p_float == 3.141f);

        p_rgb.red = 255;
        p_rgb.green = 100;
        p_rgb.blue = 50;

        p_rgba.alpha = 4;
        assign_pixel(p_gray,p_rgb);
        assign_pixel(p_rgb,p_rgb);
        assign_pixel(p_rgba,p_rgb);
        assign_pixel(p_hsi,p_rgb);
        assign_pixel(p_hsv,p_rgb);
        assign_pixel(p_lab,p_rgb);

        assign_pixel(p_float,p_rgb);
        assign_pixel(p_int,p_rgb);
        assign_pixel(p_schar,p_rgb);

        DLIB_TEST(p_schar == std::numeric_limits<signed char>::max());

        DLIB_TEST(p_int == (255+100+50)/3);
        DLIB_TEST_MSG(p_float == (255+100+50)/3, p_float - (255+100+50)/3);
        DLIB_TEST(p_gray == (255+100+50)/3);

        DLIB_TEST(p_rgb.red == 255);
        DLIB_TEST(p_rgb.green == 100);
        DLIB_TEST(p_rgb.blue == 50);

        DLIB_TEST(p_rgba.red == 255);
        DLIB_TEST(p_rgba.green == 100);
        DLIB_TEST(p_rgba.blue == 50);
        DLIB_TEST(p_rgba.alpha == 255);

        DLIB_TEST(p_hsi.i > 0);
        DLIB_TEST(p_hsi.s > 0);
        DLIB_TEST(p_hsi.h > 0);

        DLIB_TEST(p_hsv.v > 0);
        DLIB_TEST(p_hsv.s > 0);
        DLIB_TEST(p_hsv.h > 0);

        DLIB_TEST(p_lab.l > 0);
        DLIB_TEST(p_lab.a > 0);
        DLIB_TEST(p_lab.b > 0);

        assign_pixel(p_rgb,0);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 0);
        DLIB_TEST(p_rgb.blue == 0);
        assign_pixel(p_rgb, p_hsi);

        DLIB_TEST_MSG(p_rgb.red > 251 ,(int)p_rgb.green);
        DLIB_TEST_MSG(p_rgb.green > 96 && p_rgb.green < 104,(int)p_rgb.green);
        DLIB_TEST_MSG(p_rgb.blue > 47 && p_rgb.blue < 53,(int)p_rgb.green);

        assign_pixel(p_rgb,0);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 0);
        DLIB_TEST(p_rgb.blue == 0);
        assign_pixel(p_rgb, p_hsv);

        DLIB_TEST_MSG(p_rgb.red > 251 ,(int)p_rgb.green);
        DLIB_TEST_MSG(p_rgb.green > 96 && p_rgb.green < 104,(int)p_rgb.green);
        DLIB_TEST_MSG(p_rgb.blue > 47 && p_rgb.blue < 53,(int)p_rgb.green);

        assign_pixel(p_rgb,0);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 0);
        DLIB_TEST(p_rgb.blue == 0);

        assign_pixel(p_rgb, p_lab);
        DLIB_TEST_MSG(p_rgb.red > 251 ,(int)p_rgb.green);
        DLIB_TEST_MSG(p_rgb.green > 96 && p_rgb.green < 104,(int)p_rgb.green);
        DLIB_TEST_MSG(p_rgb.blue > 47 && p_rgb.blue < 53,(int)p_rgb.green);

        assign_pixel(p_hsi2, p_hsi);
        DLIB_TEST(p_hsi.h == p_hsi2.h);
        DLIB_TEST(p_hsi.s == p_hsi2.s);
        DLIB_TEST(p_hsi.i == p_hsi2.i);
        assign_pixel(p_hsi,0);
        DLIB_TEST(p_hsi.h == 0);
        DLIB_TEST(p_hsi.s == 0);
        DLIB_TEST(p_hsi.i == 0);
        assign_pixel(p_hsi, p_rgba);

        DLIB_TEST(p_hsi.h == p_hsi2.h);
        DLIB_TEST(p_hsi.s == p_hsi2.s);
        DLIB_TEST(p_hsi.i == p_hsi2.i);

        assign_pixel(p_hsv2, p_hsv);
        DLIB_TEST(p_hsv.h == p_hsv2.h);
        DLIB_TEST(p_hsv.s == p_hsv2.s);
        DLIB_TEST(p_hsv.v == p_hsv2.v);
        assign_pixel(p_hsv,0);
        DLIB_TEST(p_hsv.h == 0);
        DLIB_TEST(p_hsv.s == 0);
        DLIB_TEST(p_hsv.v == 0);
        assign_pixel(p_hsv, p_rgba);

        DLIB_TEST(p_hsv.h == p_hsv2.h);
        DLIB_TEST(p_hsv.s == p_hsv2.s);
        DLIB_TEST(p_hsv.v == p_hsv2.v);

        assign_pixel(p_lab2, p_lab);
        DLIB_TEST(p_lab.l == p_lab2.l);
        DLIB_TEST(p_lab.a == p_lab2.a);
        DLIB_TEST(p_lab.b == p_lab2.b);
        assign_pixel(p_lab,0);
        DLIB_TEST(p_lab.l == 0);
        DLIB_TEST(p_lab.a == 128);
        DLIB_TEST(p_lab.b == 128);
        assign_pixel(p_lab, p_rgba);

        DLIB_TEST(p_lab.l == p_lab2.l);
        DLIB_TEST(p_lab.a == p_lab2.a);
        DLIB_TEST(p_lab.b == p_lab2.b);

        assign_pixel(p_rgba, 100);
        assign_pixel(p_gray, 10);
        assign_pixel(p_rgb, 10);
        assign_pixel(p_hsi, 10);
        assign_pixel(p_hsv, 10);

        assign_pixel(p_schar, 10);
        assign_pixel(p_float, 10);
        assign_pixel(p_int, 10);

        p_rgba.alpha = 0;
        assign_pixel(p_gray, p_rgba);
        DLIB_TEST(p_gray == 10);
        assign_pixel(p_schar, p_rgba);
        DLIB_TEST(p_schar == 10);
        assign_pixel(p_int, p_rgba);
        DLIB_TEST(p_int == 10);
        assign_pixel(p_float, p_rgba);
        DLIB_TEST(p_float == 10);
        assign_pixel(p_rgb, p_rgba);
        DLIB_TEST(p_rgb.red == 10);
        DLIB_TEST(p_rgb.green == 10);
        DLIB_TEST(p_rgb.blue == 10);

        assign_pixel(p_hsi, p_rgba);
        assign_pixel(p_hsi2, p_rgb);
        DLIB_TEST(p_hsi.h == 0);
        DLIB_TEST(p_hsi.s == 0);
        DLIB_TEST_MSG(p_hsi.i < p_hsi2.i+2 && p_hsi.i > p_hsi2.i -2,(int)p_hsi.i << "   " << (int)p_hsi2.i);

        assign_pixel(p_hsv, p_rgba);
        assign_pixel(p_hsv2, p_rgb);
        DLIB_TEST(p_hsv.h == 0);
        DLIB_TEST(p_hsv.s == 0);
        DLIB_TEST_MSG(p_hsv.v < p_hsv2.v+2 && p_hsv.v > p_hsv2.v -2,(int)p_hsv.v << "   " << (int)p_hsv2.v);

        // this value corresponds to RGB(10,10,10)
        p_lab.l = 7;
        p_lab.a = 128;
        p_lab.b = 128;

        assign_pixel(p_lab, p_rgba);
        assign_pixel(p_lab2, p_rgb);
        DLIB_TEST(p_lab.a == 128);
        DLIB_TEST(p_lab.b == 128);
        DLIB_TEST_MSG(p_lab.l < p_lab2.l+2 && p_lab.l > p_lab2.l -2,(int)p_lab.l << "   " << (int)p_lab2.l);

        assign_pixel(p_lab, 128);
        DLIB_TEST(p_lab.l == 128);
        DLIB_TEST(p_lab.a == 128);
        DLIB_TEST(p_lab.b == 128);
        assign_pixel(p_rgb, p_lab);
        //Lab midpoint (50,0,0) is not same as RGB midpoint (127,127,127)
        DLIB_TEST(p_rgb.red == 119);
        DLIB_TEST(p_rgb.green == 119);
        DLIB_TEST(p_rgb.blue == 119);

        //Lab limit values test
        //red, green, blue, yellow, black, white
        p_lab.l = 84;
        p_lab.a = 164;
        p_lab.b = 56;
        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 64);
        DLIB_TEST(p_rgb.blue == 194);

        p_lab.l = 255;
        p_lab.a = 0;
        p_lab.b = 0;
        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 255);
        DLIB_TEST(p_rgb.blue == 255);

        p_lab.l = 0;
        p_lab.a = 255;
        p_lab.b = 0;

        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 0);
        DLIB_TEST(p_rgb.blue == 195);

        p_lab.l = 0;
        p_lab.a = 0;
        p_lab.b = 255;

        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 45);
        DLIB_TEST(p_rgb.blue == 0);

        p_lab.l = 255;
        p_lab.a = 255;
        p_lab.b = 0;
        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 255); 
        DLIB_TEST(p_rgb.green == 139);
        DLIB_TEST(p_rgb.blue == 255);

        p_lab.l = 0;
        p_lab.a = 255;
        p_lab.b = 255;
        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 132);
        DLIB_TEST(p_rgb.green == 0);
        DLIB_TEST(p_rgb.blue == 0);

        p_lab.l = 255;
        p_lab.a = 0;
        p_lab.b = 255;
        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 0);
        DLIB_TEST(p_rgb.green == 255);
        DLIB_TEST(p_rgb.blue == 0);

        p_lab.l = 255;
        p_lab.a = 255;
        p_lab.b = 255;
        assign_pixel(p_rgb, p_lab);
        DLIB_TEST(p_rgb.red == 255);
        DLIB_TEST(p_rgb.green == 70);
        DLIB_TEST(p_rgb.blue == 0);

        //RGB limit tests
        p_rgb.red = 0;
        p_rgb.green = 0;
        p_rgb.blue = 0;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red < 3);
        DLIB_TEST(p_rgb2.green < 3);
        DLIB_TEST(p_rgb2.blue < 3);

        p_rgb.red = 255;
        p_rgb.green = 0;
        p_rgb.blue = 0;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red > 252);
        DLIB_TEST(p_rgb2.green < 3);
        DLIB_TEST(p_rgb2.blue < 3);

        p_rgb.red = 0;
        p_rgb.green = 255;
        p_rgb.blue = 0;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red < 8);
        DLIB_TEST(p_rgb2.green > 252);
        DLIB_TEST(p_rgb2.blue < 5);

        p_rgb.red = 0;
        p_rgb.green = 0;
        p_rgb.blue = 255;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red < 3);
        DLIB_TEST(p_rgb2.green < 3);
        DLIB_TEST(p_rgb2.blue > 252);

        p_rgb.red = 255;
        p_rgb.green = 255;
        p_rgb.blue = 0;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red > 252);
        DLIB_TEST(p_rgb2.green > 252);
        DLIB_TEST(p_rgb2.blue < 9);

        p_rgb.red = 0;
        p_rgb.green = 255;
        p_rgb.blue = 255;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red < 5);
        DLIB_TEST(p_rgb2.green > 252);
        DLIB_TEST(p_rgb2.blue > 252);

        p_rgb.red = 255;
        p_rgb.green = 0;
        p_rgb.blue = 255;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red> 252);
        DLIB_TEST(p_rgb2.green < 6);
        DLIB_TEST(p_rgb2.blue > 252);

        p_rgb.red = 255;
        p_rgb.green = 255;
        p_rgb.blue = 255;
        assign_pixel(p_lab, p_rgb);
        assign_pixel(p_rgb2, p_lab);
        DLIB_TEST(p_rgb2.red > 252 );
        DLIB_TEST(p_rgb2.green> 252);
        DLIB_TEST(p_rgb2.blue > 252);


        assign_pixel(p_rgba, 100);
        assign_pixel(p_gray, 10);
        assign_pixel(p_schar, 10);
        assign_pixel(p_float, 10);
        assign_pixel(p_int, 10);

        assign_pixel(p_rgb, 10);
        p_rgba.alpha = 128;
        assign_pixel(p_gray, p_rgba);
        assign_pixel(p_schar, p_rgba);
        assign_pixel(p_float, p_rgba);
        assign_pixel(p_int, p_rgba);
        assign_pixel(p_rgb, p_rgba);
        DLIB_TEST(p_gray == (100 + 10)/2);
        DLIB_TEST(p_schar == (100 + 10)/2);
        DLIB_TEST(p_int == (100 + 10)/2);
        DLIB_TEST(p_float == (100 + 10)/2);
        DLIB_TEST(p_rgb.red == (100 + 10)/2);
        DLIB_TEST(p_rgb.green == (100 + 10)/2);
        DLIB_TEST(p_rgb.blue == (100 + 10)/2);

        assign_pixel(p_rgba, 100);
        assign_pixel(p_gray, 10);
        assign_pixel(p_schar, 10);
        assign_pixel(p_int, 10);
        assign_pixel(p_float, 10);
        assign_pixel(p_rgb, 10);
        DLIB_TEST(p_rgba.alpha == 255);
        assign_pixel(p_gray, p_rgba);
        assign_pixel(p_schar, p_rgba);
        assign_pixel(p_int, p_rgba);
        assign_pixel(p_float, p_rgba);
        assign_pixel(p_rgb, p_rgba);
        DLIB_TEST(p_gray == 100);
        DLIB_TEST(p_schar == 100);
        DLIB_TEST(p_int == 100);
        DLIB_TEST(p_float == 100);
        DLIB_TEST(p_rgb.red == 100);
        DLIB_TEST(p_rgb.green == 100);
        DLIB_TEST(p_rgb.blue == 100);


        p_rgb.red = 1;
        p_rgb.green = 2;
        p_rgb.blue = 3;

        p_rgba.red = 4;
        p_rgba.green = 5;
        p_rgba.blue = 6;
        p_rgba.alpha = 7;

        p_gray = 8;
        p_schar = 9;
        p_int = 10;
        p_float = 8.5;

        p_hsi.h = 9;
        p_hsi.s = 10;
        p_hsi.i = 11;

        p_hsv.h = 9;
        p_hsv.s = 10;
        p_hsv.v = 11;


        p_lab.l = 10;
        p_lab.a = 9;
        p_lab.b = 8;

        ostringstream sout;
        serialize(p_rgb,sout);
        serialize(p_rgba,sout);
        serialize(p_gray,sout);
        serialize(p_schar,sout);
        serialize(p_int,sout);
        serialize(p_float,sout);
        serialize(p_hsi,sout);
        serialize(p_hsv,sout);
        serialize(p_lab,sout);

        assign_pixel(p_rgb,0);
        assign_pixel(p_rgba,0);
        assign_pixel(p_gray,0);
        assign_pixel(p_schar,0);
        assign_pixel(p_int,0);
        assign_pixel(p_float,0);
        assign_pixel(p_hsi,0);
        assign_pixel(p_hsv,0);
        assign_pixel(p_lab,0);

        istringstream sin(sout.str());

        deserialize(p_rgb,sin);
        deserialize(p_rgba,sin);
        deserialize(p_gray,sin);
        deserialize(p_schar,sin);
        deserialize(p_int,sin);
        deserialize(p_float,sin);
        deserialize(p_hsi,sin);
        deserialize(p_hsv,sin);
        deserialize(p_lab,sin);

        DLIB_TEST(p_rgb.red == 1);
        DLIB_TEST(p_rgb.green == 2);
        DLIB_TEST(p_rgb.blue == 3);

        DLIB_TEST(p_rgba.red == 4);
        DLIB_TEST(p_rgba.green == 5);
        DLIB_TEST(p_rgba.blue == 6);
        DLIB_TEST(p_rgba.alpha == 7);

        DLIB_TEST(p_gray == 8);
        DLIB_TEST(p_schar == 9);
        DLIB_TEST(p_int == 10);
        DLIB_TEST(p_float == 8.5);

        DLIB_TEST(p_hsi.h == 9);
        DLIB_TEST(p_hsi.s == 10);
        DLIB_TEST(p_hsi.i == 11);

        DLIB_TEST(p_hsv.h == 9);
        DLIB_TEST(p_hsv.s == 10);
        DLIB_TEST(p_hsv.v == 11);

        DLIB_TEST(p_lab.l == 10);
        DLIB_TEST(p_lab.a == 9);
        DLIB_TEST(p_lab.b == 8);

        {
            matrix<double,1,1> m_gray, m_schar, m_int, m_float;
            matrix<double,3,1> m_rgb, m_hsi, m_hsv, m_lab;

            m_gray = pixel_to_vector<double>(p_gray);
            m_schar = pixel_to_vector<double>(p_schar);
            m_int = pixel_to_vector<double>(p_int);
            m_float = pixel_to_vector<double>(p_float);

            m_hsi = pixel_to_vector<double>(p_hsi);
            m_hsv = pixel_to_vector<double>(p_hsv);
            m_rgb = pixel_to_vector<double>(p_rgb);
            m_lab = pixel_to_vector<double>(p_lab);

            DLIB_TEST(m_gray(0) == p_gray);
            DLIB_TEST(m_float(0) == p_float);
            DLIB_TEST(m_int(0) == p_int);
            DLIB_TEST(m_schar(0) == p_schar);

            DLIB_TEST(m_rgb(0) == p_rgb.red);
            DLIB_TEST(m_rgb(1) == p_rgb.green);
            DLIB_TEST(m_rgb(2) == p_rgb.blue);
            DLIB_TEST(m_hsi(0) == p_hsi.h);
            DLIB_TEST(m_hsi(1) == p_hsi.s);
            DLIB_TEST(m_hsi(2) == p_hsi.i);
            DLIB_TEST(m_hsv(0) == p_hsv.h);
            DLIB_TEST(m_hsv(1) == p_hsv.s);
            DLIB_TEST(m_hsv(2) == p_hsv.v);
            DLIB_TEST(m_lab(0) == p_lab.l);
            DLIB_TEST(m_lab(1) == p_lab.a);
            DLIB_TEST(m_lab(2) == p_lab.b);

            DLIB_TEST(p_rgb.red == 1);
            DLIB_TEST(p_rgb.green == 2);
            DLIB_TEST(p_rgb.blue == 3);

            DLIB_TEST(p_rgba.red == 4);
            DLIB_TEST(p_rgba.green == 5);
            DLIB_TEST(p_rgba.blue == 6);
            DLIB_TEST(p_rgba.alpha == 7);

            DLIB_TEST(p_gray == 8);
            DLIB_TEST(p_int == 10);
            DLIB_TEST(p_float == 8.5);
            DLIB_TEST(p_schar == 9);

            DLIB_TEST(p_hsi.h == 9);
            DLIB_TEST(p_hsi.s == 10);
            DLIB_TEST(p_hsi.i == 11);

            DLIB_TEST(p_hsv.h == 9);
            DLIB_TEST(p_hsv.s == 10);
            DLIB_TEST(p_hsv.v == 11);

            DLIB_TEST(p_lab.l == 10);
            DLIB_TEST(p_lab.a == 9);
            DLIB_TEST(p_lab.b == 8);

            assign_pixel(p_gray,0);
            assign_pixel(p_hsi,0);
            assign_pixel(p_hsv,0);
            assign_pixel(p_rgb,0);
            assign_pixel(p_lab,0);

            vector_to_pixel(p_float, m_float);
            vector_to_pixel(p_gray, m_gray);
            vector_to_pixel(p_hsi, m_hsi);
            vector_to_pixel(p_hsv, m_hsv);
            vector_to_pixel(p_rgb, m_rgb);
            vector_to_pixel(p_lab, m_lab);

            DLIB_TEST(p_rgb.red == 1);
            DLIB_TEST(p_rgb.green == 2);
            DLIB_TEST(p_rgb.blue == 3);

            DLIB_TEST(p_rgba.red == 4);
            DLIB_TEST(p_rgba.green == 5);
            DLIB_TEST(p_rgba.blue == 6);
            DLIB_TEST(p_rgba.alpha == 7);

            DLIB_TEST(p_gray == 8);
            DLIB_TEST(p_float == 8.5);

            DLIB_TEST(p_hsi.h == 9);
            DLIB_TEST(p_hsi.s == 10);
            DLIB_TEST(p_hsi.i == 11);

            DLIB_TEST(p_hsv.h == 9);
            DLIB_TEST(p_hsv.s == 10);
            DLIB_TEST(p_hsv.v == 11);

            DLIB_TEST(p_lab.l == 10);
            DLIB_TEST(p_lab.a == 9);
            DLIB_TEST(p_lab.b == 8);
        }




        {
            unsigned char p_gray;
            unsigned short p_gray16;
            long p_int;
            float p_float;
            signed char p_schar;
            rgb_pixel p_rgb;
            hsi_pixel p_hsi, p_hsi2;
            hsv_pixel p_hsv, p_hsv2;
            rgb_alpha_pixel p_rgba;
            lab_pixel p_lab;


            assign_pixel(p_gray, 0);
            assign_pixel(p_gray16, 0);
            assign_pixel(p_int, 0);
            assign_pixel(p_float, 0);
            assign_pixel(p_schar, 0);
            assign_pixel(p_rgb, 0);
            assign_pixel(p_hsi, 0);
            assign_pixel(p_hsv, 0);
            assign_pixel(p_lab, 0);


            assign_pixel(p_gray, 100);
            assign_pixel(p_schar, p_gray);
            DLIB_TEST(p_schar == 100);

            assign_pixel(p_gray, 200);
            assign_pixel(p_schar, p_gray);
            DLIB_TEST(p_schar == std::numeric_limits<signed char>::max());

            assign_pixel(p_int, p_gray);
            DLIB_TEST(p_int == 200);

            assign_pixel(p_float, p_gray);
            DLIB_TEST(p_float == 200);

            assign_pixel(p_rgb, p_float);
            DLIB_TEST(p_rgb.red == 200);
            DLIB_TEST(p_rgb.green == 200);
            DLIB_TEST(p_rgb.blue == 200);

            p_schar = 0;
            assign_pixel(p_schar, p_rgb);
            DLIB_TEST(p_schar == std::numeric_limits<signed char>::max());


            p_schar = -10;
            assign_pixel(p_float, p_schar);
            DLIB_TEST(p_float == -10);
            assign_pixel(p_int, p_schar);
            DLIB_TEST(p_int == -10);
            assign_pixel(p_schar, p_schar);
            DLIB_TEST(p_schar == -10);
            assign_pixel(p_gray, p_schar);
            DLIB_TEST(p_gray == 0);

            assign_pixel(p_rgb, p_schar);
            DLIB_TEST(p_rgb.red == 0);
            DLIB_TEST(p_rgb.green == 0);
            DLIB_TEST(p_rgb.blue == 0);
            
            assign_pixel(p_gray16, p_schar);
            DLIB_TEST(p_gray16 == 0);

            DLIB_TEST(get_pixel_intensity(p_float) == -10);
            DLIB_TEST(get_pixel_intensity(p_int) == -10);
            DLIB_TEST(get_pixel_intensity(p_schar) == -10);
            DLIB_TEST(get_pixel_intensity(p_rgb) == 0);
            DLIB_TEST(get_pixel_intensity(p_gray16) == 0);

            p_rgb.red = 100;
            p_rgb.green = 100;
            p_rgb.blue = 100;
            DLIB_TEST(get_pixel_intensity(p_rgb) == 100);
            p_rgb.red = 1;
            p_rgb.green = 2;
            p_rgb.blue = 3;
            DLIB_TEST(get_pixel_intensity(p_rgb) == 2);
            p_rgba.alpha = 100;
            p_rgba.red = 100;
            p_rgba.green = 100;
            p_rgba.blue = 100;
            DLIB_TEST(get_pixel_intensity(p_rgba) == 100);
            p_rgba.red = 1;
            p_rgba.green = 2;
            p_rgba.blue = 3;
            p_rgba.alpha = 0;
            DLIB_TEST(get_pixel_intensity(p_rgba) == 2);
            p_hsi.h = 123;
            p_hsi.s = 100;
            p_hsi.i = 84;
            DLIB_TEST(get_pixel_intensity(p_hsi) == 84);
            p_hsv.h = 123;
            p_hsv.s = 100;
            p_hsv.v = 84;
            DLIB_TEST(get_pixel_intensity(p_hsv) == 84);

            p_lab.l = 123;
            p_lab.a = 100;
            p_lab.b = 84;
            DLIB_TEST(get_pixel_intensity(p_lab) == 123);

            p_float = 54.25;
            DLIB_TEST(get_pixel_intensity(p_float) == 54.25);

            assign_pixel(p_gray, p_float);
            DLIB_TEST(get_pixel_intensity(p_gray) == 54);

            assign_pixel_intensity(p_float, -1000);
            assign_pixel_intensity(p_schar, -100);
            assign_pixel_intensity(p_int, -10000);
            assign_pixel_intensity(p_gray, -100);

            p_rgba.red = 10;
            p_rgba.green = 10;
            p_rgba.blue = 10;
            p_rgba.alpha = 0;
            DLIB_TEST_MSG(get_pixel_intensity(p_rgba) == 10, (int)get_pixel_intensity(p_rgba));
            assign_pixel_intensity(p_rgba, 2);
            DLIB_TEST_MSG(p_rgba.red == 2, (int)p_rgba.red);
            DLIB_TEST_MSG(p_rgba.green == 2, (int)p_rgba.green);
            DLIB_TEST_MSG(p_rgba.blue == 2, (int)p_rgba.blue);
            DLIB_TEST_MSG(p_rgba.alpha == 0, (int)p_rgba.alpha);
            DLIB_TEST_MSG(get_pixel_intensity(p_rgba) == 2, (int)get_pixel_intensity(p_rgba));

            DLIB_TEST(p_float == -1000);
            DLIB_TEST(get_pixel_intensity(p_float) == -1000);
            DLIB_TEST(p_schar == -100);
            DLIB_TEST(get_pixel_intensity(p_schar) == -100);
            DLIB_TEST(p_int == -10000);
            DLIB_TEST(get_pixel_intensity(p_int) == -10000);
            DLIB_TEST(p_gray == 0);
            assign_pixel_intensity(p_gray, 1000);
            DLIB_TEST(p_gray == 255);
            DLIB_TEST(get_pixel_intensity(p_gray) == 255);

            assign_pixel_intensity(p_float, p_gray);
            DLIB_TEST(p_float == 255);
            DLIB_TEST(get_pixel_intensity(p_float) == 255);

            assign_pixel_intensity(p_int, p_gray);
            DLIB_TEST(p_int == 255);
            DLIB_TEST(get_pixel_intensity(p_int) == 255);


            p_float = 1e10;
            assign_pixel(p_schar, p_float);
            DLIB_TEST(p_schar == std::numeric_limits<signed char>::max());

            p_float = -1e10;
            assign_pixel(p_schar, p_float);
            DLIB_TEST(p_schar == std::numeric_limits<signed char>::min());

            double p_double = 1e200;
            assign_pixel(p_float, p_double);
            DLIB_TEST(p_float == std::numeric_limits<float>::max());

            p_double = -1e200;
            assign_pixel(p_float, p_double);
            DLIB_TEST(p_float == -std::numeric_limits<float>::max());
        }


    }




    class pixel_tester : public tester
    {
    public:
        pixel_tester (
        ) :
            tester ("test_pixel",
                    "Runs tests on the pixel objects and functions.")
        {}

        void perform_test (
        )
        {
            pixel_test();
        }
    } a;

}
