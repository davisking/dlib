// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#include <sstream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <dlib/image_transforms.h>
//#include <dlib/gui_widgets.h>
#include <dlib/rand.h>

#include "tester.h"

namespace  
{
    using namespace test;
    using namespace dlib;
    using namespace std;

    logger dlog("test.pyramid_down");

// ----------------------------------------------------------------------------------------

void test_pyramid_down_graysclae()
{
    array2d<unsigned char> img, down;
    pyramid_down pyr;

    img.set_size(300,264);

    assign_all_pixels(img, 10);

    pyr(img, down);

    DLIB_TEST(std::abs(down.nr()*2 - img.nr()) < 5);
    DLIB_TEST(std::abs(down.nc()*2 - img.nc()) < 5);

    rectangle rect1 = get_rect(img);
    rectangle rect2 = pyr.rect_up(get_rect(down));
    double overlap = rect1.intersect(rect2).area() / (double)(rect1 + rect2).area();
    DLIB_TEST(overlap > 0.95);

    rect1 = get_rect(down);
    rect2 = pyr.rect_down(get_rect(img));
    overlap = rect1.intersect(rect2).area() / (double)(rect1 + rect2).area();
    DLIB_TEST(overlap > 0.95);

    DLIB_TEST(min(array_to_matrix(down)) == 10);
    DLIB_TEST(max(array_to_matrix(down)) == 10);
}

void test_pyramid_down_rgb()
{
    array2d<rgb_pixel> img;
    array2d<bgr_pixel> down;
    pyramid_down pyr;

    img.set_size(231, 351);

    assign_all_pixels(img, rgb_pixel(1,2,3));

    pyr(img, down);

    DLIB_TEST(std::abs(down.nr()*2 - img.nr()) < 5);
    DLIB_TEST(std::abs(down.nc()*2 - img.nc()) < 5);

    rectangle rect1 = get_rect(img);
    rectangle rect2 = pyr.rect_up(get_rect(down));
    double overlap = rect1.intersect(rect2).area() / (double)(rect1 + rect2).area();
    DLIB_TEST(overlap > 0.95);

    rect1 = get_rect(down);
    rect2 = pyr.rect_down(get_rect(img));
    overlap = rect1.intersect(rect2).area() / (double)(rect1 + rect2).area();
    DLIB_TEST(overlap > 0.95);

    bool pixels_match = true;
    for (long r = 0; r < down.nr(); ++r)
    {
        for (long c = 0; c < down.nc(); ++c)
        {
            if (down[r][c].red != 1 ||
                down[r][c].green != 2 ||
                down[r][c].blue != 3 )
            {
                pixels_match = false;
            }
        }
    }
    DLIB_TEST(pixels_match);
}

//  ----------------------------------------------------------------------------

template <typename image_type, typename pixel_type>
void draw_rectangle (
    image_type& img,
    const rectangle& rect,
    const pixel_type& p
)
{
    rectangle area = rect.intersect(get_rect(img));

    for (long r = area.top(); r <= area.bottom(); ++r)
    {
        for (long c = area.left(); c <= area.right(); ++c)
        {
            assign_pixel(img[r][c], p);
        }
    }
}
    
//  ----------------------------------------------------------------------------

template <typename image_type>
rgb_pixel mean_pixel (
    const image_type& img,
    const rectangle& rect
)
{
    long red = 0;
    long green = 0;
    long blue = 0;
    for (long r = rect.top(); r <= rect.bottom(); ++r)
    {
        for (long c = rect.left(); c <= rect.right(); ++c)
        {
            red += img[r][c].red;
            green += img[r][c].green;
            blue += img[r][c].blue;
        }
    }

    const long n = rect.area();
    return rgb_pixel(red/n, green/n, blue/n);
}

//  ----------------------------------------------------------------------------

void test_pyramid_down_rgb2()
{
    array2d<rgb_pixel> img, img3;
    array2d<unsigned char> img2, img4;


    img.set_size(300,400);
    assign_all_pixels(img, 0);
    rectangle rect1 = centered_rect( 10,10, 14, 14);
    rectangle rect2 = centered_rect( 100,100, 34, 42);
    rectangle rect3 = centered_rect( 310,215, 65, 21);

    draw_rectangle(img, rect1, rgb_pixel(255,0,0));
    draw_rectangle(img, rect2, rgb_pixel(0,255,0));
    draw_rectangle(img, rect3, rgb_pixel(0,0,255));



    pyramid_down pyr;

    pyr(img, img2);
    pyr(img, img3);


    rect1 = pyr.rect_down(rect1);
    rect2 = pyr.rect_down(rect2);
    rect3 = pyr.rect_down(rect3);

    /*
    image_window my_window(img);
    image_window win2(img2);
    image_window win3(img3);
    win2.add_overlay(image_window::overlay_rect(rect1, rgb_pixel(255,0,0)));
    win2.add_overlay(image_window::overlay_rect(rect2, rgb_pixel(255,0,0)));
    win2.add_overlay(image_window::overlay_rect(rect3, rgb_pixel(255,0,0)));
    win3.add_overlay(image_window::overlay_rect(rect1, rgb_pixel(255,0,0)));
    win3.add_overlay(image_window::overlay_rect(rect2, rgb_pixel(255,0,0)));
    win3.add_overlay(image_window::overlay_rect(rect3, rgb_pixel(255,0,0)));
    */


    DLIB_TEST(std::abs((int)mean(subm(matrix_cast<long>(array_to_matrix(img2)),rect1)) - 255/3) < 9);
    DLIB_TEST(std::abs((int)mean(subm(matrix_cast<long>(array_to_matrix(img2)),rect2)) - 255/3) < 4);
    DLIB_TEST(std::abs((int)mean(subm(matrix_cast<long>(array_to_matrix(img2)),rect1)) - 255/3) < 9);
    assign_image(img4, img);
    DLIB_TEST(std::abs((int)mean(array_to_matrix(img4)) - mean(array_to_matrix(img2))) < 2);


    rgb_pixel mean1 = mean_pixel(img3, rect1);
    rgb_pixel mean2 = mean_pixel(img3, rect2);
    rgb_pixel mean3 = mean_pixel(img3, rect3);
    rgb_pixel mean_all_true = mean_pixel(img, get_rect(img));
    rgb_pixel mean_all = mean_pixel(img3, get_rect(img3));
    DLIB_TEST(mean1.red > 232);
    DLIB_TEST(mean1.green < 3);
    DLIB_TEST(mean1.blue < 3);

    DLIB_TEST(mean2.red < 3);
    DLIB_TEST(mean2.green > 240);
    DLIB_TEST(mean2.blue < 3);

    DLIB_TEST(mean3.red < 4);
    DLIB_TEST(mean3.green < 3);
    DLIB_TEST(mean3.blue > 237);

    DLIB_TEST(std::abs((int)mean_all_true.red - mean_all.red) < 1);
    DLIB_TEST(std::abs((int)mean_all_true.green - mean_all.green) < 1);
    DLIB_TEST(std::abs((int)mean_all_true.blue - mean_all.blue) < 1);

    //my_window.wait_until_closed();
}


// ----------------------------------------------------------------------------------------


    class test_pyramid_down : public tester
    {
    public:
        test_pyramid_down (
        ) :
            tester ("test_pyramid_down",
                    "Runs tests on the pyramid_down() function.")
        {}

        void perform_test (
        )
        {
            print_spinner();
            test_pyramid_down_graysclae();
            print_spinner();
            test_pyramid_down_rgb();
            print_spinner();
            test_pyramid_down_rgb2();
        }
    } a;

}




