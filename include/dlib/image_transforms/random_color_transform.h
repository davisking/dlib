// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RANDOM_cOLOR_TRANSFORM_Hh_
#define DLIB_RANDOM_cOLOR_TRANSFORM_Hh_

#include "random_color_transform_abstract.h"
#include "../image_processing/generic_image.h"
#include "../pixel.h"
#include "../rand.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class random_color_transform
    {
    public:

        random_color_transform (
            dlib::rand& rnd,
            const double gamma_magnitude = 0.5,
            const double color_magnitude = 0.2
        )
        {
            // pick a random gamma correction factor.  
            double gamma = std::max(0.0, 1 + gamma_magnitude*(rnd.get_random_double()-0.5));

            // pick a random color balancing scheme.
            double red_scale = 1-rnd.get_random_double()*color_magnitude;
            double green_scale = 1-rnd.get_random_double()*color_magnitude;
            double blue_scale = 1-rnd.get_random_double()*color_magnitude;
            const double m = 255*std::max(std::max(red_scale,green_scale),blue_scale);
            red_scale /= m;
            green_scale /= m;
            blue_scale /= m;

            // Now compute a lookup table for all the color channels.  The table tells us
            // what the transform does.
            table.resize(256*3);
            unsigned long i = 0;
            for (int k = 0; k < 256; ++k)
            {
                double v = 255*std::pow(k*red_scale, gamma);
                table[i++] = (unsigned char)(v + 0.5);
            }
            for (int k = 0; k < 256; ++k)
            {
                double v = 255*std::pow(k*green_scale, gamma);
                table[i++] = (unsigned char)(v + 0.5);
            }
            for (int k = 0; k < 256; ++k)
            {
                double v = 255*std::pow(k*blue_scale, gamma);
                table[i++] = (unsigned char)(v + 0.5);
            }
        }

        rgb_pixel operator()(rgb_pixel p) const
        {
            p.red = table[(unsigned int)p.red];
            p.green = table[(unsigned int)p.green+256];
            p.blue = table[(unsigned int)p.blue+512];
            return p;
        }

    private:
        std::vector<unsigned char> table;
    };

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    void disturb_colors (
        image_type& img_,
        dlib::rand& rnd,
        const double gamma_magnitude = 0.5,
        const double color_magnitude = 0.2
    )
    {
        image_view<image_type> img(img_);
        random_color_transform tform(rnd, gamma_magnitude, color_magnitude);
        for (long r = 0; r < img.nr(); ++r)
        {
            for (long c = 0; c < img.nc(); ++c)
            {
                rgb_pixel temp;
                assign_pixel(temp, img[r][c]);
                temp = tform(temp);
                assign_pixel(img[r][c], temp);
            }
        }
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    void apply_random_color_offset (
        image_type& img_,
        dlib::rand& rnd
    )
    {
        // Make a random color offset.  This tform matrix came from looking at the
        // covariance matrix of RGB values in a bunch of images.  In particular, if you
        // multiply Gaussian random vectors by tform it will result in vectors with the
        // same covariance matrix as the original RGB data.  Also, this color transform is
        // what is suggested by the paper:
        //  Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
        //  classification with deep convolutional neural networks." Advances in neural
        //  information processing systems. 2012.
        // Except that we used the square root of the eigenvalues (which I'm pretty sure is
        // what the authors intended).
        matrix<double,3,3> tform;
        tform = -66.379,    25.094,   6.79698, 
                -68.0492, -0.302309,  -13.9539,
                -68.4907,  -24.0199,   7.27653; 
        matrix<double,3,1> v;
        v = rnd.get_random_gaussian(),rnd.get_random_gaussian(),rnd.get_random_gaussian();
        v = round(tform*0.1*v);
        const int roffset = v(0);
        const int goffset = v(1);
        const int boffset = v(2);

        // Make up lookup tables that apply the color mapping so we don't have to put a
        // bunch of complicated conditional branches in the loop below.
        unsigned char rtable[256];
        unsigned char gtable[256];
        unsigned char btable[256];
        for (int i = 0; i < 256; ++i)
        {
            rtable[i] = put_in_range(0, 255, i+roffset);
            gtable[i] = put_in_range(0, 255, i+goffset);
            btable[i] = put_in_range(0, 255, i+boffset);
        }
        
        // now transform the image.
        image_view<image_type> img(img_);
        for (long r = 0; r < img.nr(); ++r)
        {
            for (long c = 0; c < img.nc(); ++c)
            {
                rgb_pixel temp;
                assign_pixel(temp, img[r][c]);
                temp.red   = rtable[temp.red];
                temp.green = gtable[temp.green];
                temp.blue  = btable[temp.blue];
                assign_pixel(img[r][c], temp);
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RANDOM_cOLOR_TRANSFORM_Hh_

