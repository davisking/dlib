// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_RANDOM_cOLOR_TRANSFORM_Hh_
#define DLIB_RANDOM_cOLOR_TRANSFORM_Hh_

#include "random_color_transform_abstract.h"
#include "../image_processing/generic_image.h"
#include "../pixel.h"
#include "../rand.h"
#include "../matrix.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class color_transform
    {
    public:
        color_transform(
            const double gamma_ = 1,
            const double red_scale_ = 1,
            const double green_scale_ = 1,
            const double blue_scale_ = 1
        ) : gamma(gamma_), red_scale(red_scale_), green_scale(green_scale_), blue_scale(blue_scale_)
        {
            DLIB_CASSERT(gamma_ >= 0)
            DLIB_CASSERT(0 <= red_scale_ && red_scale_ <= 1)
            DLIB_CASSERT(0 <= green_scale_ && green_scale_ <= 1)
            DLIB_CASSERT(0 <= blue_scale_ && blue_scale_ <= 1)
            const double m = 255 * std::max({red_scale_, green_scale_, blue_scale_});
            red_scale /= m;
            green_scale /= m;
            blue_scale /= m;
            // Now compute a lookup table for all the color channels.  The table tells us
            // what the transform does.
            table.resize(256 * 3);
            unsigned long i = 0;
            for (int k = 0; k < 256; ++k)
            {
                table[i++] = static_cast<unsigned char>(255 * std::pow(k * red_scale, gamma) + 0.5);
            }
            for (int k = 0; k < 256; ++k)
            {
                table[i++] = static_cast<unsigned char>(255 * std::pow(k * green_scale, gamma) + 0.5);
            }
            for (int k = 0; k < 256; ++k)
            {
                table[i++] = static_cast<unsigned char>(255 * std::pow(k * blue_scale, gamma) + 0.5);
            }
        }

        rgb_pixel operator()(rgb_pixel p) const
        {
            p.red = table[static_cast<unsigned int>(p.red)];
            p.green = table[static_cast<unsigned int>(p.green + 256)];
            p.blue = table[static_cast<unsigned int>(p.blue + 512)];
            return p;
        }

        double get_gamma() const { return gamma; }
        double get_red_scale() const { return red_scale; }
        double get_green_scale() const { return green_scale; }
        double get_blue_scale() const { return blue_scale; }

    private:
        std::vector<unsigned char> table;
        double gamma;
        double red_scale;
        double green_scale;
        double blue_scale;
    };

    class inv_color_transform
    {
    public:
        inv_color_transform(
            const color_transform& tform
        )
        {
            const auto gamma = tform.get_gamma();
            const auto red_scale = tform.get_red_scale();
            const auto green_scale = tform.get_green_scale();
            const auto blue_scale = tform.get_blue_scale();

            // Now compute a lookup table for all the color channels.  The table tells us
            // what the transform does.
            table.resize(256 * 3);
            unsigned long i = 0;
            for (int k = 0; k < 256; ++k)
            {
                table[i++] = static_cast<unsigned char>(std::pow(k / 255.0, 1 / gamma) / red_scale + 0.5);
            }
            for (int k = 0; k < 256; ++k)
            {
                table[i++] = static_cast<unsigned char>(std::pow(k / 255.0, 1 / gamma) / green_scale + 0.5);
            }
            for (int k = 0; k < 256; ++k)
            {
                table[i++] = static_cast<unsigned char>(std::pow(k / 255.0, 1 / gamma) / blue_scale + 0.5);
            }
        }

        rgb_pixel operator()(rgb_pixel p) const
        {
            p.red = table[static_cast<unsigned int>(p.red)];
            p.green = table[static_cast<unsigned int>(p.green + 256)];
            p.blue = table[static_cast<unsigned int>(p.blue + 512)];
            return p;
        }

    private:
        std::vector<unsigned char> table;
    };

// ----------------------------------------------------------------------------------------

    inline color_transform random_color_transform (
            dlib::rand& rnd,
            const double gamma_magnitude = 0.5,
            const double color_magnitude = 0.2
    )
    {
        // pick a random gamma correction factor.
        const double gamma = std::max(0.0, 1 + gamma_magnitude * (rnd.get_random_double() - 0.5));
        // pick a random color balancing scheme.
        const double red_scale = 1 - rnd.get_random_double() * color_magnitude;
        const double green_scale = 1 - rnd.get_random_double() * color_magnitude;
        const double blue_scale = 1 - rnd.get_random_double() * color_magnitude;
        return color_transform(gamma, red_scale, green_scale, blue_scale);
    }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    color_transform disturb_colors (
        image_type& img_,
        dlib::rand& rnd,
        const double gamma_magnitude = 0.5,
        const double color_magnitude = 0.2
    )
    {
        if (gamma_magnitude == 0 && color_magnitude == 0)
            return {};

        image_view<image_type> img(img_);
        const auto tform = random_color_transform(rnd, gamma_magnitude, color_magnitude);
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
        return tform;
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
