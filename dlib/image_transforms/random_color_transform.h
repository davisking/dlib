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

}

#endif // DLIB_RANDOM_cOLOR_TRANSFORM_Hh_

