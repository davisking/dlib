// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RANDOM_cOLOR_TRANSFORM_ABSTRACT_Hh_
#ifdef DLIB_RANDOM_cOLOR_TRANSFORM_ABSTRACT_Hh_

#include "../image_processing/generic_image.h"
#include "../pixel.h"
#include "../rand.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class random_color_transform
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object generates a random color balancing and gamma correction
                transform.  It then allows you to apply that specific transform to as many
                rgb_pixel objects as you like.
        !*/

    public:

        random_color_transform (
            dlib::rand& rnd,
            const double gamma_magnitude = 0.5,
            const double color_magnitude = 0.2
        );
        /*!
            requires
                - 0 <= gamma_magnitude 
                - 0 <= color_magnitude <= 1
            ensures
                - This constructor generates a random color transform which can be applied
                  by calling this object's operator() method.
                - The color transform is a gamma correction and color rebalancing.  If
                  gamma_magnitude == 0 and color_magnitude == 0 then the transform doesn't
                  change any colors at all.  However, the larger these parameters the more
                  noticeable the resulting transform.
        !*/

        rgb_pixel operator()(
            rgb_pixel p
        ) const;
        /*!
            ensures
                - returns the color transformed version of p. 
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    void disturb_colors (
        image_type& img,
        dlib::rand& rnd,
        const double gamma_magnitude = 0.5,
        const double color_magnitude = 0.2
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - Applies a random color transform to the given image.  This is done by
              creating a random_color_transform with the given parameters and then
              transforming each pixel in the image with the resulting transform.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    void apply_random_color_offset (
        image_type& img,
        dlib::rand& rnd
    );
    /*!
        ensures
            - Picks a random color offset vector and adds it to the given image.  The offset
              vector is selected using the method described in the paper:
                Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet
                classification with deep convolutional neural networks." Advances in neural
                information processing systems. 2012.
              In particular, we sample an RGB value from the typical distribution of RGB
              values, assuming it has a Gaussian distribution, and then divide it by 10.
              This sampled RGB vector is added to each pixel of img.
    !*/

// ----------------------------------------------------------------------------------------

#endif // DLIB_RANDOM_cOLOR_TRANSFORM_ABSTRACT_Hh_

