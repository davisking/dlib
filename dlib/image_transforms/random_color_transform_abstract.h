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

    class color_transform
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object generates a color balancing and gamma correction transform.
                It then allows you to apply that specific transform to as many
                rgb_pixel objects as you like.
        !*/

    public:

        color_transform (
            const double gamma = 1.0,
            const double red_scale = 1.0,
            const double green_scale = 1.0,
            const double blue_scale = 1.0
        );
        /*!
            requires
                - 0 <= gamma
                - 0 <= red_scale <= 1
                - 0 <= green_scale <= 1
                - 0 <= blue_scale <= 1
            ensures
                - This constructor generates a color transform which can be applied by
                  calling this object's operator() method.
                - The color transform is a gamma correction and color rebalancing.  If
                  gamma == 1, red_scale == 1, green_scale == 1 and blue_scale == 1 then
                  the transform doesn't change any colors at all.  However, the farther
                  away from 1 these parameters are, the more noticeable the resulting
                  transform.
        !*/

        rgb_pixel operator()(
            rgb_pixel p
        ) const;
        /*!
            ensures
                - returns the color transformed version of p.
        !*/

        double get_gamma() const;
        /*!
            ensures
                - returns the gamma used in this color transform.
        !*/

        double get_red_scale() const;
        /*!
            ensures
                - returns the red scale used in this color transform.
        !*/

        double get_green_scale() const;
        /*!
            ensures
                - returns the green scale used in this color transform.
        !*/

        double get_blue_scale() const;
        /*!
            ensures
                - returns the blue scale used in this color transform.
        !*/
    };

// ----------------------------------------------------------------------------------------

    class inv_color_transform
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object generates a color balancing and gamma correction transform.
                It then allows you to apply that specific transform to as many
                rgb_pixel objects as you like. In particular, it generates the inverse
                transform of the one passed as an argument to the constructor.
        !*/

    public:

        inv_color_transform (
            const color_transform& tform
        );
        /*!
            ensures
                - This constructor generates a color transform which can be applied by
                  calling this object's operator() method.
                - The resulting transform is the inverse of tform, which can be used to
                  undo the effect of tform.
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

    inline color_transform random_color_transform (
            dlib::rand& rnd,
            const double gamma_magnitude = 0.5,
            const double color_magnitude = 0.2
    );
    /*!
        ensures
            - returns a random color balancing and gamma corection transform.  It then
              allows you to apply that specific transform to as many rgb_pixel objects as
              you like.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    color_transform disturb_colors (
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
            - Returns the color transform used to transform the given image.
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
