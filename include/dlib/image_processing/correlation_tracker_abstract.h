// Copyright (C) 2015  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_CORRELATION_TrACKER_ABSTRACT_H_
#ifdef DLIB_CORRELATION_TrACKER_ABSTRACT_H_

#include "../geometry/drectangle_abstract.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class correlation_tracker
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a tool for tracking moving objects in a video stream.  You give it
                the bounding box of an object in the first frame and it attempts to track the
                object in the box from frame to frame.  

                This tool is an implementation of the method described in the following paper:
                    Danelljan, Martin, et al. "Accurate scale estimation for robust visual
                    tracking." Proceedings of the British Machine Vision Conference BMVC. 2014.
        !*/

    public:

        explicit correlation_tracker (unsigned long filter_size = 6, 
            unsigned long num_scale_levels = 5, 
            unsigned long scale_window_size = 23,
            double regularizer_space = 0.001,
            double nu_space = 0.025,
            double regularizer_scale = 0.001,
            double nu_scale = 0.025,
            double scale_pyramid_alpha = 1.020
        );
        /*!
            requires
                - p.is_empty() == false
            ensures
                - Initializes correlation_tracker. Higher value of filter_size and 
                  num_scale_levels increases tracking precision but requires more CPU 
                  for processing. Recommended values for filter_size = 5-7, 
                  default = 6, for num_scale_levels = 4-6, default = 5
                - #get_position().is_empty() == true
        !*/

        template <
            typename image_type
            >
        void start_track (
            const image_type& img,
            const drectangle& p
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - p.is_empty() == false
            ensures
                - This object will start tracking the thing inside the bounding box in the
                  given image.  That is, if you call update() with subsequent video frames 
                  then it will try to keep track of the position of the object inside p.
                - #get_position() == p
        !*/

        drectangle get_position (
        ) const;
        /*!
            ensures
                - returns the predicted position of the object under track.  
        !*/

        template <
            typename image_type
            >
        double update_noscale (
            const image_type& img,
            const drectangle& guess
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - get_position().is_empty() == false
                  (i.e. you must have started tracking by calling start_track())
            ensures
                - When searching for the object in img, we search in the area around the
                  provided guess. This function only tracks object position without trying
                  to track the scale
                - #get_position() == the new predicted location of the object in img.  This
                  location will be a copy of guess that has been translated and NOT scaled
                  appropriately based on the content of img so that it, hopefully, bounds
                  the object in img.
                - Returns the peak to side-lobe ratio.  This is a number that measures how
                  confident the tracker is that the object is inside #get_position().
                  Larger values indicate higher confidence.
        !*/

        template <
            typename image_type
            >
        double update (
            const image_type& img,
            const drectangle& guess
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - get_position().is_empty() == false
                  (i.e. you must have started tracking by calling start_track())
            ensures
                - When searching for the object in img, we search in the area around the
                  provided guess.
                - #get_position() == the new predicted location of the object in img.  This
                  location will be a copy of guess that has been translated and scaled
                  appropriately based on the content of img so that it, hopefully, bounds
                  the object in img.
                - Returns the peak to side-lobe ratio.  This is a number that measures how
                  confident the tracker is that the object is inside #get_position().
                  Larger values indicate higher confidence.
        !*/

        template <
            typename image_type
            >
        double update_noscale (
            const image_type& img
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - get_position().is_empty() == false
                  (i.e. you must have started tracking by calling start_track())
            ensures
                - performs: return update_noscale(img, get_position())
        !*/
        template <
            typename image_type
            >
        double update (
            const image_type& img
        );
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - get_position().is_empty() == false
                  (i.e. you must have started tracking by calling start_track())
            ensures
                - performs: return update(img, get_position())
        !*/

    };
}

#endif // DLIB_CORRELATION_TrACKER_ABSTRACT_H_



