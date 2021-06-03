// Copyright (C) 2016  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_RaNDOM_CROPPER_ABSTRACT_H_
#ifdef DLIB_RaNDOM_CROPPER_ABSTRACT_H_

#include "../threads.h"
#include <mutex>
#include <vector>
#include "interpolation.h"
#include "../image_processing/full_object_detection.h"
#include "../rand.h"

namespace dlib
{
    class random_cropper
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for extracting random crops of objects from a set of
                images.  The crops are randomly jittered in scale, translation, and
                rotation but more or less centered on objects specified by mmod_rect
                objects.
                
            THREAD SAFETY
                It is safe for multiple threads to make concurrent calls to this object's
                operator() methods.
        !*/

    public:

        random_cropper (
        );
        /*!
            ensures
                - #get_chip_dims() == chip_dims(300,300)
                - #get_randomly_flip() == true
                - #get_max_rotation_degrees() == 30
                - #get_min_object_length_long_dim() == 70 
                - #get_min_object_length_short_dim() == 30
                - #get_max_object_size() == 0.7
                - #get_background_crops_fraction() == 0.5
                - #get_translate_amount() == 0.1
        !*/

        void set_seed (
            time_t seed
        );
        /*!
            ensures
                - Seeds the internal random number generator with the given seed.
        !*/

        double get_translate_amount (
        ) const; 
        /*!
            ensures
                - When a box is cropped out, it will be randomly translated prior to
                  cropping by #get_translate_amount()*(the box's height) up or down and
                  #get_translate_amount()*(the box's width) left or right.
        !*/

        void set_translate_amount (
            double value
        );
        /*!
            requires
                - value >= 0
            ensures
                - #get_translate_amount() == value
        !*/

        double get_background_crops_fraction (
        ) const; 
        /*!
            ensures
                - When making random crops, get_background_crops_fraction() fraction of
                  them will be from random background rather than being centered on some
                  object in the dataset.
        !*/

        void set_background_crops_fraction (
            double value
        );
        /*!
            requires
                - 0 <= value <= 1
            ensures
                - #get_background_crops_fraction() == value
        !*/

        const chip_dims& get_chip_dims(
        ) const; 
        /*!
            ensures
                - returns the dimensions of image chips produced by this object.
        !*/

        void set_chip_dims (
            const chip_dims& dims
        );
        /*!
            ensures
                - #get_chip_dims() == dims
        !*/

        void set_chip_dims (
            unsigned long rows,
            unsigned long cols
        );
        /*!
            ensures
                - #get_chip_dims() == chip_dims(rows,cols)
        !*/

        bool get_randomly_flip (
        ) const;
        /*!
            ensures
                - if this object will randomly mirror chips left to right.
        !*/

        void set_randomly_flip (
            bool value
        );
        /*!
            ensures
                - #get_randomly_flip() == value
        !*/

        double get_max_rotation_degrees (
        ) const;
        /*!
            ensures
                - When extracting an image chip, this object will pick a random rotation
                  in the range [-get_max_rotation_degrees(), get_max_rotation_degrees()]
                  and rotate the chip by that amount.
        !*/

        void set_max_rotation_degrees (
            double value
        );
        /*!
            ensures
                - #get_max_rotation_degrees() == std::abs(value)
        !*/

        long get_min_object_length_long_dim (
        ) const; 
        /*!
            ensures
                - When a chip is extracted around an object, the chip will be sized so that
                  the longest edge of the object (i.e. either its height or width,
                  whichever is longer) is at least #get_min_object_length_long_dim() pixels
                  in length.  When we say "object" here we are referring specifically to
                  the rectangle in the mmod_rect output by the cropper.
        !*/

        long get_min_object_length_short_dim (
        ) const;
        /*!
            ensures
                - When a chip is extracted around an object, the chip will be sized so that
                  the shortest edge of the object (i.e. either its height or width,
                  whichever is shorter) is at least #get_min_object_length_short_dim()
                  pixels in length.  When we say "object" here we are referring
                  specifically to the rectangle in the mmod_rect output by the cropper.
        !*/

        void set_min_object_size (
            long long_dim,
            long short_dim
        ); 
        /*!
            requires
                - 0 < short_dim <= long_dim
            ensures
                - #get_min_object_length_short_dim() == short_dim
                - #get_min_object_length_long_dim() == long_dim
        !*/

        double get_max_object_size (
        ) const; 
        /*!
            ensures
                - When a chip is extracted around an object, the chip will be sized so that
                  both the object's height and width are at most get_max_object_size() *
                  the chip's height and width, respectively.  E.g. if the chip is 640x480
                  pixels in size then the object will be at most 480*get_max_object_size()
                  pixels tall and 640*get_max_object_size() pixels wide. 
        !*/

        void set_max_object_size (
            double value
        ); 
        /*!
            requires
                - 0 < value 
            ensures
                - #get_max_object_size() == value
        !*/

        template <
            typename array_type
            >
        void append (
            size_t num_crops,
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            array_type& crops,
            std::vector<std::vector<mmod_rect>>& crop_rects
        );
        /*!
            requires
                - images.size() == rects.size()
                - crops.size() == crop_rects.size()
                - for all valid i:
                    - images[i].size() != 0
                - array_type is a type with an interface compatible with dlib::array or
                  std::vector and it must in turn contain image objects that implement the
                  interface defined in dlib/image_processing/generic_image.h 
            ensures
                - Randomly extracts num_crops chips from images and appends them to the end
                  of crops.  We also copy the object metadata for each extracted crop and
                  store it into #crop_rects.  In particular, calling this function is the
                  same as making multiple calls to the version of operator() below that
                  outputs a single crop, except that append() will use multiple CPU cores
                  to do the processing and is therefore faster.
                - #crops.size() == crops.size()+num_crops
                - #crop_rects.size() == crop_rects.size()+num_crops
        !*/

        template <
            typename array_type
            >
        void operator() (
            size_t num_crops,
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            array_type& crops,
            std::vector<std::vector<mmod_rect>>& crop_rects
        );
        /*!
            requires
                - images.size() == rects.size()
                - for all valid i:
                    - images[i].size() != 0
                - array_type is a type with an interface compatible with dlib::array or
                  std::vector and it must in turn contain image objects that implement the
                  interface defined in dlib/image_processing/generic_image.h 
            ensures
                - Randomly extracts num_crops chips from images.  We also copy the object
                  metadata for each extracted crop and store it into #crop_rects.  In
                  particular, calling this function is the same as invoking the version of
                  operator() below multiple times, except that this version of operator()
                  will use multiple CPU cores to do the processing and is therefore faster.
                - #crops.size() == num_crops
                - #crop_rects.size() == num_crops
        !*/

        template <
            typename array_type,
            typename image_type
            >
        void operator() (
            const array_type& images,
            const std::vector<std::vector<mmod_rect>>& rects,
            image_type& crop,
            std::vector<mmod_rect>& crop_rects
        );
        /*!
            requires
                - images.size() == rects.size()
                - for all valid i:
                    - images[i].size() != 0
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - array_type is a type with an interface compatible with dlib::array or
                  std::vector and it must in turn contain image objects that implement the
                  interface defined in dlib/image_processing/generic_image.h 
            ensures
                - Selects a random image and creates a random crop from it.  Specifically,
                  we pick a random index IDX < images.size() and then execute 
                    (*this)(images[IDX],rects[IDX],crop,crop_rects) 
        !*/

        template <
            typename image_type1,
            typename image_type2
            >
        void operator() (
            const image_type1& img,
            const std::vector<mmod_rect>& rects,
            image_type2& crop,
            std::vector<mmod_rect>& crop_rects
        );
        /*!
            requires
                - img.size() != 0
                - image_type1 == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - image_type2 == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
            ensures
                - Extracts a random crop from img and copies over the mmod_rect objects in
                  rects to #crop_rects if they are contained inside the crop.  Moreover,
                  rectangles are marked as ignore if they aren't completely contained
                  inside the crop.
                - #crop_rects.size() <= rects.size()
        !*/

        template <
            typename image_type1
            >
        image_type1 operator() (
            const image_type1& img
        );
        /*!
            requires
                - img.size() != 0
                - image_type1 == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
            ensures
                - This function simply calls (*this)(img, junk1, crop, junk2) and returns
                  crop.  Therefore it is simply a convenience function for extracting a
                  random background patch.
        !*/
    };

// ----------------------------------------------------------------------------------------

    std::ostream& operator<< (
        std::ostream& out,
        const random_cropper& item
    );
    /*!
        ensures
            - Prints the state of all the parameters of item to out.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_RaNDOM_CROPPER_ABSTRACT_H_


