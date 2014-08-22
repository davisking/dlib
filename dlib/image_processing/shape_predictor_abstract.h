// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_SHAPE_PREDICToR_ABSTRACT_H_
#ifdef DLIB_SHAPE_PREDICToR_ABSTRACT_H_

#include "full_object_detection_abstract.h"
#include "../matrix.h"
#include "../geometry.h"
#include "../pixel.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    class shape_predictor
    {
        /*!
        !*/

    public:

        shape_predictor (
        );
        /*!
        !*/

        unsigned long num_parts (
        ) const;
        /*!
            ensures
                - returns the number of points in the shape
        !*/

        template <typename image_type>
        full_object_detection operator()(
            const image_type& img,
            const rectangle& rect
        ) const;
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
            ensures
                - runs the tree regressor on the detection rect inside img and returns a 
                  full_object_detection DET such that:
                    - DET.get_rect() == rect
                    - DET.num_parts() == num_parts()
        !*/

    };

    void serialize (const shape_predictor& item, std::ostream& out);
    void deserialize (shape_predictor& item, std::istream& in);
    /*!
        provides serialization support
    !*/

// ----------------------------------------------------------------------------------------

    class shape_predictor_trainer
    {
        /*!
            This thing really only works with unsigned char or rgb_pixel images (since we assume the threshold 
            should be in the range [-128,128]).
        !*/

    public:

        unsigned long cascade_depth (
        ) const { return 10; }

        unsigned long tree_depth (
        ) const { return 2; }

        unsigned long num_trees_per_cascade_level (
        ) const { return 500; }

        double get_nu (
        ) const { return 0.1; } // the regularizer 

        std::string random_seed (
        ) const { return "dlib rules"; }

        unsigned long oversampling_amount (
        ) const { return 20; }

        // feature sampling parameters
        unsigned long feature_pool_size (
        ) const { return 400; }// this must be > 1
        double get_lambda (
        ) const { return 0.1; }
        unsigned long get_num_test_splits (
        ) const { return 20; }
        double get_feature_pool_region_padding (
        ) const { return 0; }

        template <typename image_array>
        shape_predictor train (
            const image_array& images,
            const std::vector<std::vector<full_object_detection> >& objects
        ) const;

    };

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename image_array
        >
    double test_shape_predictor (
        const shape_predictor& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects,
        const std::vector<std::vector<double> >& scales
    );
    /*!
        requires
            - images.size() == objects.size()
            - for all valid i and j:
                - objects[i][j].num_parts() == sp.num_parts()
            - if (scales.size() != 0) then
                - There must be a scale value for each full_object_detection in objects.
                  That is, it must be the case that:
                    - scales.size() == objects.size()
                    - for all valid i:
                        - scales[i].size() == objects[i].size()
        ensures
            - Tests the given shape_predictor by running it on each of the given objects and
              checking how well it recovers the part positions.  In particular, for all 
              valid i and j we perform:
                sp(images[i], objects[i][j].get_rect())
              and compare the result with the truth part positions in objects[i][j].  We 
              then return the average distance between a predicted part location and its
              true position.  This value is then returned. 
            - if (scales.size() != 0) then
                - Each time we compute the distance between a predicted part location and
                  its true location in objects[i][j] we divide the distance by
                  scales[i][j].  Therefore, if you want the reported error to be the
                  average pixel distance then give an empty scales vector, but if you want
                  the returned value to be something else like the average distance
                  normalized by some feature of the objects (e.g. the interocular distance)
                  then you an supply those normalizing values via scales.
    !*/

    template <
        typename image_array
        >
    double test_shape_predictor (
        const shape_predictor& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects
    );
    /*!
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_ABSTRACT_H_

