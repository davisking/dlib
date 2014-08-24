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
            WHAT THIS OBJECT REPRESENTS
                This object is a tool that takes in an image region containing some object
                and outputs a "shape" or set of point locations that define the pose of the
                object.  The classic example of this is human face pose prediction, where
                you take an image of a human face as input and are expected to identify the
                locations of important facial landmarks such as the corners of the mouth
                and eyes, tip of the nose, and so forth.

                To create useful instantiations of this object you need to use the
                shape_predictor_trainer object defined below to train a shape_predictor
                using a set of training images, each annotated with shapes you want to
                predict.
        !*/

    public:

        shape_predictor (
        );
        /*!
            ensures
                - #num_parts() == 0
        !*/

        unsigned long num_parts (
        ) const;
        /*!
            ensures
                - returns the number of parts in the shapes predicted by this object.
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
                - Runs the shape prediction algorithm on the part of the image contained in
                  the given bounding rectangle.  So it will try and fit the shape model to
                  the contents of the given rectangle in the image.  For example, if there
                  is a human face inside the rectangle and you use a face landmarking shape
                  model then this function will return the locations of the face landmarks
                  as the parts.  So the return value is a full_object_detection DET such
                  that:
                    - DET.get_rect() == rect
                    - DET.num_parts() == num_parts()
                    - for all valid i:
                        - DET.part(i) == the location in img for the i-th part of the shape
                          predicted by this object.
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
            WHAT THIS OBJECT REPRESENTS
                This object is a tool for training shape_predictors based on annotated training
                images.  Its implementation uses the algorithm described in:
                    One Millisecond Face Alignment with an Ensemble of Regression Trees
                    by Vahid Kazemi and Josephine Sullivan, CVPR 2014
        !*/

    public:

        shape_predictor_trainer (
        )
        {
            _cascade_depth = 10;
            _tree_depth = 2;
            _num_trees_per_cascade_level = 500;
            _nu = 0.1;
            _oversampling_amount = 20;
            _feature_pool_size = 400;
            _lambda = 0.1;
            _num_test_splits = 20;
            _feature_pool_region_padding = 0;
            _verbose = false;
        }

        unsigned long get_cascade_depth (
        ) const;
        /*!
        !*/

        void set_cascade_depth (
            unsigned long depth
        );
        /*!
            requires
                - depth > 0
            ensures
                - #get_cascade_depth() == depth
        !*/

        unsigned long get_tree_depth (
        ) const; 
        /*!
        !*/

        void set_tree_depth (
            unsigned long depth
        );
        /*!
            requires
                - depth > 0
            ensures
                - #get_tree_depth() == depth
        !*/

        unsigned long get_num_trees_per_cascade_level (
        ) const;
        /*!
        !*/

        void set_num_trees_per_cascade_level (
            unsigned long num
        );
        /*!
            requires
                - num > 0
            ensures
                - #get_num_trees_per_cascade_level() == num
        !*/

        double get_nu (
        ) const; 
        /*!
        !*/

        void set_nu (
            double nu
        );
        /*!
            requires
                - nu > 0
            ensures
                - #get_nu() == nu
        !*/

        std::string get_random_seed (
        ) const;
        /*!
        !*/

        void set_random_seed (
            const std::string& seed
        );
        /*!
            ensures
                - #get_random_seed() == seed
        !*/

        unsigned long get_oversampling_amount (
        ) const;
        /*!
        !*/

        void set_oversampling_amount (
            unsigned long amount
        );
        /*!
            requires
                - amount > 0
            ensures
                - #get_oversampling_amount() == amount
        !*/

        unsigned long get_feature_pool_size (
        ) const;
        /*!
        !*/

        void set_feature_pool_size (
            unsigned long size
        );
        /*!
            requires
                - size > 1
            ensures
                - #get_feature_pool_size() == size
        !*/

        double get_lambda (
        ) const;
        /*!
        !*/

        void set_lambda (
            double lambda
        );
        /*!
            requires
                - lambda > 0
            ensures
                - #get_lambda() == lambda
        !*/

        unsigned long get_num_test_splits (
        ) const;
        /*!
        !*/

        void set_num_test_splits (
            unsigned long num
        );
        /*!
            requires
                - num > 0
            ensures
                - #get_num_test_splits() == num
        !*/

        double get_feature_pool_region_padding (
        ) const; 
        /*!
        !*/

        void set_feature_pool_region_padding (
            double padding 
        );
        /*!
            ensures
                - #get_feature_pool_region_padding() == padding
        !*/

        void be_verbose (
        );
        /*!
            ensures
                - This object will print status messages to standard out so that a 
                  user can observe the progress of the algorithm.
        !*/

        void be_quiet (
        );
        /*!
            ensures
                - this object will not print anything to standard out
        !*/

        template <typename image_array>
        shape_predictor train (
            const image_array& images,
            const std::vector<std::vector<full_object_detection> >& objects
        ) const;
        /*!
            requires
                - images.size() == objects.size()
                - images.size() > 0
            ensures
                - This object will try to learn to predict the locations of an object's parts 
                  based on the object bounding box (i.e.  full_object_detection::get_rect()) 
                  and the image pixels in that box.  That is, we will try to learn a
                  shape_predictor, SP, such that:
                    SP(images[i], objects[i][j].get_rect()) == objects[i][j]
                  This learned SP object is then returned.
        !*/
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
              then return the average distance (measured in pixels) between a predicted
              part location and its true position.  
            - if (scales.size() != 0) then
                - Each time we compute the distance between a predicted part location and
                  its true location in objects[i][j] we divide the distance by
                  scales[i][j].  Therefore, if you want the reported error to be the
                  average pixel distance then give an empty scales vector, but if you want
                  the returned value to be something else like the average distance
                  normalized by some feature of each object (e.g. the interocular distance)
                  then you can supply those normalizing values via scales.
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

