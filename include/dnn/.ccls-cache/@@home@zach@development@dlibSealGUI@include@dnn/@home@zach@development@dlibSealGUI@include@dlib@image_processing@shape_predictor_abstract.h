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
                and outputs a set of point locations that define the pose of the object.
                The classic example of this is human face pose prediction, where you take
                an image of a human face as input and are expected to identify the
                locations of important facial landmarks such as the corners of the mouth
                and eyes, tip of the nose, and so forth.

                To create useful instantiations of this object you need to use the
                shape_predictor_trainer object defined in the
                shape_predictor_trainer_abstract.h file to train a shape_predictor using a
                set of training images, each annotated with shapes you want to predict.

            THREAD SAFETY
                No synchronization is required when using this object.  In particular, a
                single instance of this object can be used from multiple threads at the
                same time.  
        !*/

    public:

        shape_predictor (
        );
        /*!
            ensures
                - #num_parts() == 0
                - #num_features() == 0
        !*/

        unsigned long num_parts (
        ) const;
        /*!
            ensures
                - returns the number of parts in the shapes predicted by this object.
        !*/

        unsigned long num_features (
        ) const;
        /*!
            ensures
                - Returns the dimensionality of the feature vector output by operator().
                  This number is the total number of trees in this object times the number
                  of leaves on each tree.  
        !*/

        template <typename image_type, typename T, typename U>
        full_object_detection operator()(
            const image_type& img,
            const rectangle& rect,
            std::vector<std::pair<T,U> >& feats
        ) const;
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - T is some unsigned integral type (e.g. unsigned int).
                - U is any scalar type capable of storing the value 1 (e.g. float).
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
                - #feats == a sparse vector that records which leaf each tree used to make
                  the shape prediction.   Moreover, it is an indicator vector, Therefore,
                  for all valid i:
                    - #feats[i].second == 1
                  Further, #feats is a vector from the space of num_features() dimensional
                  vectors.  The output shape positions can be represented as the dot
                  product between #feats and a weight vector.  Therefore, #feats encodes
                  all the information from img that was used to predict the returned shape
                  object.
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
                - Calling this function is equivalent to calling (*this)(img, rect, ignored)
                  where the 3d argument is discarded.
        !*/

    };

    void serialize (const shape_predictor& item, std::ostream& out);
    void deserialize (shape_predictor& item, std::istream& in);
    /*!
        provides serialization support
    !*/

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
            - image_array is a dlib::array of image objects where each image object
              implements the interface defined in dlib/image_processing/generic_image.h 
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
            - Note that any parts in objects that are set to OBJECT_PART_NOT_PRESENT are
              simply ignored.
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
        requires
            - image_array is a dlib::array of image objects where each image object
              implements the interface defined in dlib/image_processing/generic_image.h 
            - images.size() == objects.size()
            - for all valid i and j:
                - objects[i][j].num_parts() == sp.num_parts()
        ensures
            - returns test_shape_predictor(sp, images, objects, no_scales) where no_scales
              is an empty vector.  So this is just a convenience function for calling the
              above test_shape_predictor() routine without a scales argument.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_ABSTRACT_H_

