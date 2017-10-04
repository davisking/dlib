// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_H_
#define DLIB_SHAPE_PREDICToR_H_

#include "shape_predictor_abstract.h"
#include "full_object_detection.h"
#include "../algs.h"
#include "../matrix.h"
#include "../geometry.h"
#include "../pixel.h"
#include "../statistics.h"
#include <utility>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        struct split_feature
        {
            unsigned long idx1;
            unsigned long idx2;
            float thresh;

            friend inline void serialize (const split_feature& item, std::ostream& out)
            {
                dlib::serialize(item.idx1, out);
                dlib::serialize(item.idx2, out);
                dlib::serialize(item.thresh, out);
            }
            friend inline void deserialize (split_feature& item, std::istream& in)
            {
                dlib::deserialize(item.idx1, in);
                dlib::deserialize(item.idx2, in);
                dlib::deserialize(item.thresh, in);
            }
        };


        // a tree is just a std::vector<impl::split_feature>.  We use this function to navigate the
        // tree nodes
        inline unsigned long left_child (unsigned long idx) { return 2*idx + 1; }
        /*!
            ensures
                - returns the index of the left child of the binary tree node idx
        !*/
        inline unsigned long right_child (unsigned long idx) { return 2*idx + 2; }
        /*!
            ensures
                - returns the index of the left child of the binary tree node idx
        !*/

        struct regression_tree
        {
            std::vector<split_feature> splits;
            std::vector<matrix<float,0,1> > leaf_values;

            unsigned long num_leaves() const { return leaf_values.size(); }

            inline const matrix<float,0,1>& operator()(
                const std::vector<float>& feature_pixel_values,
                unsigned long& i
            ) const
            /*!
                requires
                    - All the index values in splits are less than feature_pixel_values.size()
                    - leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - leaf_values.size() == splits.size()+1
                      (i.e. there needs to be the right number of leaves given the number of splits in the tree)
                ensures
                    - runs through the tree and returns the vector at the leaf we end up in.
                    - #i == the selected leaf node index.
            !*/
            {
                i = 0;
                while (i < splits.size())
                {
                    if ((float)feature_pixel_values[splits[i].idx1] - (float)feature_pixel_values[splits[i].idx2] > splits[i].thresh)
                        i = left_child(i);
                    else
                        i = right_child(i);
                }
                i = i - splits.size();
                return leaf_values[i];
            }

            friend void serialize (const regression_tree& item, std::ostream& out)
            {
                dlib::serialize(item.splits, out);
                dlib::serialize(item.leaf_values, out);
            }
            friend void deserialize (regression_tree& item, std::istream& in)
            {
                dlib::deserialize(item.splits, in);
                dlib::deserialize(item.leaf_values, in);
            }
        };

    // ------------------------------------------------------------------------------------

        inline vector<float,2> location (
            const matrix<float,0,1>& shape,
            unsigned long idx
        )
        /*!
            requires
                - idx < shape.size()/2
                - shape.size()%2 == 0
            ensures
                - returns the idx-th point from the shape vector.
        !*/
        {
            return vector<float,2>(shape(idx*2), shape(idx*2+1));
        }

    // ------------------------------------------------------------------------------------

        inline unsigned long nearest_shape_point (
            const matrix<float,0,1>& shape,
            const dlib::vector<float,2>& pt
        )
        {
            // find the nearest part of the shape to this pixel
            float best_dist = std::numeric_limits<float>::infinity();
            const unsigned long num_shape_parts = shape.size()/2;
            unsigned long best_idx = 0;
            for (unsigned long j = 0; j < num_shape_parts; ++j)
            {
                const float dist = length_squared(location(shape,j)-pt);
                if (dist < best_dist)
                {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            return best_idx;
        }

    // ------------------------------------------------------------------------------------

        inline void create_shape_relative_encoding (
            const matrix<float,0,1>& shape,
            const std::vector<dlib::vector<float,2> >& pixel_coordinates,
            std::vector<unsigned long>& anchor_idx, 
            std::vector<dlib::vector<float,2> >& deltas
        )
        /*!
            requires
                - shape.size()%2 == 0 
                - shape.size() > 0
            ensures
                - #anchor_idx.size() == pixel_coordinates.size()
                - #deltas.size()     == pixel_coordinates.size()
                - for all valid i:
                    - pixel_coordinates[i] == location(shape,#anchor_idx[i]) + #deltas[i]
        !*/
        {
            anchor_idx.resize(pixel_coordinates.size());
            deltas.resize(pixel_coordinates.size());


            for (unsigned long i = 0; i < pixel_coordinates.size(); ++i)
            {
                anchor_idx[i] = nearest_shape_point(shape, pixel_coordinates[i]);
                deltas[i] = pixel_coordinates[i] - location(shape,anchor_idx[i]);
            }
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine find_tform_between_shapes (
            const matrix<float,0,1>& from_shape,
            const matrix<float,0,1>& to_shape
        )
        {
            DLIB_ASSERT(from_shape.size() == to_shape.size() && (from_shape.size()%2) == 0 && from_shape.size() > 0,"");
            std::vector<vector<float,2> > from_points, to_points;
            const unsigned long num = from_shape.size()/2;
            from_points.reserve(num);
            to_points.reserve(num);
            if (num == 1)
            {
                // Just use an identity transform if there is only one landmark.
                return point_transform_affine();
            }

            for (unsigned long i = 0; i < num; ++i)
            {
                from_points.push_back(location(from_shape,i));
                to_points.push_back(location(to_shape,i));
            }
            return find_similarity_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine normalizing_tform (
            const rectangle& rect
        )
        /*!
            ensures
                - returns a transform that maps rect.tl_corner() to (0,0) and rect.br_corner()
                  to (1,1).
        !*/
        {
            std::vector<vector<float,2> > from_points, to_points;
            from_points.push_back(rect.tl_corner()); to_points.push_back(point(0,0));
            from_points.push_back(rect.tr_corner()); to_points.push_back(point(1,0));
            from_points.push_back(rect.br_corner()); to_points.push_back(point(1,1));
            return find_affine_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        inline point_transform_affine unnormalizing_tform (
            const rectangle& rect
        )
        /*!
            ensures
                - returns a transform that maps (0,0) to rect.tl_corner() and (1,1) to
                  rect.br_corner().
        !*/
        {
            std::vector<vector<float,2> > from_points, to_points;
            to_points.push_back(rect.tl_corner()); from_points.push_back(point(0,0));
            to_points.push_back(rect.tr_corner()); from_points.push_back(point(1,0));
            to_points.push_back(rect.br_corner()); from_points.push_back(point(1,1));
            return find_affine_transform(from_points, to_points);
        }

    // ------------------------------------------------------------------------------------

        template <typename image_type, typename feature_type>
        void extract_feature_pixel_values (
            const image_type& img_,
            const rectangle& rect,
            const matrix<float,0,1>& current_shape,
            const matrix<float,0,1>& reference_shape,
            const std::vector<unsigned long>& reference_pixel_anchor_idx,
            const std::vector<dlib::vector<float,2> >& reference_pixel_deltas,
            std::vector<feature_type>& feature_pixel_values
        )
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - reference_pixel_anchor_idx.size() == reference_pixel_deltas.size()
                - current_shape.size() == reference_shape.size()
                - reference_shape.size()%2 == 0
                - max(mat(reference_pixel_anchor_idx)) < reference_shape.size()/2
            ensures
                - #feature_pixel_values.size() == reference_pixel_deltas.size()
                - for all valid i:
                    - #feature_pixel_values[i] == the value of the pixel in img_ that
                      corresponds to the pixel identified by reference_pixel_anchor_idx[i]
                      and reference_pixel_deltas[i] when the pixel is located relative to
                      current_shape rather than reference_shape.
        !*/
        {
            const matrix<float,2,2> tform = matrix_cast<float>(find_tform_between_shapes(reference_shape, current_shape).get_m());
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            const rectangle area = get_rect(img_);

            const_image_view<image_type> img(img_);
            feature_pixel_values.resize(reference_pixel_deltas.size());
            for (unsigned long i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                point p = tform_to_img(tform*reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));
                if (area.contains(p))
                    feature_pixel_values[i] = get_pixel_intensity(img[p.y()][p.x()]);
                else
                    feature_pixel_values[i] = 0;
            }
        }

    } // end namespace impl

// ----------------------------------------------------------------------------------------

    class shape_predictor
    {
    public:


        shape_predictor (
        ) 
        {}

        shape_predictor (
            const matrix<float,0,1>& initial_shape_,
            const std::vector<std::vector<impl::regression_tree> >& forests_,
            const std::vector<std::vector<dlib::vector<float,2> > >& pixel_coordinates
        ) : initial_shape(initial_shape_), forests(forests_)
        /*!
            requires
                - initial_shape.size()%2 == 0
                - forests.size() == pixel_coordinates.size() == the number of cascades
                - for all valid i:
                    - all the index values in forests[i] are less than pixel_coordinates[i].size()
                - for all valid i and j: 
                    - forests[i][j].leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - forests[i][j].leaf_values.size() == forests[i][j].splits.size()+1
                      (i.e. there need to be the right number of leaves given the number of splits in the tree)
        !*/
        {
            anchor_idx.resize(pixel_coordinates.size());
            deltas.resize(pixel_coordinates.size());
            // Each cascade uses a different set of pixels for its features.  We compute
            // their representations relative to the initial shape now and save it.
            for (unsigned long i = 0; i < pixel_coordinates.size(); ++i)
                impl::create_shape_relative_encoding(initial_shape, pixel_coordinates[i], anchor_idx[i], deltas[i]);
        }

        unsigned long num_parts (
        ) const
        {
            return initial_shape.size()/2;
        }

        unsigned long num_features (
        ) const
        {
            unsigned long num = 0;
            for (unsigned long iter = 0; iter < forests.size(); ++iter)
                for (unsigned long i = 0; i < forests[iter].size(); ++i)
                    num += forests[iter][i].num_leaves();
            return num;
        }

        template <typename image_type>
        full_object_detection operator()(
            const image_type& img,
            const rectangle& rect
        ) const
        {
            using namespace impl;
            matrix<float,0,1> current_shape = initial_shape;
            std::vector<float> feature_pixel_values;
            for (unsigned long iter = 0; iter < forests.size(); ++iter)
            {
                extract_feature_pixel_values(img, rect, current_shape, initial_shape,
                                             anchor_idx[iter], deltas[iter], feature_pixel_values);
                unsigned long leaf_idx;
                // evaluate all the trees at this level of the cascade.
                for (unsigned long i = 0; i < forests[iter].size(); ++i)
                    current_shape += forests[iter][i](feature_pixel_values, leaf_idx);
            }

            // convert the current_shape into a full_object_detection
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);
            std::vector<point> parts(current_shape.size()/2);
            for (unsigned long i = 0; i < parts.size(); ++i)
                parts[i] = tform_to_img(location(current_shape, i));
            return full_object_detection(rect, parts);
        }

        template <typename image_type, typename T, typename U>
        full_object_detection operator()(
            const image_type& img,
            const rectangle& rect,
            std::vector<std::pair<T,U> >& feats
        ) const
        {
            feats.clear();
            using namespace impl;
            matrix<float,0,1> current_shape = initial_shape;
            std::vector<float> feature_pixel_values;
            unsigned long feat_offset = 0;
            for (unsigned long iter = 0; iter < forests.size(); ++iter)
            {
                extract_feature_pixel_values(img, rect, current_shape, initial_shape,
                                             anchor_idx[iter], deltas[iter], feature_pixel_values);
                // evaluate all the trees at this level of the cascade.
                for (unsigned long i = 0; i < forests[iter].size(); ++i)
                {
                    unsigned long leaf_idx;
                    current_shape += forests[iter][i](feature_pixel_values, leaf_idx);

                    feats.push_back(std::make_pair(feat_offset+leaf_idx, 1));
                    feat_offset += forests[iter][i].num_leaves();
                }
            }

            // convert the current_shape into a full_object_detection
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);
            std::vector<point> parts(current_shape.size()/2);
            for (unsigned long i = 0; i < parts.size(); ++i)
                parts[i] = tform_to_img(location(current_shape, i));
            return full_object_detection(rect, parts);
        }

        friend void serialize (const shape_predictor& item, std::ostream& out);

        friend void deserialize (shape_predictor& item, std::istream& in);

    private:
        matrix<float,0,1> initial_shape;
        std::vector<std::vector<impl::regression_tree> > forests;
        std::vector<std::vector<unsigned long> > anchor_idx; 
        std::vector<std::vector<dlib::vector<float,2> > > deltas;
    };

    inline void serialize (const shape_predictor& item, std::ostream& out)
    {
        int version = 1;
        dlib::serialize(version, out);
        dlib::serialize(item.initial_shape, out);
        dlib::serialize(item.forests, out);
        dlib::serialize(item.anchor_idx, out);
        dlib::serialize(item.deltas, out);
    }

    inline void deserialize (shape_predictor& item, std::istream& in)
    {
        int version = 0;
        dlib::deserialize(version, in);
        if (version != 1)
            throw serialization_error("Unexpected version found while deserializing dlib::shape_predictor.");
        dlib::deserialize(item.initial_shape, in);
        dlib::deserialize(item.forests, in);
        dlib::deserialize(item.anchor_idx, in);
        dlib::deserialize(item.deltas, in);
    }

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
    )
    {
        // make sure requires clause is not broken
#ifdef ENABLE_ASSERTS
        DLIB_CASSERT( images.size() == objects.size() ,
            "\t double test_shape_predictor()"
            << "\n\t Invalid inputs were given to this function. "
            << "\n\t images.size():  " << images.size() 
            << "\n\t objects.size(): " << objects.size() 
        );
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                DLIB_CASSERT(objects[i][j].num_parts() == sp.num_parts(), 
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                    << "\n\t sp.num_parts(): " << sp.num_parts()
                );
            }
            if (scales.size() != 0)
            {
                DLIB_CASSERT(objects[i].size() == scales[i].size(), 
                    "\t double test_shape_predictor()"
                    << "\n\t Invalid inputs were given to this function. "
                    << "\n\t objects["<<i<<"].size(): " << objects[i].size()
                    << "\n\t scales["<<i<<"].size(): " << scales[i].size()
                );

            }
        }
#endif

        running_stats<double> rs;
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                // Just use a scale of 1 (i.e. no scale at all) if the caller didn't supply
                // any scales.
                const double scale = scales.size()==0 ? 1 : scales[i][j]; 

                full_object_detection det = sp(images[i], objects[i][j].get_rect());

                for (unsigned long k = 0; k < det.num_parts(); ++k)
                {
                    if (objects[i][j].part(k) != OBJECT_PART_NOT_PRESENT)
                    {
                        double score = length(det.part(k) - objects[i][j].part(k))/scale;
                        rs.add(score);
                    }
                }
            }
        }
        return rs.mean();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename image_array
        >
    double test_shape_predictor (
        const shape_predictor& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects
    )
    {
        std::vector<std::vector<double> > no_scales;
        return test_shape_predictor(sp, images, objects, no_scales);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_H_

