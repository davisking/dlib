// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SCAN_IMaGE_PYRAMID_H__
#define DLIB_SCAN_IMaGE_PYRAMID_H__

#include "scan_image_pyramid_abstract.h"
#include "../matrix.h"
#include "../geometry.h"
#include "../image_processing.h"
#include "../array2d.h"
#include <vector>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    class scan_image_pyramid : noncopyable
    {

    public:

        typedef matrix<double,0,1> feature_vector_type;

        typedef Pyramid_type pyramid_type;
        typedef Feature_extractor_type feature_extractor_type;

        scan_image_pyramid (
        );  

        template <
            typename image_type
            >
        void load (
            const image_type& img
        );

        inline bool is_loaded_with_image (
        ) const;

        inline void copy_configuration(
            const feature_extractor_type& fe
        );

        inline void copy_configuration (
            const scan_image_pyramid& item
        );

        void add_detection_template (
            const rectangle& object_box,
            const std::vector<rectangle>& feature_extraction_regions 
        );

        inline unsigned long get_num_detection_templates (
        ) const;

        inline unsigned long get_num_components_per_detection_template (
        ) const;

        inline long get_num_dimensions (
        ) const;

        unsigned long get_max_pyramid_levels (
        ) const;

        void set_max_pyramid_levels (
            unsigned long max_levels
        );

        inline unsigned long get_max_detections_per_template (
        ) const;

        void set_max_detections_per_template (
            unsigned long max_dets
        );

        void detect (
            const feature_vector_type& w,
            std::vector<std::pair<double, rectangle> >& dets,
            const double thresh
        ) const;

        void get_feature_vector (
            const std::vector<rectangle>& rects,
            feature_vector_type& psi
        ) const;

        const rectangle get_best_matching_rect (
            const rectangle& rect
        ) const;

        template <typename T, typename U>
        friend void serialize (
            const scan_image_pyramid<T,U>& item,
            std::ostream& out
        );

        template <typename T, typename U>
        friend void deserialize (
            scan_image_pyramid<T,U>& item,
            std::istream& in 
        );

    private:
        static bool compare_pair_rect (
            const std::pair<double, rectangle>& a,
            const std::pair<double, rectangle>& b
        )
        {
            return a.first < b.first;
        }

        struct detection_template
        {
            rectangle object_box; // always centered at (0,0)
            std::vector<rectangle> rects; // template with respect to (0,0)
        };

        friend void serialize(const detection_template& item, std::ostream& out)
        {
            serialize(item.object_box, out);
            serialize(item.rects, out);
        }
        friend void deserialize(detection_template& item, std::istream& in)
        {
            deserialize(item.object_box, in);
            deserialize(item.rects, in);
        }

        void get_mapped_rect_and_metadata (
            rectangle rect,
            rectangle& mapped_rect,
            detection_template& best_template,
            unsigned long& best_level
        ) const;


        feature_extractor_type feats_config; // just here to hold configuration.  use it to populate the feats elements.
        typename array<feature_extractor_type>::kernel_2a feats;
        std::vector<detection_template> det_templates;
        unsigned long max_dets_per_template;
        unsigned long max_pyramid_levels;

    };

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    void serialize (
        const scan_image_pyramid<T,U>& item,
        std::ostream& out
    )
    {
        serialize(item.feats_config, out);
        serialize(item.feats, out);
        serialize(item.det_templates, out);
        serialize(item.max_dets_per_template, out);
        serialize(item.max_pyramid_levels, out);
    }

// ----------------------------------------------------------------------------------------

    template <typename T, typename U>
    void deserialize (
        scan_image_pyramid<T,U>& item,
        std::istream& in 
    )
    {
        deserialize(item.feats_config, in);
        deserialize(item.feats, in);
        deserialize(item.det_templates, in);
        deserialize(item.max_dets_per_template, in);
        deserialize(item.max_pyramid_levels, in);
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                         scan_image_pyramid member functions
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    scan_image_pyramid (
    ) : 
        max_dets_per_template(10000),
        max_pyramid_levels(1000)
    {
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    template <
        typename image_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    load (
        const image_type& img
    )
    {
        unsigned long levels = 0;
        rectangle rect = get_rect(img);

        // figure out how many pyramid levels we should be using based on the image size
        pyramid_type pyr;
        while (rect.width() > 20 && rect.height() > 20)
        {
            rect = pyr.rect_down(rect);
            ++levels;

            if (levels >= max_pyramid_levels)
                break;
        }

        if (feats.max_size() < levels)
            feats.set_max_size(levels);
        feats.set_size(levels);

        for (unsigned long i = 0; i < feats.size(); ++i)
            feats[i].copy_configuration(feats_config);

        // build our feature pyramid
        feats[0].load(img);
        if (feats.size() > 1)
        {
            image_type temp1, temp2;
            pyr(img, temp1);
            feats[1].load(temp1);
            swap(temp1,temp2);

            for (unsigned long i = 2; i < feats.size(); ++i)
            {
                pyr(temp2, temp1);
                feats[i].load(temp1);
                swap(temp1,temp2);
            }
        }


    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    unsigned long scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_max_detections_per_template (
    ) const
    {
        return max_dets_per_template;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    set_max_detections_per_template (
        unsigned long max_dets
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(max_dets > 0 ,
            "\t void scan_image_pyramid::set_max_detections_per_template()"
            << "\n\t The max number of possible detections can't be zero. "
            << "\n\t max_dets: " << max_dets
            << "\n\t this: " << this
            );

        max_dets_per_template = max_dets;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    bool scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    is_loaded_with_image (
    ) const
    {
        return feats.size() != 0;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    copy_configuration(
        const feature_extractor_type& fe
    )
    {
        return feats_config.copy_configuration(fe);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    copy_configuration (
        const scan_image_pyramid& item
    )
    {
        feats_config.copy_configuration(item.feats_config);
        det_templates = item.det_templates;
        max_dets_per_template = item.max_dets_per_template;
        max_pyramid_levels = item.max_pyramid_levels;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    add_detection_template (
        const rectangle& object_box,
        const std::vector<rectangle>& feature_extraction_regions 
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT((get_num_detection_templates() == 0 || 
                        get_num_components_per_detection_template() == feature_extraction_regions.size()) &&
                        center(object_box) == point(0,0),
            "\t void scan_image_pyramid::add_detection_template()"
            << "\n\t The number of rects in this new detection template doesn't match "
            << "\n\t the number in previous detection templates."
            << "\n\t get_num_components_per_detection_template(): " << get_num_components_per_detection_template()
            << "\n\t feature_extraction_regions.size(): " << feature_extraction_regions.size()
            << "\n\t this: " << this
            );

        detection_template temp;
        temp.object_box = object_box;
        temp.rects = feature_extraction_regions;
        det_templates.push_back(temp);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    unsigned long scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_num_detection_templates (
    ) const
    {
        return det_templates.size();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    unsigned long scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_num_components_per_detection_template (
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(get_num_detection_templates() > 0 ,
            "\t unsigned long scan_image_pyramid::get_num_components_per_detection_template()"
            << "\n\t You need to give some detection templates before calling this function. "
            << "\n\t get_num_detection_templates(): " << get_num_detection_templates()
            << "\n\t this: " << this
            );

        return det_templates[0].rects.size();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    long scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_num_dimensions (
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(get_num_detection_templates() > 0 ,
            "\t long scan_image_pyramid::get_num_dimensions()"
            << "\n\t You need to give some detection templates before calling this function. "
            << "\n\t get_num_detection_templates(): " << get_num_detection_templates()
            << "\n\t this: " << this
            );

        return feats_config.get_num_dimensions()*get_num_components_per_detection_template();
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    unsigned long scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_max_pyramid_levels (
    ) const
    {
        return max_pyramid_levels;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    set_max_pyramid_levels (
        unsigned long max_levels
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(max_levels > 0 ,
            "\t void scan_image_pyramid::set_max_pyramid_levels()"
            << "\n\t You can't have zero levels. "
            << "\n\t max_levels: " << max_levels 
            << "\n\t this: " << this
            );

        max_pyramid_levels = max_levels;

    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    detect (
        const feature_vector_type& w,
        std::vector<std::pair<double, rectangle> >& dets,
        const double thresh
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(get_num_detection_templates() > 0 &&
                    is_loaded_with_image() &&
                    w.size() >= get_num_dimensions(), 
            "\t void scan_image_pyramid::detect()"
            << "\n\t Invalid inputs were given to this function "
            << "\n\t get_num_detection_templates(): " << get_num_detection_templates()
            << "\n\t is_loaded_with_image(): " << is_loaded_with_image()
            << "\n\t w.size():               " << w.size()
            << "\n\t get_num_dimensions():   " << get_num_dimensions()
            << "\n\t this: " << this
            );

        dets.clear();

        array<array2d<double> >::kernel_2a saliency_images;
        saliency_images.set_max_size(get_num_components_per_detection_template());
        saliency_images.set_size(get_num_components_per_detection_template());
        std::vector<std::pair<unsigned int,rectangle> > region_rects(get_num_components_per_detection_template()); 
        pyramid_type pyr;
        std::vector<std::pair<double, point> > point_dets;

        // for all pyramid levels
        for (unsigned long l = 0; l < feats.size(); ++l)
        {
            for (unsigned long i = 0; i < saliency_images.size(); ++i)
                saliency_images[i].set_size(feats[l].nr(), feats[l].nc());

            // build saliency images for pyramid level l 
            for (long r = 0; r < feats[l].nr(); ++r)
            {
                for (long c = 0; c < feats[l].nc(); ++c)
                {
                    const typename feature_extractor_type::descriptor_type& descriptor = feats[l](r,c);
                    for (unsigned long i = 0; i < saliency_images.size(); ++i)
                    {
                        const unsigned long offset = feats_config.get_num_dimensions()*i;

                        double sum = 0;
                        for (unsigned long k = 0; k < descriptor.size(); ++k)
                        {
                            sum += w(descriptor[k].first + offset)*descriptor[k].second;
                        }
                        saliency_images[i][r][c] = sum;
                    }
                }
            }

            // now search the saliency images
            for (unsigned long i = 0; i < det_templates.size(); ++i)
            {
                const point offset = -feats[l].image_to_feat_space(point(0,0));
                for (unsigned long j = 0; j < region_rects.size(); ++j)
                    region_rects[j] = std::make_pair(j, translate_rect(feats[l].image_to_feat_space(det_templates[i].rects[j]),offset)); 

                scan_image(point_dets, saliency_images, region_rects, thresh, max_dets_per_template); 

                // convert all the point detections into rectangles at the original image scale and coordinate system
                for (unsigned long j = 0; j < point_dets.size(); ++j)
                {
                    const double score = point_dets[j].first;
                    point p = point_dets[j].second;
                    p = feats[l].feat_to_image_space(p);
                    rectangle rect = translate_rect(det_templates[i].object_box, p);
                    rect = pyr.rect_up(rect, l);

                    dets.push_back(std::make_pair(score, rect));
                }
            }
        }

        std::sort(dets.rbegin(), dets.rend(), compare_pair_rect);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    const rectangle scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_best_matching_rect (
        const rectangle& rect
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(get_num_detection_templates() > 0 &&
                    is_loaded_with_image(),
            "\t const rectangle scan_image_pyramid::get_best_matching_rect()"
            << "\n\t Invalid inputs were given to this function "
            << "\n\t get_num_detection_templates(): " << get_num_detection_templates()
            << "\n\t is_loaded_with_image(): " << is_loaded_with_image()
            << "\n\t this: " << this
            );

        rectangle mapped_rect;
        detection_template best_template;
        unsigned long best_level;
        get_mapped_rect_and_metadata(rect, mapped_rect, best_template, best_level);
        return mapped_rect;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_mapped_rect_and_metadata (
        rectangle rect,
        rectangle& mapped_rect,
        detection_template& best_template,
        unsigned long& best_level
    ) const
    {
        pyramid_type pyr;
        // Figure out the pyramid level which best matches rect against one of our 
        // detection template object boxes.
        best_level = 0;
        double match_score = std::numeric_limits<double>::infinity();

        const dlib::vector<double,2> p(rect.width(), rect.height());

        // for all the levels
        for (unsigned long l = 0; l < feats.size(); ++l)
        {
            // Run the center point through the feature/image space transformation just to make
            // sure we exactly replicate the procedure for shifting an object_box used elsewhere 
            // in this file.
            const point origin = feats[l].feat_to_image_space(feats[l].image_to_feat_space(center(pyr.rect_down(rect,l))));

            for (unsigned long t = 0; t < det_templates.size(); ++t)
            {
                // Map this detection template into the normal image space and see how
                // close it is to the rect we are looking for.  We do the translation here
                // because the rect_up() routine takes place using integer arithmetic and
                // could potentially give slightly different results with and without the
                // translation.
                rectangle mapped_rect = translate_rect(det_templates[t].object_box, origin);
                mapped_rect = pyr.rect_up(mapped_rect, l);

                const dlib::vector<double,2> p2(mapped_rect.width(),
                                                mapped_rect.height());
                if ((p-p2).length() < match_score)
                {
                    match_score = (p-p2).length();
                    best_level = l;
                    best_template = det_templates[t];
                }
            }
        }


        // Now get the features out of feats[best_level].  But first translate best_template 
        // into the right spot (it should be centered at the location determined by rect)
        // and convert it into the feature image coordinate system.
        rect = pyr.rect_down(rect,best_level);
        const point offset = -feats[best_level].image_to_feat_space(point(0,0));
        const point origin = feats[best_level].image_to_feat_space(center(rect)) + offset;
        for (unsigned long k = 0; k < best_template.rects.size(); ++k)
        {
            rectangle temp = best_template.rects[k];
            temp = feats[best_level].image_to_feat_space(temp);
            temp = translate_rect(temp, origin);
            temp = get_rect(feats[best_level]).intersect(temp);
            best_template.rects[k] = temp;
        }

        // The input rectangle was mapped to one of the detection templates.  Reverse the process
        // to figure out what the mapped rectangle is in the original input space.
        mapped_rect = translate_rect(best_template.object_box, feats[best_level].feat_to_image_space(origin-offset));
        mapped_rect = pyr.rect_up(mapped_rect, best_level);
    }

// ----------------------------------------------------------------------------------------

    template <
        typename Pyramid_type,
        typename Feature_extractor_type
        >
    void scan_image_pyramid<Pyramid_type,Feature_extractor_type>::
    get_feature_vector (
        const std::vector<rectangle>& rects,
        feature_vector_type& psi
    ) const
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(get_num_detection_templates() > 0 &&
                    is_loaded_with_image() &&
                    psi.size() >= get_num_dimensions(), 
            "\t void scan_image_pyramid::get_feature_vector()"
            << "\n\t Invalid inputs were given to this function "
            << "\n\t get_num_detection_templates(): " << get_num_detection_templates()
            << "\n\t is_loaded_with_image(): " << is_loaded_with_image()
            << "\n\t psi.size():             " << psi.size()
            << "\n\t get_num_dimensions():   " << get_num_dimensions()
            << "\n\t this: " << this
            );

        psi = 0;


        pyramid_type pyr;
        for (unsigned long i = 0; i < rects.size(); ++i)
        {
            rectangle mapped_rect;
            detection_template best_template;
            unsigned long best_level;
            get_mapped_rect_and_metadata (rects[i], mapped_rect, best_template, best_level);

            for (unsigned long j = 0; j < best_template.rects.size(); ++j)
            {
                const rectangle rect = best_template.rects[j];
                const unsigned long template_region_id = j;
                const unsigned long offset = feats_config.get_num_dimensions()*template_region_id;
                for (long r = rect.top(); r <= rect.bottom(); ++r)
                {
                    for (long c = rect.left(); c <= rect.right(); ++c)
                    {
                        const typename feature_extractor_type::descriptor_type& descriptor = feats[best_level](r,c);
                        for (unsigned long k = 0; k < descriptor.size(); ++k)
                        {
                            psi(descriptor[k].first + offset) += descriptor[k].second;
                        }
                    }
                }
            }
        }

    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SCAN_IMaGE_PYRAMID_H__


