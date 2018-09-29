// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_LABEL_CONNeCTED_BLOBS_H_
#define DLIB_LABEL_CONNeCTED_BLOBS_H_

#include "label_connected_blobs_abstract.h"
#include "../geometry.h"
#include <stack>
#include <vector>
#include "thresholding.h"
#include "assign_image.h"
#include <queue>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    struct neighbors_24
    {
        void operator() (
            const point& p,
            std::vector<point>& neighbors
        ) const
        {
            for (long i = -2; i <= 2; ++i)
                for (long j = -2; j <= 2; ++j)
                    if (i!=0||j!=0)
                        neighbors.push_back(point(p.x()+i,p.y()+j));
        }
    };

    struct neighbors_8 
    {
        void operator() (
            const point& p,
            std::vector<point>& neighbors
        ) const
        {
            neighbors.push_back(point(p.x()+1,p.y()+1));
            neighbors.push_back(point(p.x()+1,p.y()  ));
            neighbors.push_back(point(p.x()+1,p.y()-1));

            neighbors.push_back(point(p.x(),p.y()+1));
            neighbors.push_back(point(p.x(),p.y()-1));

            neighbors.push_back(point(p.x()-1,p.y()+1));
            neighbors.push_back(point(p.x()-1,p.y()  ));
            neighbors.push_back(point(p.x()-1,p.y()-1));
        }
    };

    struct neighbors_4 
    {
        void operator() (
            const point& p,
            std::vector<point>& neighbors
        ) const
        {
            neighbors.push_back(point(p.x()+1,p.y()));
            neighbors.push_back(point(p.x()-1,p.y()));
            neighbors.push_back(point(p.x(),p.y()+1));
            neighbors.push_back(point(p.x(),p.y()-1));
        }
    };

// ----------------------------------------------------------------------------------------

    struct connected_if_both_not_zero
    {
        template <typename image_type>
        bool operator() (
            const image_type& img,
            const point& a,
            const point& b
        ) const
        {
            return (img[a.y()][a.x()] != 0 && img[b.y()][b.x()] != 0);
        }
    };

    struct connected_if_equal
    {
        template <typename image_type>
        bool operator() (
            const image_type& img,
            const point& a,
            const point& b
        ) const
        {
            return (img[a.y()][a.x()] == img[b.y()][b.x()]);
        }
    };

// ----------------------------------------------------------------------------------------

    struct zero_pixels_are_background
    {
        template <typename image_type>
        bool operator() (
            const image_type& img,
            const point& p
        ) const
        {
            return img[p.y()][p.x()] == 0;
        }

    };

    struct nothing_is_background 
    {
        template <typename image_type>
        bool operator() (
            const image_type&, 
            const point& 
        ) const
        {
            return false;
        }

    };

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename label_image_type,
        typename background_functor_type,
        typename neighbors_functor_type,
        typename connected_functor_type
        >
    unsigned long label_connected_blobs (
        const image_type& img_,
        const background_functor_type& is_background,
        const neighbors_functor_type&  get_neighbors,
        const connected_functor_type&  is_connected,
        label_image_type& label_img_
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_same_object(img_, label_img_) == false,
            "\t unsigned long label_connected_blobs()"
            << "\n\t The input image and output label image can't be the same object."
            );

        const_image_view<image_type> img(img_);
        image_view<label_image_type> label_img(label_img_);

        std::stack<point> neighbors;
        label_img.set_size(img.nr(), img.nc());
        assign_all_pixels(label_img, 0);
        unsigned long next = 1;

        if (img.size() == 0)
            return 0;

        const rectangle area = get_rect(img);

        std::vector<point> window;

        for (long r = 0; r < img.nr(); ++r)
        {
            for (long c = 0; c < img.nc(); ++c)
            {
                // skip already labeled pixels or background pixels
                if (label_img[r][c] != 0 || is_background(img,point(c,r)))
                    continue;

                label_img[r][c] = next;

                // label all the neighbors of this point 
                neighbors.push(point(c,r));
                while (neighbors.size() > 0)
                {
                    const point p = neighbors.top();
                    neighbors.pop();

                    window.clear();
                    get_neighbors(p, window);

                    for (unsigned long i = 0; i < window.size(); ++i)
                    {
                        if (area.contains(window[i]) &&                     // point in image.
                            !is_background(img,window[i]) &&                // isn't background.
                            label_img[window[i].y()][window[i].x()] == 0 && // haven't already labeled it.
                            is_connected(img, p, window[i]))                // it's connected.
                        {
                            label_img[window[i].y()][window[i].x()] = next;
                            neighbors.push(window[i]);
                        }
                    }
                }

                ++next;
            }
        }

        return next;
    }

// ----------------------------------------------------------------------------------------

    template <
        typename in_image_type,
        typename out_image_type
        >
    unsigned long label_connected_blobs_watershed (
        const in_image_type& img_,
        out_image_type& labels_,
        typename pixel_traits<typename image_traits<in_image_type>::pixel_type>::basic_pixel_type background_thresh,
        const double smoothing = 0
    )
    {
        // make sure requires clause is not broken
        DLIB_ASSERT(is_same_object(img_, labels_) == false,
            "\t unsigned long segment_image_watersheds()"
            << "\n\t The input images can't be the same object."
            );

        using label_pixel_type = typename image_traits<out_image_type>::pixel_type;

        DLIB_ASSERT(smoothing >= 0);
        COMPILE_TIME_ASSERT(is_unsigned_type<label_pixel_type>::value);


        struct watershed_points
        {
            watershed_points() = default;
            watershed_points(const point& p_, float score_, label_pixel_type label_): p(p_), score(score_), label(label_) {}

            point p;
            float score = 0;
            label_pixel_type label = std::numeric_limits<label_pixel_type>::max();

            bool is_seed() const { return label == std::numeric_limits<label_pixel_type>::max(); }

            bool operator< (const watershed_points& rhs) const
            {
                // If two pixels have the same score then we take the one with the smallest 
                // label out of the priority queue first.  We do this so that seed points
                // that are downhill from some larger blob will be consumed by it if they
                // haven't grown before the larger blob's flooding reaches them.  Doing
                // this helps a lot to avoid spuriously splitting blobs.
                if (score == rhs.score)
                {
                    return label > rhs.label;
                }
                return score < rhs.score;
            }

        };

        const_image_view<in_image_type> img(img_);
        image_view<out_image_type> labels(labels_);
        
        labels.set_size(img.nr(), img.nc());
        // Initially, all pixels have the background label of 0.
        assign_all_pixels(labels, 0);

        std::priority_queue<watershed_points> next;


        // Note that we never blur the image values we use to check against the
        // background_thresh.  We do however blur, if smoothing!=0, the pixel values used
        // to do the watershed.
        in_image_type img2_;
        if (smoothing != 0)
            gaussian_blur(img_, img2_, smoothing); 
        const_image_view<in_image_type> img2view(img2_);
        // point us at img2 if we are doing smoothing, otherwise point us at the input
        // image.
        const auto& img2 = smoothing!=0?img2view:img;

        // first find all the local maxima 
        for (long r = 1; r+1 < img.nr(); ++r)
        {
            for (long c = 1; c+1 < img.nc(); ++c)
            {

                if (img[r][c] < background_thresh)
                    continue;

                auto val = img2[r][c];
                // if img2[r][c] isn't a local maximum then skip it
                if (val < img2[r+1][c] ||
                    val < img2[r-1][c] ||
                    val < img2[r][c+1] ||
                    val < img2[r][c-1]
                )
                {
                    continue;
                }

                next.push(watershed_points(point(c,r), val, std::numeric_limits<label_pixel_type>::max()));
            }
        }


        const rectangle area = get_rect(img);


        label_pixel_type next_label = 1;


        std::vector<point> neighbors;
        neighbors_8 get_neighbors;
        while(next.size() > 0)
        {
            auto p = next.top();
            next.pop();

            label_pixel_type label;
            // If the next pixel is a seed of a new blob and is still labeled as a
            // background pixel (i.e. it hasn't been flooded over by a neighboring blob and
            // consumed by it) then we create a new label for this new blob.
            if (p.is_seed() && labels[p.p.y()][p.p.x()] == 0)
            {
                label = next_label++;
                labels[p.p.y()][p.p.x()] = label;
            }
            else
            {
                label = p.label;
            }


            neighbors.clear();
            get_neighbors(p.p, neighbors);
            for (auto& n : neighbors)
            {
                if (!area.contains(n) || labels[n.y()][n.x()] != 0 || img[n.y()][n.x()] < background_thresh)
                    continue;

                labels[n.y()][n.x()] = label;
                next.push(watershed_points(n, img2[n.y()][n.x()], label));
            }
        }

        return next_label;
    }

    template <
        typename in_image_type,
        typename out_image_type
        >
    unsigned long label_connected_blobs_watershed (
        const in_image_type& img,
        out_image_type& labels
    )
    {
        return label_connected_blobs_watershed(img, labels, partition_pixels(img));
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_LABEL_CONNeCTED_BLOBS_H_

