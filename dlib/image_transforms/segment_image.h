// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SEGMENT_ImAGE_H__
#define DLIB_SEGMENT_ImAGE_H__

#include "segment_image_abstract.h"
#include "../algs.h"
#include <vector>
#include "../geometry.h"
#include "../disjoint_subsets.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
        struct graph_image_segmentation_data
        {
            graph_image_segmentation_data() : component_size(1), internal_diff(0) {}
            unsigned long component_size;
            unsigned short internal_diff;
        };

        template <typename T>
        inline T edge_diff(
            const T& a,
            const T& b
        )
        {
            if (a > b)
                return a - b;
            else
                return b - a;
        }

        struct segment_image_edge_data
        {
            segment_image_edge_data (){}

            segment_image_edge_data (
                const rectangle& rect,
                const point& p1,
                const point& p2,
                const unsigned short& diff_
            ) :
                idx1(p1.y()*rect.width() + p1.x()),
                idx2(p2.y()*rect.width() + p2.x()),
                diff(diff_)
            {}

            unsigned long idx1;
            unsigned long idx2;
            unsigned short diff;
        };
    }

// ----------------------------------------------------------------------------------------

// This is an overload of segment_image() that is optimized to segment images with 8bit
// pixels very quickly.  We do this by using a radix sort instead of quicksort.
    template <
        typename in_image_type,
        typename out_image_type
        >
    typename enable_if_c<is_same_type<typename in_image_type::type,uint8>::value ||
                         is_same_type<typename in_image_type::type,uint16>::value>::type 
    segment_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        const double k = 200,
        const unsigned long min_size = 10
    )
    {
        using namespace dlib::impl;
        typedef typename in_image_type::type ptype;

        // make sure requires clause is not broken
        DLIB_ASSERT(is_same_object(in_img, out_img) == false,
            "\t void segment_image()"
            << "\n\t The input images can't be the same object."
            );

        COMPILE_TIME_ASSERT(is_unsigned_type<typename out_image_type::type>::value);

        out_img.set_size(in_img.nr(), in_img.nc());
        // don't bother doing anything if the image is too small
        if (in_img.nr() < 2 || in_img.nc() < 2)
        {
            assign_all_pixels(out_img,0);
            return;
        }

        disjoint_subsets sets;
        sets.set_size(in_img.size());


        std::vector<graph_image_segmentation_data> data(in_img.size());

        std::vector<unsigned long> counts(std::numeric_limits<ptype>::max()+1, 0);

        border_enumerator be(get_rect(in_img), 1);
        // we are going to do a radix sort on the edge weights.  So the first step
        // is to accumulate them into count.
        const rectangle area = get_rect(in_img);
        while (be.move_next())
        {
            const long r = be.element().y();
            const long c = be.element().x();
            const ptype pix = in_img[r][c];
            if (area.contains(c-1,r))   counts[edge_diff(pix, in_img[r  ][c-1])] += 1;
            if (area.contains(c+1,r))   counts[edge_diff(pix, in_img[r  ][c+1])] += 1;
            if (area.contains(c  ,r-1)) counts[edge_diff(pix, in_img[r-1][c  ])] += 1;
            if (area.contains(c  ,r+1)) counts[edge_diff(pix, in_img[r+1][c  ])] += 1;
        }
        for (long r = 1; r+1 < in_img.nr(); ++r)
        {
            for (long c = 1; c+1 < in_img.nc(); ++c)
            {
                const ptype pix = in_img[r][c];
                counts[edge_diff(pix, in_img[r-1][c+1])] += 1;
                counts[edge_diff(pix, in_img[r  ][c+1])] += 1;
                counts[edge_diff(pix, in_img[r+1][c  ])] += 1;
                counts[edge_diff(pix, in_img[r+1][c+1])] += 1;
            }
        }

        const unsigned long num_edges = shrink_rect(area,1).area()*4 + in_img.nr()*2*3 - 4 + (in_img.nc()-2)*2*3;
        std::vector<segment_image_edge_data> sorted_edges(num_edges);

        // integrate counts.  The idea is to have sorted_edges[counts[i]] be the location that edges
        // with an edge_diff of i go.  So counts[0] == 0, counts[1] == number of 0 edge diff edges, etc.
        unsigned long prev = counts[0];
        for (unsigned long i = 1; i < counts.size(); ++i)
        {
            const unsigned long temp = counts[i];
            counts[i] += counts[i-1];
            counts[i-1] -= prev;
            prev = temp;
        }
        counts[counts.size()-1] -= prev;


        // now build a sorted list of all the edges
        be.reset();
        while(be.move_next())
        {
            const point p = be.element();
            const long r = p.y();
            const long c = p.x();
            const ptype pix = in_img[r][c];
            if (area.contains(c-1,r))
            {
                const ptype diff = edge_diff(pix, in_img[r  ][c-1]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c-1,r),diff);
            }

            if (area.contains(c+1,r))
            {
                const ptype diff = edge_diff(pix, in_img[r  ][c+1]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c+1,r),diff);
            }

            if (area.contains(c  ,r-1))
            {
                const ptype diff = edge_diff(pix, in_img[r-1][c  ]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c  ,r-1),diff);
            }

            if (area.contains(c  ,r+1))
            {
                const ptype diff = edge_diff(pix, in_img[r+1][c  ]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c  ,r+1),diff);
            }
        }
        // same thing as the above loop but now we do it on the interior of the image and therefore
        // don't have to include the boundary checking if statements used above.
        for (long r = 1; r+1 < in_img.nr(); ++r)
        {
            for (long c = 1; c+1 < in_img.nc(); ++c)
            {
                const point p(c,r);
                const ptype pix = in_img[r][c];
                ptype diff;

                diff = edge_diff(pix, in_img[r  ][c+1]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c+1,r),diff);
                diff = edge_diff(pix, in_img[r-1][c+1]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c+1,r-1),diff);
                diff = edge_diff(pix, in_img[r+1][c+1]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c+1,r+1),diff);
                diff = edge_diff(pix, in_img[r+1][c  ]);
                sorted_edges[counts[diff]++] = segment_image_edge_data(area,p,point(c  ,r+1),diff);
            }
        }



        // now start connecting blobs together to make a minimum spanning tree.
        for (unsigned long i = 0; i < sorted_edges.size(); ++i)
        {
            const unsigned long idx1 = sorted_edges[i].idx1;
            const unsigned long idx2 = sorted_edges[i].idx2;

            unsigned long set1 = sets.find_set(idx1);
            unsigned long set2 = sets.find_set(idx2);
            if (set1 != set2)
            {
                const ptype diff = sorted_edges[i].diff;
                const ptype tau1 = static_cast<ptype>(std::floor(k/data[set1].component_size));
                const ptype tau2 = static_cast<ptype>(std::floor(k/data[set2].component_size));

                const ptype mint = std::min(data[set1].internal_diff + tau1, 
                                            data[set2].internal_diff + tau2);
                if (diff <= mint)
                {
                    const unsigned long new_set = sets.merge_sets(set1, set2);
                    data[new_set].component_size = data[set1].component_size + data[set2].component_size;
                    data[new_set].internal_diff = diff;
                }
            }
        }

        // now merge any really small blobs
        if (min_size != 0)
        {
            for (unsigned long i = 0; i < sorted_edges.size(); ++i)
            {
                const unsigned long idx1 = sorted_edges[i].idx1;
                const unsigned long idx2 = sorted_edges[i].idx2;

                unsigned long set1 = sets.find_set(idx1);
                unsigned long set2 = sets.find_set(idx2);
                if (set1 != set2 && (data[set1].component_size < min_size || data[set2].component_size < min_size))
                {
                    const unsigned long new_set = sets.merge_sets(set1, set2);
                    data[new_set].component_size = data[set1].component_size + data[set2].component_size;
                    const ptype diff = sorted_edges[i].diff;
                    data[new_set].internal_diff = diff;
                }
            }
        }

        unsigned long idx = 0;
        for (long r = 0; r < out_img.nr(); ++r)
        {
            for (long c = 0; c < out_img.nc(); ++c)
            {
                out_img[r][c] = sets.find_set(idx++);
            }
        }
    }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    namespace impl
    {
        template <typename T, typename enabled = void>
        struct edge_diff_funct 
        {
            template <typename pixel_type>
            double operator()(
                const pixel_type& a,
                const pixel_type& b
            ) const
            {
                return length(pixel_to_vector<double>(a) - pixel_to_vector<double>(b));
            }
        };

        template <typename T>
        struct edge_diff_funct<T, typename enable_if<is_matrix<T> >::type>
        {
            double operator()(
                const T& a,
                const T& b
            ) const
            {
                return length(a-b);
            }
        };

        template <typename T>
        struct graph_image_segmentation_data2
        {
            graph_image_segmentation_data2() : component_size(1), internal_diff(0) {}
            unsigned long component_size;
            T internal_diff;
        };

        template <typename T>
        struct segment_image_edge_data2
        {
            segment_image_edge_data2 (){}

            segment_image_edge_data2 (
                const rectangle& rect,
                const point& p1,
                const point& p2,
                const T& diff_
            ) :
                idx1(p1.y()*rect.width() + p1.x()),
                idx2(p2.y()*rect.width() + p2.x()),
                diff(diff_)
            {}

            bool operator<(const segment_image_edge_data2& item) const
            { return diff < item.diff; }

            unsigned long idx1;
            unsigned long idx2;
            T diff;
        };
    }

    // This is the general purpose version of segment_image().  It handles all pixel types.
    template <
        typename in_image_type,
        typename out_image_type
        >
    typename disable_if_c<is_same_type<typename in_image_type::type,uint8>::value ||
                         is_same_type<typename in_image_type::type,uint16>::value>::type 
    segment_image (
        const in_image_type& in_img,
        out_image_type& out_img,
        const double k = 200,
        const unsigned long min_size = 10
    )
    {
        using namespace dlib::impl;


        typedef double diff_type;
        typedef typename in_image_type::type ptype;
        edge_diff_funct<ptype> edge_diff;

        // make sure requires clause is not broken
        DLIB_ASSERT(is_same_object(in_img, out_img) == false,
            "\t void segment_image()"
            << "\n\t The input images can't be the same object."
            );

        COMPILE_TIME_ASSERT(is_unsigned_type<typename out_image_type::type>::value);

        out_img.set_size(in_img.nr(), in_img.nc());
        // don't bother doing anything if the image is too small
        if (in_img.nr() < 2 || in_img.nc() < 2)
        {
            assign_all_pixels(out_img,0);
            return;
        }

        disjoint_subsets sets;
        sets.set_size(in_img.size());


        std::vector<graph_image_segmentation_data2<diff_type> > data(in_img.size());

        const rectangle area = get_rect(in_img);

        typedef segment_image_edge_data2<diff_type> segment_image_edge_data;

        std::vector<segment_image_edge_data> sorted_edges;
        sorted_edges.reserve(area.area()*4);


        border_enumerator be(get_rect(in_img), 1);

        // now build a sorted list of all the edges
        be.reset();
        while(be.move_next())
        {
            const point p = be.element();
            const long r = p.y();
            const long c = p.x();
            const ptype& pix = in_img[r][c];
            if (area.contains(c-1,r))
            {
                const diff_type diff = edge_diff(pix, in_img[r  ][c-1]);
                sorted_edges.push_back(segment_image_edge_data(area,p,point(c-1,r),diff));
            }

            if (area.contains(c+1,r))
            {
                const diff_type diff = edge_diff(pix, in_img[r  ][c+1]);
                sorted_edges.push_back(segment_image_edge_data(area,p,point(c+1,r),diff));
            }

            if (area.contains(c  ,r-1))
            {
                const diff_type diff = edge_diff(pix, in_img[r-1][c  ]);
                sorted_edges.push_back( segment_image_edge_data(area,p,point(c  ,r-1),diff));
            }
            if (area.contains(c  ,r+1))
            {
                const diff_type diff = edge_diff(pix, in_img[r+1][c  ]);
                sorted_edges.push_back( segment_image_edge_data(area,p,point(c  ,r+1),diff));
            }
        }
        // same thing as the above loop but now we do it on the interior of the image and therefore
        // don't have to include the boundary checking if statements used above.
        for (long r = 1; r+1 < in_img.nr(); ++r)
        {
            for (long c = 1; c+1 < in_img.nc(); ++c)
            {
                const point p(c,r);
                const ptype& pix = in_img[r][c];
                diff_type diff;

                diff = edge_diff(pix, in_img[r  ][c+1]);
                sorted_edges.push_back( segment_image_edge_data(area,p,point(c+1,r),diff));
                diff = edge_diff(pix, in_img[r+1][c+1]);
                sorted_edges.push_back( segment_image_edge_data(area,p,point(c+1,r+1),diff));
                diff = edge_diff(pix, in_img[r+1][c  ]);
                sorted_edges.push_back( segment_image_edge_data(area,p,point(c  ,r+1),diff));
                diff = edge_diff(pix, in_img[r-1][c+1]);
                sorted_edges.push_back( segment_image_edge_data(area,p,point(c+1,r-1),diff));
            }
        }

        std::sort(sorted_edges.begin(), sorted_edges.end());



        // now start connecting blobs together to make a minimum spanning tree.
        for (unsigned long i = 0; i < sorted_edges.size(); ++i)
        {
            const unsigned long idx1 = sorted_edges[i].idx1;
            const unsigned long idx2 = sorted_edges[i].idx2;

            unsigned long set1 = sets.find_set(idx1);
            unsigned long set2 = sets.find_set(idx2);
            if (set1 != set2)
            {
                const diff_type diff = sorted_edges[i].diff;
                const diff_type tau1 = k/data[set1].component_size;
                const diff_type tau2 = k/data[set2].component_size;

                const diff_type mint = std::min(data[set1].internal_diff + tau1, 
                                            data[set2].internal_diff + tau2);
                if (diff <= mint)
                {
                    const unsigned long new_set = sets.merge_sets(set1, set2);
                    data[new_set].component_size = data[set1].component_size + data[set2].component_size;
                    data[new_set].internal_diff = diff;
                }
            }
        }

        // now merge any really small blobs
        if (min_size != 0)
        {
            for (unsigned long i = 0; i < sorted_edges.size(); ++i)
            {
                const unsigned long idx1 = sorted_edges[i].idx1;
                const unsigned long idx2 = sorted_edges[i].idx2;

                unsigned long set1 = sets.find_set(idx1);
                unsigned long set2 = sets.find_set(idx2);
                if (set1 != set2 && (data[set1].component_size < min_size || data[set2].component_size < min_size))
                {
                    const unsigned long new_set = sets.merge_sets(set1, set2);
                    data[new_set].component_size = data[set1].component_size + data[set2].component_size;
                }
            }
        }

        unsigned long idx = 0;
        for (long r = 0; r < out_img.nr(); ++r)
        {
            for (long c = 0; c < out_img.nc(); ++c)
            {
                out_img[r][c] = sets.find_set(idx++);
            }
        }
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SEGMENT_ImAGE_H__

