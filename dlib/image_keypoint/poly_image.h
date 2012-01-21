// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_POLY_ImAGE_H__
#define DLIB_POLY_ImAGE_H__

#include "poly_image_abstract.h"
#include "build_separable_poly_filters.h"
#include "../algs.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../geometry.h"
#include <cmath>

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        long Downsample
        >
    class poly_image : noncopyable
    {
        COMPILE_TIME_ASSERT(Downsample >= 1);
    public:
        const static long downsample = Downsample;
        typedef matrix<double, 0, 1> descriptor_type;

        poly_image(
            long order_,
            long window_size_,
            bool normalization = true
        )
        {
            setup(order_, window_size_);
            set_uses_normalization(normalization);
        }

        poly_image (
        ) 
        {
            clear();
        }

        void clear (
        )
        {
            normalize = true;
            poly_coef.clear();
            order = 3;
            window_size = 13;
            border_size = (long)std::ceil(std::floor(window_size/2.0)/downsample);
            num_rows = 0;
            num_cols = 0;
            filters = build_separable_poly_filters(order, window_size);
        }

        long get_order (
        ) const
        {
            return order;
        }

        long get_window_size (
        ) const
        {
            return window_size;
        }

        void setup (
            long order_,
            long window_size_
        )
        {
            // make sure requires clause is not broken
            DLIB_ASSERT(1 <= order_ && order_ <= 6 &&
                        window_size_ >= 3 && (window_size_%2) == 1,
                "\t descriptor_type poly_image::setup()"
                << "\n\t Invalid arguments were given to this function."
                << "\n\t order_:       " << order_ 
                << "\n\t window_size_: " << window_size_ 
                << "\n\t this: " << this
                );


            poly_coef.clear();
            order = order_;
            window_size = window_size_;
            border_size = (long)std::ceil(std::floor(window_size/2.0)/downsample);
            num_rows = 0;
            num_cols = 0;
            filters = build_separable_poly_filters(order, window_size);
        }

        bool uses_normalization (
        ) const { return normalize; }

        void set_uses_normalization (
            bool normalization
        )
        {
            normalize = normalization;
        }

        void copy_configuration (
            const poly_image& item
        )
        {
            normalize = item.normalize;
            if (order != item.order || 
                window_size != item.window_size)
            {
                order = item.order;
                window_size = item.window_size;
                border_size = item.border_size;
                filters = item.filters;
            }
        }

        template <
            typename image_type
            >
        inline void load (
            const image_type& img
        )
        {
            COMPILE_TIME_ASSERT( pixel_traits<typename image_type::type>::has_alpha == false );

            poly_coef.resize(get_num_dimensions());
            des.set_size(get_num_dimensions());


            if (normalize)
            {
                array2d<float> coef0;
                rectangle rect = filter_image(img, coef0, filters[0]);
                num_rows = rect.height();
                num_cols = rect.width();

                for (unsigned long i = 1; i < filters.size(); ++i)
                {
                    filter_image(img, poly_coef[i-1], filters[i]);

                    // intensity normalize everything
                    for (long r = 0; r < coef0.nr(); ++r)
                    {
                        for (long c = 0; c < coef0.nc(); ++c)
                        {
                            if (coef0[r][c] >= 1)
                                poly_coef[i-1][r][c] /= coef0[r][c];
                            else
                                poly_coef[i-1][r][c] = 0;
                        }
                    }
                }
            }
            else
            {
                rectangle rect;
                for (unsigned long i = 0; i < filters.size(); ++i)
                {
                    rect = filter_image(img, poly_coef[i], filters[i]);
                }
                num_rows = rect.height();
                num_cols = rect.width();
            }
        }

        void unload()
        {
            poly_coef.clear();
            num_rows = 0;
            num_cols = 0;
        }

        inline unsigned long size (
        ) const { return static_cast<unsigned long>(nr()*nc()); }

        inline long nr (
        ) const { return num_rows; }

        inline long nc (
        ) const { return num_cols; }

        long get_num_dimensions (
        ) const
        {
            if (normalize)
            {
                // -1 because we discard the constant term of the polynomial.
                return filters.size()-1;
            }
            else
            {
                return filters.size();
            }
        }

        inline const descriptor_type& operator() (
            long row,
            long col
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT( 0 <= row && row < nr() &&
                         0 <= col && col < nc(),
                "\t descriptor_type poly_image::operator()()"
                << "\n\t invalid row or col argument"
                << "\n\t row:  " << row
                << "\n\t col:  " << col 
                << "\n\t nr(): " << nr() 
                << "\n\t nc(): " << nc() 
                << "\n\t this: " << this
                );

            // add because of the zero border around the poly_coef images
            row += border_size;
            col += border_size;

            for (long i = 0; i < des.size(); ++i)
                des(i) = poly_coef[i][row][col];

            return des;
        }

        const rectangle get_block_rect (
            long row,
            long col
        ) const
        {
            return centered_rect(downsample*point(col+border_size, row+border_size), 
                                 window_size, window_size);
        }

        const point image_to_feat_space (
            const point& p
        ) const
        {
            return p/downsample - point(border_size, border_size);
        }

        const rectangle image_to_feat_space (
            const rectangle& rect
        ) const
        {
            return rectangle(image_to_feat_space(rect.tl_corner()), image_to_feat_space(rect.br_corner()));
        }

        const point feat_to_image_space (
            const point& p
        ) const
        {
            return (p + point(border_size, border_size))*downsample;
        }

        const rectangle feat_to_image_space (
            const rectangle& rect
        ) const
        {
            return rectangle(feat_to_image_space(rect.tl_corner()), feat_to_image_space(rect.br_corner()));
        }



        friend void serialize (const poly_image& item, std::ostream& out) 
        {
            int version = 1;
            serialize(version, out);
            serialize(item.poly_coef, out);
            serialize(item.order, out);
            serialize(item.window_size, out);
            serialize(item.border_size, out);
            serialize(item.num_rows, out);
            serialize(item.num_cols, out);
            serialize(item.normalize, out);
            serialize(item.filters, out);
        }

        friend void deserialize (poly_image& item, std::istream& in )
        {
            int version = 0;
            deserialize(version, in);
            if (version != 1)
                throw dlib::serialization_error("Unexpected version found while deserializing dlib::poly_image");

            deserialize(item.poly_coef, in);
            deserialize(item.order, in);
            deserialize(item.window_size, in);
            deserialize(item.border_size, in);
            deserialize(item.num_rows, in);
            deserialize(item.num_cols, in);
            deserialize(item.normalize, in);
            deserialize(item.filters, in);
        }

    private:

        template <typename image_type>
        rectangle filter_image (
            const image_type& img,
            array2d<float>& out,
            const std::vector<separable_filter_type>& filter
        ) const
        {
            rectangle rect = spatially_filter_image_separable_down(downsample, img, out, filter[0].first, filter[0].second);
            for (unsigned long i = 1; i < filter.size(); ++i)
            {
                spatially_filter_image_separable_down(downsample, img, out, filter[i].first, filter[i].second, 1, false, true);
            }
            return rect;
        }



        std::vector<std::vector<separable_filter_type> > filters;

        dlib::array<array2d<float> >::expand_1b poly_coef;
        long order;
        long window_size;
        long border_size;
        long num_rows;
        long num_cols;

        bool normalize;

        mutable descriptor_type des;
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_POLY_ImAGE_H__


