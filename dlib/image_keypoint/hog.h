// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_HoG_H__
#define DLIB_HoG_H__

#include "hog_abstract.h"
#include "../algs.h"
#include "../matrix.h"
#include "../array2d.h"
#include "../geometry.h"
#include <cmath>

namespace dlib
{
    enum 
    {
        hog_no_interpolation,
        hog_angle_interpolation,
        hog_full_interpolation,
        hog_signed_gradient,
        hog_unsigned_gradient
    };

    template <
        unsigned long cell_size_,
        unsigned long block_size_,
        unsigned long cell_stride_,
        unsigned long num_orientation_bins_,
        int           gradient_type_,
        int           interpolation_type_
        >
    class hog_image : noncopyable
    {
        COMPILE_TIME_ASSERT(cell_size_ > 1);
        COMPILE_TIME_ASSERT(block_size_ > 0);
        COMPILE_TIME_ASSERT(cell_stride_ > 0);
        COMPILE_TIME_ASSERT(num_orientation_bins_ > 0);

        COMPILE_TIME_ASSERT( gradient_type_ == hog_signed_gradient ||
                             gradient_type_ == hog_unsigned_gradient);

        COMPILE_TIME_ASSERT( interpolation_type_ == hog_no_interpolation ||
                             interpolation_type_ == hog_angle_interpolation ||
                             interpolation_type_ == hog_full_interpolation );


    public:

        const static unsigned long cell_size = cell_size_;
        const static unsigned long block_size = block_size_;
        const static unsigned long cell_stride = cell_stride_;
        const static unsigned long num_orientation_bins = num_orientation_bins_;
        const static int           gradient_type = gradient_type_;
        const static int           interpolation_type = interpolation_type_;

        const static long min_size = cell_size*block_size+2;

        typedef matrix<double, block_size*block_size*num_orientation_bins, 1> descriptor_type;

        hog_image (
        ) : 
            num_block_rows(0),
            num_block_cols(0)
        {}

        template <
            typename image_type
            >
        inline void load (
            const image_type& img
        )
        {
            COMPILE_TIME_ASSERT( pixel_traits<typename image_type::type>::has_alpha == false );
            load_impl(array_to_matrix(img));
        }

        inline unsigned long size (
        ) const { return static_cast<unsigned long>(nr()*nc()); }

        inline long nr (
        ) const { return num_block_rows; }

        inline long nc (
        ) const { return num_block_cols; }

        inline const descriptor_type& operator() (
            long row,
            long col
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT( 0 <= row && row < nr() &&
                         0 <= col && col < nc(),
                "\t descriptor_type hog_image::operator()()"
                << "\n\t invalid row or col argument"
                << "\n\t row:  " << row
                << "\n\t col:  " << col 
                << "\n\t nr(): " << nr() 
                << "\n\t nc(): " << nc() 
                << "\n\t this: " << this
                );

            row *= cell_stride;
            col *= cell_stride;
            ++row;
            ++col;

            int feat = 0;
            for (unsigned long r = 0; r < block_size; ++r)
            {
                for (unsigned long c = 0; c < block_size; ++c)
                {
                    for (unsigned long i = 0; i < num_orientation_bins; ++i)
                    {
                        des(feat++) = hist_cells[row+r][col+c].values[i];
                    }
                }
            }

            des /= length(des) + 1e-8;

            return des;
        }

        const rectangle get_block_rect (
            long row,
            long col
        ) const
        {
            // make sure requires clause is not broken
            DLIB_ASSERT( 0 <= row && row < nr() &&
                         0 <= col && col < nc(),
                "\t rectangle hog_image::get_block_rect()"
                << "\n\t invalid row or col argument"
                << "\n\t row:  " << row
                << "\n\t col:  " << col 
                << "\n\t nr(): " << nr() 
                << "\n\t nc(): " << nc() 
                << "\n\t this: " << this
                );

            row *= cell_stride;
            col *= cell_stride;

            row *= cell_size;
            col *= cell_size;

            // do this to account for the 1 pixel padding we use all around the image
            ++row;
            ++col;

            return rectangle(col, row, col+cell_size*block_size-1, row+cell_size*block_size-1);
        }

    private:

        template <
            typename image_type
            >
        void load_impl (
            const image_type& img
        )
        {
            // Note that we keep a border of 1 pixel all around the image so that we don't have
            // to worry about running outside the image when computing the horizontal and vertical 
            // gradients.

            // Note also that we have a border of unused cells around the hist_cells array so that we
            // don't have to worry about edge effects when doing the interpolation in the main loop
            // below.


            // check if the window is just too small
            if (img.nr() < min_size || img.nc() < min_size)
            {
                // If the image is smaller than our windows then there aren't any descriptors at all!
                num_block_rows = 0;
                num_block_cols = 0;
                return;
            }

            // Make sure we have the right number of cell histograms and that they are
            // all set to zero.
            hist_cells.set_size((img.nr()-2)/cell_size+2, (img.nc()-2)/cell_size+2);
            for (long r = 0; r < hist_cells.nr(); ++r)
            {
                for (long c = 0; c < hist_cells.nc(); ++c)
                {
                    hist_cells[r][c].zero();
                }
            }

            const double pi = 3.1415926535898;

            // loop over all the histogram cells and fill them out
            for (long rh = 1; rh < hist_cells.nr()-1; ++rh)
            {
                for (long ch = 1; ch < hist_cells.nc()-1; ++ch)
                {
                    // Fill out the current histogram cell.
                    // First, figure out the row and column offsets into the image for the current histogram cell.
                    const long roff = (rh-1)*cell_size + 1;
                    const long coff = (ch-1)*cell_size + 1;

                    for (long r = 0; r < (long)cell_size; ++r)
                    {
                        for (long c = 0; c < (long)cell_size; ++c)
                        {
                            unsigned long left; 
                            unsigned long right;
                            unsigned long top;   
                            unsigned long bottom; 

                            assign_pixel(left,   img(r+roff,c+coff-1));
                            assign_pixel(right,  img(r+roff,c+coff+1));
                            assign_pixel(top,    img(r+roff-1,c+coff));
                            assign_pixel(bottom, img(r+roff+1,c+coff));

                            double grad_x = (long)right-(long)left;
                            double grad_y = (long)top-(long)bottom;

                            // obtain the angle of the gradient.  Make sure it is scaled between 0 and 1.
                            double angle = std::max(0.0, std::atan2(grad_y, grad_x)/pi + 1)/2;


                            if (gradient_type == hog_unsigned_gradient)
                            {
                                angle *= 2;
                                if (angle >= 1)
                                    angle -= 1;
                            }


                            // now scale angle to between 0 and num_orientation_bins
                            angle *= num_orientation_bins;


                            const double strength = std::sqrt(grad_y*grad_y + grad_x*grad_x);


                            if (interpolation_type == hog_no_interpolation)
                            {
                                // no interpolation
                                hist_cells[rh][ch].values[round_to_int(angle)%num_orientation_bins] += strength;
                            }
                            else  // if we should do some interpolation
                            {
                                unsigned long quantized_angle_lower = std::floor(angle);
                                unsigned long quantized_angle_upper = std::ceil(angle);

                                quantized_angle_lower %= num_orientation_bins;
                                quantized_angle_upper %= num_orientation_bins;

                                const double angle_split = (angle-std::floor(angle));
                                const double upper_strength = angle_split*strength;
                                const double lower_strength = (1-angle_split)*strength;

                                if (interpolation_type == hog_angle_interpolation)
                                {
                                    // Stick into gradient histogram.  Note that we linearly interpolate between neighboring
                                    // histogram buckets.
                                    hist_cells[rh][ch].values[quantized_angle_lower] += lower_strength;
                                    hist_cells[rh][ch].values[quantized_angle_upper] += upper_strength;
                                }
                                else // here we do hog_full_interpolation
                                {
                                    const double center_r = (cell_size-1)/2.0;
                                    const double center_c = (cell_size-1)/2.0;

                                    const double lin_neighbor_r = std::abs(center_r - r)/cell_size;
                                    const double lin_main_r = 1-lin_neighbor_r;

                                    const double lin_neighbor_c = std::abs(center_c - c)/cell_size;
                                    const double lin_main_c = 1-lin_neighbor_c;

                                    // Which neighboring cells we interpolate into depends on which
                                    // corner of our main cell we are nearest.
                                    if (r < center_r)
                                    {
                                        if (c < center_c)
                                        {
                                            hist_cells[rh][ch].values[quantized_angle_upper] += upper_strength   * lin_main_r*lin_main_c;
                                            hist_cells[rh][ch].values[quantized_angle_lower] += lower_strength   * lin_main_r*lin_main_c;

                                            hist_cells[rh-1][ch].values[quantized_angle_upper] += upper_strength * lin_neighbor_r*lin_main_c;
                                            hist_cells[rh-1][ch].values[quantized_angle_lower] += lower_strength * lin_neighbor_r*lin_main_c;

                                            hist_cells[rh][ch-1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_main_r;
                                            hist_cells[rh][ch-1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_main_r;

                                            hist_cells[rh-1][ch-1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_neighbor_r;
                                            hist_cells[rh-1][ch-1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_neighbor_r;
                                        }
                                        else
                                        {
                                            hist_cells[rh][ch].values[quantized_angle_upper] += upper_strength   * lin_main_r*lin_main_c;
                                            hist_cells[rh][ch].values[quantized_angle_lower] += lower_strength   * lin_main_r*lin_main_c;

                                            hist_cells[rh-1][ch].values[quantized_angle_upper] += upper_strength * lin_neighbor_r*lin_main_c;
                                            hist_cells[rh-1][ch].values[quantized_angle_lower] += lower_strength * lin_neighbor_r*lin_main_c;

                                            hist_cells[rh][ch+1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_main_r;
                                            hist_cells[rh][ch+1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_main_r;

                                            hist_cells[rh-1][ch+1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_neighbor_r;
                                            hist_cells[rh-1][ch+1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_neighbor_r;
                                        }
                                    }
                                    else 
                                    {
                                        if (c < center_c)
                                        {
                                            hist_cells[rh][ch].values[quantized_angle_upper] += upper_strength   * lin_main_r*lin_main_c;
                                            hist_cells[rh][ch].values[quantized_angle_lower] += lower_strength   * lin_main_r*lin_main_c;

                                            hist_cells[rh+1][ch].values[quantized_angle_upper] += upper_strength * lin_neighbor_r*lin_main_c;
                                            hist_cells[rh+1][ch].values[quantized_angle_lower] += lower_strength * lin_neighbor_r*lin_main_c;

                                            hist_cells[rh][ch-1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_main_r;
                                            hist_cells[rh][ch-1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_main_r;

                                            hist_cells[rh+1][ch-1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_neighbor_r;
                                            hist_cells[rh+1][ch-1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_neighbor_r;
                                        }
                                        else
                                        {
                                            hist_cells[rh][ch].values[quantized_angle_upper] += upper_strength   * lin_main_r*lin_main_c;
                                            hist_cells[rh][ch].values[quantized_angle_lower] += lower_strength   * lin_main_r*lin_main_c;

                                            hist_cells[rh+1][ch].values[quantized_angle_upper] += upper_strength * lin_neighbor_r*lin_main_c;
                                            hist_cells[rh+1][ch].values[quantized_angle_lower] += lower_strength * lin_neighbor_r*lin_main_c;

                                            hist_cells[rh][ch+1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_main_r;
                                            hist_cells[rh][ch+1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_main_r;

                                            hist_cells[rh+1][ch+1].values[quantized_angle_upper] += upper_strength * lin_neighbor_c*lin_neighbor_r;
                                            hist_cells[rh+1][ch+1].values[quantized_angle_lower] += lower_strength * lin_neighbor_c*lin_neighbor_r;
                                        }
                                    }
                                }
                            }


                        }
                    }
                }
            }


            // Now figure out how many blocks we should have.  Note again that the hist_cells has a border of 
            // unused cells (thats where that -2 comes from).
            num_block_rows = (hist_cells.nr()-2 - (block_size-1) + cell_stride - 1)/cell_stride; 
            num_block_cols = (hist_cells.nc()-2 - (block_size-1) + cell_stride - 1)/cell_stride; 

        }

        unsigned long round_to_int(
            double val
        ) const
        {
            return static_cast<unsigned long>(std::floor(val + 0.5));
        }

        struct histogram
        {
            void zero()
            {
                for (unsigned long i = 0; i < num_orientation_bins; ++i)
                    values[i] = 0;
            }
            double values[num_orientation_bins];
        };

        typename array2d<histogram>::kernel_1a hist_cells;

        mutable descriptor_type des;

        long num_block_rows;
        long num_block_cols;


    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_HoG_H__

