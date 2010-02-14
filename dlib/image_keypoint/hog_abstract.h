// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_HoG_ABSTRACT_H__
#ifdef DLIB_HoG_ABSTRACT_H__

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
        /*!
            REQUIREMENTS ON TEMPLATE PARAMETERS 
                - cell_size_ > 1
                - block_size_ > 0
                - cell_stride_ > 0
                - num_orientation_bins_ > 0
                - gradient_type_ == hog_signed_gradient or hog_unsigned_gradient
                - interpolation_type_ == hog_no_interpolation, hog_angle_interpolation, or 
                                         hog_full_interpolation

            INITIAL VALUE
                 - size() == 0

            WHAT THIS OBJECT REPRESENTS
                This object is a tool for performing the image feature extraction algorithm
                described in the following paper:
                    Histograms of Oriented Gradients for Human Detection
                    by Navneet Dalal and Bill Triggs

                
                To summarize the technique, this object tiles non-overlapping cells over an 
                image.  Each of these cells is a box that is cell_size by cell_size pixels 
                in size.  Each cell contains an array of size num_orientation_bins.  The array 
                in a cell is used to store a histogram of all the edge orientations contained
                within the cell's image region.  

                Once the grid of cells and their histograms has been computed (via load())
                you can obtain descriptors for each "block" in the image.  A block is just a
                group of cells and blocks are allowed to overlap.  Each block is square and
                made up of block_size*block_size cells.  So when you call operator()(r,c)
                what you obtain is a vector that is just a bunch of cell histograms that
                have been concatenated (and length normalized).

                The template arguments control the various parameters of this algorithm.

                The interpolation_type parameter controls the amount of interpolation
                that happens during the creation of the edge orientation histograms.  It
                varies from no interpolation at all to full spatial and angle interpolation.
                
                Angle interpolation means that an edge doesn't just go into its nearest 
                histogram bin but instead gets interpolated into its two nearest neighbors.
                Similarly, spatial interpolation means that an edge doesn't just go into
                the cell it is in but it also contributes to nearby cells depending on how
                close they are.  

                The gradient_type parameter controls how edge orientations are measured.  
                Consider the following ASCII art:
                    signed gradients:           unsigned gradients:
                           /\                           |
                           ||                           |
                       <---  ---->                ------+------
                           ||                           |
                           \/                           |

                An image is full of gradients caused by edges between objects.  The direction 
                of a gradient is determined by which end of it has pixels of highest intensity.  
                So for example, suppose you had a picture containing black and white stripes.  
                Then the magnitude of the gradient at each point in the image tells you if you 
                are on the edge of a stripe and the gradient's orientation tells you which 
                direction you have to move get into the white stripe.   

                Signed gradients preserve this direction information while unsigned gradients
                do not.  An unsigned gradient will only tell you the orientation of the stripe
                but not which direction leads to the white stripe.   

                Finally, the cell_stride parameter controls how much overlap you get between
                blocks.  The maximum amount of overlap is obtained when cell_stride == 1.
                At the other extreme, you would have no overlap if cell_stride == block_size. 
        !*/

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
        );
        /*!
            ensures
                - this object is properly initialized
        !*/

        template <
            typename image_type
            >
        inline void load (
            const image_type& img
        );
        /*!
            requires
                - image_type is a dlib::matrix or something convertible to a matrix
                  via array_to_matrix()
                - pixel_traits<typename image_type::type>::has_alpha == false
            ensures
                - if (img.nr() < min_size || img.nc() < min_size) then
                    - the image is too small so we don't compute anything on it
                    - #size() == 0
                - else
                    - generates a HOG image from the given image.   
                    - #size() > 0
        !*/

        inline unsigned long size (
        ) const;
        /*!
            ensures
                - returns nr()*nc()
        !*/

        inline long nr (
        ) const;
        /*!
            ensures
                - returns the number of rows in this HOG image
        !*/

        inline long nc (
        ) const;
        /*!
            ensures
                - returns the number of columns in this HOG image
        !*/

        inline const descriptor_type& operator() (
            long row,
            long col
        ) const;
        /*!
            requires
                - 0 <= row < nr()
                - 0 <= col < nc()
            ensures
                - returns the descriptor for the HOG block at the given row and column.  This descriptor 
                  will include information from a window that is located at get_block_rect(row,col) in
                  the original image given to load().
                - The returned descriptor vector will have block_size*block_size*num_orientation_bins elements.
        !*/

        const rectangle get_block_rect (
            long row,
            long col
        ) const;
        /*!
            requires
                - 0 <= row < nr()
                - 0 <= col < nc()
            ensures
                - returns a rectangle that tells you what part of the original image is associated
                  with a particular HOG block.
                - The returned rectangle will be cell_size*block_size pixels wide and tall.
        !*/
    };

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_HoG_ABSTRACT_H__


