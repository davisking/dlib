// Copyright (C) 2008  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_DRAW_IMAGe_ABSTRACT
#ifdef DLIB_DRAW_IMAGe_ABSTRACT

#include "../matrix.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_line (
        image_type& img,
        const point& p1,
        const point& p2,
        const pixel_type& val
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - #img.nr() == img.nr() && #img.nc() == img.nc()
              (i.e. the dimensions of the input image are not changed)
            - for all valid r and c that are on the line between point p1 and p2:
                - performs assign_pixel(img[r][c], val)
                  (i.e. it draws the line from p1 to p2 onto the image)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_line (
        long x1,
        long y1,
        long x2,
        long y2,
        image_type& img,
        const pixel_type& val
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
        ensures
            - performs draw_line(img, point(x1,y1), point(x2,y2), val)
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_rectangle (
        image_type& img,
        const rectangle& rect,
        const pixel_type& val,
        unsigned int thickness = 1
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<pixel_type> is defined
        ensures
            - Draws the given rectangle onto the image img.  It does this by calling
              draw_line() four times to draw the four sides of the rectangle.  
            - The rectangle is drawn with the color given by val.
            - The drawn rectangle will have edges that are thickness pixels wide.
    !*/

// ----------------------------------------------------------------------------------------

    struct string_dims
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a simple struct that represents the size (width and height) of a
                string in pixels.
        !*/

        string_dims() = default;
        string_dims (
            unsigned long width,
            unsigned long height
        ) : width(width), height(height) {}
        unsigned long width = 0;
        unsigned long height = 0;
    };

// ----------------------------------------------------------------------------------------

    template <
        typename T, typename traits,
        typename alloc
    >
    string_dims compute_string_dims (
        const std::basic_string<T, traits, alloc>& str,
        const std::shared_ptr<font>& f_ptr = default_font::get_font()
    )

    /*!
        ensures
            - computes the size of the given string with the specified font in pixels.  To be very specific,
              if dims is the returned object by this function, then:
              - dims.width == width of the string
              - dims.height == height of the string
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename T,
        typename traits,
        typename alloc,
        typename image_type,
        typename pixel_type
    >
    void draw_string (
        image_type& img,
        const dlib::point& p,
        const std::basic_string<T,traits,alloc>& str,
        const pixel_type& val,
        const std::shared_ptr<font>& f = default_font::get_font(),
        typename std::basic_string<T,traits,alloc>::size_type first = 0,
        typename std::basic_string<T,traits,alloc>::size_type last = (std::basic_string<T,traits,alloc>::npos)
    );

    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h
            - pixel_traits<pixel_type> is defined
        ensures
            - Draws the given string from first to last character onto the image img
              starting from point p.
            - The string is drawn with the color given by val and font f;
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_type,
        typename pixel_type
        >
    void draw_solid_circle (
        image_type& img,
        const dpoint& center_point,
        double radius,
        const pixel_type& pixel
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<pixel_type> is defined
        ensures
            - Draws a fully filled in circle onto image that is centered at center_point
              and has the given radius.  The circle will be filled by assigning the given
              pixel value to each element of the circle.
    !*/

// ----------------------------------------------------------------------------------------
    
    template <
        typename image_type,
        typename pixel_type
        >
    void fill_rect (
        image_type& img,
        const rectangle& rect,
        const pixel_type& pixel
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - pixel_traits<pixel_type> is defined
        ensures
            - fills the area defined by rect in the given image with the given pixel value.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename image_type, typename pixel_type>
    void draw_solid_convex_polygon (
        image_type& image,
        const polygon& poly,
        const pixel_type& color,
        const bool antialias = true,
        const rectangle& area = rectangle(std::numeric_limits<long>::min(), std::numeric_limits<long>::min(),
                                          std::numeric_limits<long>::max(), std::numeric_limits<long>::max())
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h
            - pixel_traits<pixel_type> is defined
        ensures
            - Assigns all pixels in the given polygon with the given color.  Moreover, some regions
              that are also in the convex hull of the polygon will also be filled as well.  In
              particular, any row of pixels falling horizontally between two parts of the polygon
              will be filled as well.
            - When drawing the polygon, only the part of the polygon which overlaps both
              the given image and area rectangle is drawn.
            - Uses the given pixel color to draw the polygon.
            - if (antialias == true) then we draw anti-aliased edges so they don't
              look all pixely by alpha-blending them with the image.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename image_type, typename pixel_type>
    void draw_solid_polygon (
        image_type& image,
        const polygon& poly,
        const pixel_type& color,
        const rectangle& area = rectangle(std::numeric_limits<long>::min(), std::numeric_limits<long>::min(),
                                          std::numeric_limits<long>::max(), std::numeric_limits<long>::max())
    );
    /*!
        requires
            - image_type == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h
            - pixel_traits<pixel_type> is defined
        ensures
            - Draws the given polygon onto the image, filling in the interior and edges of the
              polygon.
            - When drawing the polygon, only the part of the polygon which overlaps both
              the given image and area rectangle is drawn.
            - Uses the given pixel color to draw the polygon.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename image_array_type
        >
    matrix<typename image_traits<typename image_array_type::value_type>::pixel_type> tile_images (
        const image_array_type& images
    );
    /*!
        requires
            - image_array_type is a dlib::array of image objects where each image object
              implements the interface defined in dlib/image_processing/generic_image.h 
        ensures
            - This function takes the given images and tiles them into a single large
              square image and returns this new big tiled image.  Therefore, it is a useful
              method to visualize many small images at once.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_DRAW_IMAGe_ABSTRACT


