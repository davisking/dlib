// Copyright (C) 2010  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#undef DLIB_IMAGE_PYRaMID_ABSTRACT_Hh_
#ifdef DLIB_IMAGE_PYRaMID_ABSTRACT_Hh_

#include "../pixel.h"
#include "../array2d.h"
#include "../geometry.h"
#include "../image_processing/generic_image.h"

namespace dlib
{

    template <
        unsigned int N
        >
    class pyramid_down : noncopyable
    {
        /*!
            REQUIREMENTS ON N
                N > 0

            WHAT THIS OBJECT REPRESENTS
                This is a simple functor to help create image pyramids.  In particular, it
                downsamples images at a ratio of N to N-1.

                Note that setting N to 1 means that this object functions like
                pyramid_disable (defined at the bottom of this file).  

                WARNING, when mapping rectangles from one layer of a pyramid
                to another you might end up with rectangles which extend slightly 
                outside your images.  This is because points on the border of an 
                image at a higher pyramid layer might correspond to points outside 
                images at lower layers.  So just keep this in mind.  Note also
                that it's easy to deal with.  Just say something like this:
                    rect = rect.intersect(get_rect(my_image)); // keep rect inside my_image 
        !*/
    public:

        template <
            typename in_image_type,
            typename out_image_type
            >
        void operator() (
            const in_image_type& original,
            out_image_type& down
        ) const;
        /*!
            requires
                - is_same_object(original, down) == false
                - in_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - out_image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - for both pixel types P in the input and output images, we require:
                    - pixel_traits<P>::has_alpha == false
            ensures
                - #down will contain an image that is roughly (N-1)/N times the size of the
                  original image.  
                - If both input and output images contain RGB pixels then the downsampled image will
                  be in color.  Otherwise, the downsampling will be performed in a grayscale mode.
                - The location of a point P in original image will show up at point point_down(P)
                  in the #down image.  
                - Note that some points on the border of the original image might correspond to 
                  points outside the #down image.  
        !*/

        template <
            typename image_type
            >
        void operator() (
            image_type& img
        ) const;
        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - pixel_traits<typename image_traits<image_type>::pixel_type>::has_alpha == false
            ensures
                - This function downsamples the given image and stores the results in #img.
                  In particular, it is equivalent to performing: 
                    (*this)(img, temp); 
                    swap(img, temp);
        !*/

    // -------------------------------

        template <typename T>
        vector<double,2> point_down (
            const vector<T,2>& p
        ) const;
        /*!
            ensures
                - interprets p as a point in a parent image and returns the
                  point in a downsampled image which corresponds to p.
                - This function is the inverse of point_up().  I.e. for a point P:
                  point_down(point_up(P)) == P
        !*/

        template <typename T>
        vector<double,2> point_up (
            const vector<T,2>& p
        ) const;
        /*!
            ensures
                - interprets p as a point in a downsampled image and returns the
                  point in a parent image which corresponds to p.
                - This function is the inverse of point_down().  I.e. for a point P:
                  point_up(point_down(P)) == P
        !*/

        drectangle rect_down (
            const drectangle& rect
        ) const;
        /*!
            ensures
                - returns drectangle(point_down(rect.tl_corner()), point_down(rect.br_corner()));
                  (i.e. maps rect into a downsampled)
        !*/

        drectangle rect_up (
            const drectangle& rect
        ) const;
        /*!
            ensures
                - returns drectangle(point_up(rect.tl_corner()), point_up(rect.br_corner()));
                  (i.e. maps rect into a parent image)
        !*/

    // -------------------------------

        template <typename T>
        vector<double,2> point_down (
            const vector<T,2>& p,
            unsigned int levels
        ) const;
        /*!
            ensures
                - applies point_down() to p levels times and returns the result.
                  (i.e. point_down(p,2) == point_down(point_down(p)),
                        point_down(p,1) == point_down(p),
                        point_down(p,0) == p,  etc. )
        !*/

        template <typename T>
        vector<double,2> point_up (
            const vector<T,2>& p,
            unsigned int levels
        ) const;
        /*!
            ensures
                - applies point_up() to p levels times and returns the result.
                  (i.e. point_up(p,2) == point_up(point_up(p)),
                        point_up(p,1) == point_up(p),
                        point_up(p,0) == p,  etc. )
        !*/

        drectangle rect_down (
            const drectangle& rect,
            unsigned int levels
        ) const;
        /*!
            ensures
                - returns drectangle(point_down(rect.tl_corner(),levels), point_down(rect.br_corner(),levels));
                  (i.e. Basically applies rect_down() to rect levels times and returns the result.)
        !*/

        drectangle rect_up (
            const drectangle& rect,
            unsigned int levels
        ) const;
        /*!
            ensures
                - returns drectangle(point_up(rect.tl_corner(),levels), point_up(rect.br_corner(),levels));
                  (i.e. Basically applies rect_up() to rect levels times and returns the result.)
        !*/

    };

// ----------------------------------------------------------------------------------------

    class pyramid_disable : noncopyable
    {
        /*!
            WHAT THIS OBJECT REPRESENTS
                This is a function object with an interface identical to pyramid_down (defined
                at the top of this file) except that it downsamples images at a ratio of infinity
                to 1.  That means it always outputs images of size 0 regardless of the size
                of the inputs.  
                
                This is useful because it can be supplied to routines which take a pyramid_down 
                function object and it will essentially disable pyramid processing.  This way, 
                a pyramid oriented function can be turned into a regular routine which processes
                just the original undownsampled image.
        !*/
    };

// ----------------------------------------------------------------------------------------

    template <
        unsigned int N
        >
    double pyramid_rate(
        const pyramid_down<N>& pyr
    );
    /*!
        ensures
            - returns (N-1.0)/N
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename pyramid_type,
        typename image_type1,
        typename image_type2
        >
    void create_tiled_pyramid (
        const image_type1& img,
        image_type2& out_img,
        std::vector<rectangle>& rects,
        const unsigned long padding = 10,
        const unsigned long outer_padding = 0
    );
    /*!
        requires
            - pyramid_type == one of the dlib::pyramid_down template instances defined above.
            - is_same_object(img, out_img) == false
            - image_type1 == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - image_type2 == an image object that implements the interface defined in
              dlib/image_processing/generic_image.h 
            - for both pixel types P in the input and output images, we require:
                - pixel_traits<P>::has_alpha == false
        ensures
            - Creates an image pyramid from the input image img.  The pyramid is made using
              pyramid_type.  The highest resolution image is img and then all further
              pyramid levels are generated from pyramid_type's downsampling.  The entire
              resulting pyramid is packed into a single image and stored in out_img.
            - When packing pyramid levels into out_img, there will be padding pixels of
              space between each sub-image.  There will also be outer_padding pixels of
              padding around the edge of the image.  All padding pixels have a value of 0.
            - The resulting pyramid will be composed of #rects.size() images packed into
              out_img.  Moreover, #rects[i] is the location inside out_img of the i-th
              pyramid level. 
            - #rects.size() > 0
            - #rects[0] == get_rect(img).  I.e. the first rectangle is the highest
              resolution pyramid layer.  Subsequent elements of #rects correspond to
              smaller and smaller pyramid layers inside out_img.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename pyramid_type
        >
    dpoint image_to_tiled_pyramid (
        const std::vector<rectangle>& rects,
        double scale,
        dpoint p
    );
    /*!
        requires
            - pyramid_type == one of the dlib::pyramid_down template instances defined above.
            - 0 < scale <= 1
            - rects.size() > 0
        ensures
            - The function create_tiled_pyramid() converts an image, img, to a "tiled
              pyramid" called out_img.  It also outputs a vector of rectangles, rect, that
              show where each pyramid layer appears in out_img.   Therefore,
              image_to_tiled_pyramid() allows you to map from coordinates in img (i.e. p)
              to coordinates in the tiled pyramid out_img, when given the rects metadata.  

              So given a point p in img, you can ask, what coordinate in out_img
              corresponds to img[p.y()][p.x()] when things are scale times smaller?  This
              new coordinate is a location in out_img and is what is returned by this
              function.
            - A scale of 1 means we don't move anywhere in the pyramid scale space relative
              to the input image while smaller values of scale mean we move down the
              pyramid.
            - Assumes pyramid_type is the pyramid class used to produce the tiled image.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename pyramid_type
        >
    drectangle image_to_tiled_pyramid (
        const std::vector<rectangle>& rects,
        double scale,
        drectangle r
    );
    /*!
        requires
            - pyramid_type == one of the dlib::pyramid_down template instances defined above.
            - 0 < scale <= 1
            - rects.size() > 0
        ensures
            - This function maps from input image space to tiled pyramid coordinate space
              just as the above image_to_tiled_pyramid() does, except it operates on
              rectangle objects instead of points.
            - Assumes pyramid_type is the pyramid class used to produce the tiled image.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename pyramid_type
        >
    dpoint tiled_pyramid_to_image (
        const std::vector<rectangle>& rects,
        dpoint p
    );
    /*!
        requires
            - pyramid_type == one of the dlib::pyramid_down template instances defined above.
            - rects.size() > 0
        ensures
            - This function maps from a coordinate in a tiled pyramid to the corresponding
              input image coordinate.  Therefore, it is essentially the inverse of
              image_to_tiled_pyramid().
            - It should be noted that this function isn't always an inverse of
              image_to_tiled_pyramid().  This is because you can ask
              image_to_tiled_pyramid() for the coordinates of points outside the input
              image and they will be mapped to somewhere that doesn't have an inverse.  But
              for points actually inside the image this function performs an approximate
              inverse mapping.
            - Assumes pyramid_type is the pyramid class used to produce the tiled image.
    !*/

// ----------------------------------------------------------------------------------------

    template <
        typename pyramid_type
        >
    drectangle tiled_pyramid_to_image (
        const std::vector<rectangle>& rects,
        drectangle r 
    );
    /*!
        requires
            - pyramid_type == one of the dlib::pyramid_down template instances defined above.
            - rects.size() > 0
        ensures
            - This function maps from a coordinate in a tiled pyramid to the corresponding
              input image coordinate.  Therefore, it is essentially the inverse of
              image_to_tiled_pyramid().
            - It should be noted that this function isn't always an inverse of
              image_to_tiled_pyramid().  This is because you can ask
              image_to_tiled_pyramid() for the coordinates of points outside the input
              image and they will be mapped to somewhere that doesn't have an inverse.  But
              for points actually inside the image this function performs an approximate
              inverse mapping.
            - Assumes pyramid_type is the pyramid class used to produce the tiled image.
    !*/

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_IMAGE_PYRaMID_ABSTRACT_Hh_


