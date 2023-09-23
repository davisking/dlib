// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_GeNERIC_IMAGE_Hh_
#define DLIB_GeNERIC_IMAGE_Hh_

#include "../assert.h"
#include "../pixel.h"
#include <type_traits>

namespace dlib
{

    /*!
        In dlib, an "image" is any object that implements the generic image interface.  In
        particular, this simply means that an image type (let's refer to it as image_type
        from here on) has the following seven global functions defined for it:
            - long        num_rows      (const image_type& img)
            - long        num_columns   (const image_type& img)
            - void        set_image_size(      image_type& img, long rows, long cols)
            - void*       image_data    (      image_type& img)
            - const void* image_data    (const image_type& img)
            - long        width_step    (const image_type& img)
            - void        swap          (      image_type& a, image_type& b)
        And also provides a specialization of the image_traits template that looks like:
            namespace dlib
            {
                template <> 
                struct image_traits<image_type>
                {
                    typedef the_type_of_pixel_used_in_image_type pixel_type;
                };
            }

        Additionally, an image object must be default constructable.  This means that 
        expressions of the form:
            image_type img;
        Must be legal.

        Finally, the type of pixel in image_type must have a pixel_traits specialization.
        That is, pixel_traits<typename image_traits<image_type>::pixel_type> must be one of
        the specializations of pixel_traits.  
        
        
        To be very precise, the seven functions defined above are defined thusly:

            long num_rows(
                const image_type& img
            ); 
            /!*
                ensures
                    - returns the number of rows in the given image
            *!/

            long num_columns(
                const image_type& img
            );
            /!*
                ensures
                    - returns the number of columns in the given image
            *!/

            void set_image_size(
                image_type& img,
                long rows,
                long cols 
            );
            /!*
                requires
                    - rows >= 0 && cols >= 0
                ensures
                    - num_rows(#img) == rows
                    - num_columns(#img) == cols
            *!/

            void* image_data(
                image_type& img
            );
            /!*
                ensures
                    - returns a non-const pointer to the pixel at row and column position 0,0
                      in the given image.  Or if the image has zero rows or columns in it
                      then this function returns NULL.
                    - The image lays pixels down in row major order.  However, there might
                      be padding at the end of each row.  The amount of padding is given by
                      width_step(img).
            *!/

            const void* image_data(
                const image_type& img
            );
            /!*
                ensures
                    - returns a const pointer to the pixel at row and column position 0,0 in
                      the given image.  Or if the image has zero rows or columns in it then
                      this function returns NULL.
                    - The image lays pixels down in row major order.  However, there might
                      be padding at the end of each row.  The amount of padding is given by
                      width_step(img).
            *!/

            long width_step(
                const image_type& img
            );
            /!*
                ensures
                    - returns the size of one row of the image, in bytes.  More precisely,
                      return a number N such that: (char*)image_data(img) + N*R == a
                      pointer to the first pixel in the R-th row of the image. This means
                      that the image must lay its pixels down in row major order.
            *!/

            void swap(
                image_type& a,
                image_type& b
            );
            /!*
                ensures
                    - swaps the state of a and b
            *!/
    !*/

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    struct image_traits;
    /*!
        WHAT THIS OBJECT REPRESENTS
            This is a traits class for generic image objects.  You can use it to find out
            the pixel type contained within an image via an expression of the form:
                image_traits<image_type>::pixel_type
    !*/

    /*!A pixel_type_t 
        An alias for the type of pixel in an image.
    !*/
    template <typename image_type>
    using pixel_type_t = typename image_traits<image_type>::pixel_type;

    /*!A is_rgb_image
        A type traits class telling you if a type is an image holding RGB pixels.
    !*/
    template <typename image_type>
    struct is_rgb_image { const static bool value = pixel_traits<pixel_type_t<image_type>>::rgb; };

    /*!A is_color_space_cartesian_image
        A type traits class telling you if a type is an image holding some type of cartesian pixel type.

        E.g. as contrasted with polar coordinates pixel types.
    !*/
    template <typename image_type>
    struct is_color_space_cartesian_image { const static bool value = 
        pixel_traits<pixel_type_t<image_type>>::rgb || 
        pixel_traits<pixel_type_t<image_type>>::lab || 
        pixel_traits<pixel_type_t<image_type>>::grayscale; };

    /*!A is_grayscale_image
        A type traits class telling you if a type is an image holding a single channel (i.e.
        grayscale) pixel type.
    !*/
    template <typename image_type>
    struct is_grayscale_image { const static bool value = pixel_traits<pixel_type_t<image_type>>::grayscale; };

// ----------------------------------------------------------------------------------------

    namespace details
    {
        template<class Container, class Alwaysvoid = void>
        struct is_image_type : std::false_type{};

        template<class Container>
        struct is_image_type<Container, dlib::void_t<is_pixel_check<pixel_type_t<Container>>>> : std::true_type{};
    }

    /*!A is_image_type 

        A type traits struct telling you if Container satisfies the generic image interface.

        i.e. there exists an image_traits<> specialization and the underlying pixel type has a pixel_trait<> specialiation
        e.g. array2d<rgb_pixel>, matrix<float>, etc...
    !*/
    template<class Container>
    using is_image_type = details::is_image_type<Container>;

    /*!A is_image_check

        This is a SFINAE tool for restricting a template to only image types.
    !*/
    template<class Container>
    using is_image_check = std::enable_if_t<is_image_type<Container>::value, bool>;

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//                   UTILITIES TO MAKE ACCESSING IMAGE PIXELS SIMPLER
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <
        typename image_type
        >
    class image_view
    {
        /*!
            REQUIREMENTS ON image_type
                image_type must be an image object as defined at the top of this file.  

            WHAT THIS OBJECT REPRESENTS
                This object takes an image object and wraps it with an interface that makes
                it look like a dlib::array2d.  That is, it makes it look similar to a
                regular 2-dimensional C style array, making code which operates on the
                pixels simple to read.

                Note that an image_view instance is valid until the image given to its
                constructor is modified through an interface other than the image_view
                instance.  This is because, for example, someone might cause the underlying
                image object to reallocate its memory, thus invalidating the pointer to its
                pixel data stored in the image_view.    

                As an side, the reason why this object stores a pointer to the image
                object's data and uses that pointer instead of calling image_data() each
                time a pixel is accessed is to allow for image objects to implement
                complex, and possibly slow, image_data() functions.  For example, an image
                object might perform some kind of synchronization between a GPU and the
                host memory during a call to image_data().  Therefore, we call image_data()
                only in image_view's constructor to avoid the performance penalty of
                calling it for each pixel access.
        !*/

    public:
        using pixel_type = pixel_type_t<image_type>;

        image_view(
            image_type& img
        ) : 
            _data(reinterpret_cast<char*>(image_data(img))), 
            _width_step(width_step(img)),
            _nr(num_rows(img)),
            _nc(num_columns(img)),
            _img(&img) 
        {}

        long nr() const { return _nr; }
        /*!
            ensures
                - returns the number of rows in this image.
        !*/

        long nc() const { return _nc; }
        /*!
            ensures
                - returns the number of columns in this image.
        !*/

        unsigned long size() const { return static_cast<unsigned long>(nr()*nc()); }
        /*!
            ensures
                - returns the number of pixels in this image.
        !*/

#ifndef ENABLE_ASSERTS
        pixel_type* operator[] (long row) { return (pixel_type*)(_data+_width_step*row); }
        /*!
            requires
                - 0 <= row < nr()
            ensures
                - returns a pointer to the first pixel in the row-th row.  Therefore, the
                  pixel at row and column position r,c can be accessed via (*this)[r][c].
        !*/

        const pixel_type* operator[] (long row) const { return (const pixel_type*)(_data+_width_step*row); }
        /*!
            requires
                - 0 <= row < nr()
            ensures
                - returns a const pointer to the first pixel in the row-th row.  Therefore,
                  the pixel at row and column position r,c can be accessed via
                  (*this)[r][c].
        !*/
#else
        // If asserts are enabled then we need to return a proxy class so we can make sure
        // the column accesses don't go out of bounds.
        struct pix_row
        {
            pix_row(pixel_type* data_, long nc_) : data(data_),_nc(nc_) {}
            const pixel_type& operator[] (long col) const
            {
                DLIB_ASSERT(0 <= col && col < _nc, 
                    "\t The given column index is out of range."
                    << "\n\t col: " << col 
                    << "\n\t _nc: " << _nc);
                return data[col];
            }
            pixel_type& operator[] (long col)
            {
                DLIB_ASSERT(0 <= col && col < _nc, 
                    "\t The given column index is out of range."
                    << "\n\t col: " << col 
                    << "\n\t _nc: " << _nc);
                return data[col];
            }
        private:
            pixel_type* const data;
            const long _nc;
        };
        pix_row operator[] (long row) 
        { 
            DLIB_ASSERT(0 <= row && row < _nr, 
                "\t The given row index is out of range."
                << "\n\t row: " << row 
                << "\n\t _nr: " << _nr);
            return pix_row((pixel_type*)(_data+_width_step*row), _nc); 
        }
        const pix_row operator[] (long row) const 
        { 
            DLIB_ASSERT(0 <= row && row < _nr, 
                "\t The given row index is out of range."
                << "\n\t row: " << row 
                << "\n\t _nr: " << _nr);
            return pix_row((pixel_type*)(_data+_width_step*row), _nc); 
        }
#endif

        void set_size(long rows, long cols) 
        /*!
            requires
                - rows >= 0 && cols >= 0
            ensures
                - Tells the underlying image to resize itself to have the given number of
                  rows and columns.
                - #nr() == rows
                - #nc() == cols
        !*/
        { 
            DLIB_ASSERT((cols >= 0 && rows >= 0),
                        "\t image_view::set_size(long rows, long cols)"
                        << "\n\t The images can't have negative rows or columns."
                        << "\n\t cols: " << cols 
                        << "\n\t rows: " << rows 
            );
            set_image_size(*_img, rows, cols); *this = *_img; 
        }

        void clear() { set_size(0,0); }
        /*!
            ensures
                - sets the image to have 0 pixels in it.
        !*/

        long get_width_step() const { return _width_step; }

    private:

        char* _data;
        long _width_step;
        long _nr;
        long _nc;
        image_type* _img;
    };

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    class const_image_view
    {
        /*!
            REQUIREMENTS ON image_type
                image_type must be an image object as defined at the top of this file.  

            WHAT THIS OBJECT REPRESENTS
                This object is just like the image_view except that it provides a "const"
                view into an image.  That is, it has the same interface as image_view
                except that you can't modify the image through a const_image_view.
        !*/

    public:
        using pixel_type = pixel_type_t<image_type>;

        const_image_view(
            const image_type& img
        ) : 
            _data(reinterpret_cast<const char*>(image_data(img))), 
            _width_step(width_step(img)),
            _nr(num_rows(img)),
            _nc(num_columns(img))
        {}

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        unsigned long size() const { return static_cast<unsigned long>(nr()*nc()); }
#ifndef ENABLE_ASSERTS
        const pixel_type* operator[] (long row) const { return (const pixel_type*)(_data+_width_step*row); }
#else
        // If asserts are enabled then we need to return a proxy class so we can make sure
        // the column accesses don't go out of bounds.
        struct pix_row
        {
            pix_row(pixel_type* data_, long nc_) : data(data_),_nc(nc_) {}
            const pixel_type& operator[] (long col) const
            {
                DLIB_ASSERT(0 <= col && col < _nc, 
                    "\t The given column index is out of range."
                    << "\n\t col: " << col 
                    << "\n\t _nc: " << _nc);
                return data[col];
            }
        private:
            pixel_type* const data;
            const long _nc;
        };
        const pix_row operator[] (long row) const 
        { 
            DLIB_ASSERT(0 <= row && row < _nr, 
                "\t The given row index is out of range."
                << "\n\t row: " << row 
                << "\n\t _nr: " << _nr);
            return pix_row((pixel_type*)(_data+_width_step*row), _nc); 
        }
#endif

        long get_width_step() const { return _width_step; }

    private:
        const char* _data;
        long _width_step;
        long _nr;
        long _nc;
    };

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    image_view<image_type> make_image_view ( image_type& img) 
    { return image_view<image_type>(img); }
    /*!
        requires
            - image_type == an image object that implements the interface defined at the
              top of this file.
        ensures
            - constructs an image_view from an image object
    !*/

    template <typename image_type>
    const_image_view<image_type> make_image_view (const image_type& img) 
    { return const_image_view<image_type>(img); }
    /*!
        requires
            - image_type == an image object that implements the interface defined at the
              top of this file.
        ensures
            - constructs a const_image_view from an image object
    !*/


    // Don't stack image views on image views since that's pointless and just slows the
    // compilation.
    template <typename T> image_view<T>&             make_image_view ( image_view<T>& img)             { return img; }
    template <typename T> const image_view<T>&       make_image_view ( const image_view<T>& img)       { return img; }
    template <typename T> const_image_view<T>&       make_image_view ( const_image_view<T>& img)       { return img; }
    template <typename T> const const_image_view<T>& make_image_view ( const const_image_view<T>& img) { return img; }

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    inline unsigned long image_size(
        const image_type& img
    ) { return num_columns(img)*num_rows(img); }
    /*!
        requires
            - image_type == an image object that implements the interface defined at the
              top of this file.
        ensures
            - returns the number of pixels in the given image.
    !*/

// ----------------------------------------------------------------------------------------

    template <typename image_type>
    inline long num_rows(
        const image_type& img
    ) { return img.nr(); }
    /*!
        ensures
            - By default, try to use the member function .nr() to determine the number
              of rows in an image.  However, as stated at the top of this file, image
              objects should provide their own overload of num_rows() if needed.
    !*/

    template <typename image_type>
    inline long num_columns(
        const image_type& img
    ) { return img.nc(); }
    /*!
        ensures
            - By default, try to use the member function .nc() to determine the number
              of columns in an image.  However, as stated at the top of this file, image
              objects should provide their own overload of num_rows() if needed.
    !*/

    template <typename image_type1, typename image_type2>
    typename std::enable_if<is_image_type<image_type1>::value&&is_image_type<image_type2>::value, bool>::type 
    have_same_dimensions (
        const image_type1& img1,
        const image_type2& img2
    ) { return num_rows(img1)==num_rows(img2) && num_columns(img1)==num_columns(img2); }
    /*!
        ensures
            - returns true if and only if the two given images have the same dimensions.
    !*/

    template <typename image_type1, typename image_type2, typename ...T>
    typename std::enable_if<is_image_type<image_type1>::value&&is_image_type<image_type2>::value, bool>::type 
    have_same_dimensions (
        const image_type1& img1,
        const image_type2& img2,
        T&& ...args
    ) { return have_same_dimensions(img1,img2) && have_same_dimensions(img1,args...); }

// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------
//            Make the image views implement the generic image interface
// ----------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------

    template <typename T>
    struct image_traits<image_view<T>> { using pixel_type = pixel_type_t<T>; };

    template <typename T>
    struct image_traits<const image_view<T>> { using pixel_type = pixel_type_t<T>; };

    template <typename T>
    inline long num_rows( const image_view<T>& img) { return img.nr(); }
    template <typename T>
    inline long num_columns( const image_view<T>& img) { return img.nc(); }

    template <typename T>
    inline void set_image_size( image_view<T>& img, long rows, long cols ) { img.set_size(rows,cols); }

    template <typename T>
    inline void* image_data( image_view<T>& img)
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline const void* image_data(
        const image_view<T>& img
    )
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline long width_step( const image_view<T>& img) { return img.get_width_step(); }

// ----------------------------------------------------------------------------------------

    template <typename T>
    struct image_traits<const_image_view<T>> {using pixel_type = pixel_type_t<T>; };

    template <typename T>
    struct image_traits<const const_image_view<T>> {using pixel_type = pixel_type_t<T>; };

    template <typename T>
    inline long num_rows( const const_image_view<T>& img) { return img.nr(); }
    template <typename T>
    inline long num_columns( const const_image_view<T>& img) { return img.nc(); }

    template <typename T>
    inline const void* image_data(
        const const_image_view<T>& img
    )
    {
        if (img.size() != 0)
            return &img[0][0];
        else
            return 0;
    }

    template <typename T>
    inline long width_step( const const_image_view<T>& img) { return img.get_width_step(); }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_GeNERIC_IMAGE_Hh_

