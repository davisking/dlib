// Copyright (C) 2009  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_CvIMAGE_H_
#define DLIB_CvIMAGE_H_

#include "cv_image_abstract.h"
#include "../algs.h"

namespace dlib
{

    template <
        typename pixel_type
        >
    class cv_image
    {
    public:
        typedef pixel_type type;
        typedef default_memory_manager mem_manager_type;

        cv_image (const IplImage* img) 
        {
            check_image_type(img);
            _data = img->imageData;
            _widthStep = img->widthStep;
            _nr = img->height;
            _nc = img->width;

        }
        cv_image() : _data(0), _widthStep(0), _nr(0), _nc(0) {}

        unsigned long size () const { return static_cast<unsigned long>(_nr*_nc); }

        inline pixel_type* operator[](const long row ) 
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(0 <= row && row < nr(),
                "\tpixel_type* operator[](row)"
                << "\n\t you have asked for an out of bounds row " 
                << "\n\t row:  " << row
                << "\n\t nr(): " << nr() 
                << "\n\t this:  " << this
                );

            return reinterpret_cast<pixel_type*>( _data + _widthStep*row);
        }

        inline const pixel_type* operator[](const long row ) const
        { 
            // make sure requires clause is not broken
            DLIB_ASSERT(0 <= row && row < nr(),
                "\tconst pixel_type* operator[](row)"
                << "\n\t you have asked for an out of bounds row " 
                << "\n\t row:  " << row
                << "\n\t nr(): " << nr() 
                << "\n\t this:  " << this
                );

            return reinterpret_cast<const pixel_type*>( _data + _widthStep*row);
        }

        long nr() const { return _nr; }
        long nc() const { return _nc; }
        long width_step() const { return _widthStep; }

        cv_image& operator=( const cv_image& item)
        {
            _data = item._data;
            _widthStep = item._widthStep;
            _nr = item._nr;
            _nc = item._nc;
            return *this;
        }

        cv_image& operator=( const IplImage* img)
        {
            check_image_type(img);
            _data = img->imageData;
            _widthStep = img->widthStep;
            _nr = img->height;
            _nc = img->width;
            return *this;
        }

    private:

        inline void check_image_type (const IplImage* 
#ifdef ENABLE_ASSERTS
                                      img // the #ifdef is here to avoid an unused warning argument from the compiler
#endif
                                      ) const
        {
            DLIB_ASSERT( img->dataOrder == 0, "Only interleaved color channels are supported with cv_image"); 
            DLIB_ASSERT((img->depth&0xFF)/8*img->nChannels == sizeof(pixel_type), 
                         "The pixel type you gave doesn't match the size of pixel used by the open cv image struct");
   
        }

        char* _data;
        long _widthStep;
        long _nr;
        long _nc;
    };

}

#endif // DLIB_CvIMAGE_H_

