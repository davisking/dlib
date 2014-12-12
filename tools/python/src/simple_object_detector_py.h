// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMPLE_OBJECT_DETECTOR_PY_H__
#define DLIB_SIMPLE_OBJECT_DETECTOR_PY_H__

#include <dlib/python.h>
#include <dlib/matrix.h>
#include <boost/python/args.hpp>
#include <dlib/geometry.h>
#include <dlib/image_processing/frontal_face_detector.h>

namespace dlib
{
    typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > simple_object_detector;

    inline std::vector<dlib::rectangle> run_detector_with_upscale (
        dlib::simple_object_detector& detector,
        boost::python::object img,
        const unsigned int upsampling_amount
    )
    {
        pyramid_down<2> pyr;

        if (is_gray_python_image(img))
        {
            array2d<unsigned char> temp;
            if (upsampling_amount == 0)
            {
                return detector(numpy_gray_image(img));
            }
            else
            {
                pyramid_up(numpy_gray_image(img), temp, pyr);
                unsigned int levels = upsampling_amount-1;
                while (levels > 0)
                {
                    levels--;
                    pyramid_up(temp);
                }

                std::vector<rectangle> res = detector(temp);
                for (unsigned long i = 0; i < res.size(); ++i)
                    res[i] = pyr.rect_down(res[i], upsampling_amount);
                return res;
            }
        }
        else if (is_rgb_python_image(img))
        {
            array2d<rgb_pixel> temp;
            if (upsampling_amount == 0)
            {
                return detector(numpy_rgb_image(img));
            }
            else
            {
                pyramid_up(numpy_rgb_image(img), temp, pyr);
                unsigned int levels = upsampling_amount-1;
                while (levels > 0)
                {
                    levels--;
                    pyramid_up(temp);
                }

                std::vector<rectangle> res = detector(temp);
                for (unsigned long i = 0; i < res.size(); ++i)
                    res[i] = pyr.rect_down(res[i], upsampling_amount);
                return res;
            }
        }
        else
        {
            throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
        }
    }

    struct simple_object_detector_py
    {
        simple_object_detector detector;
        unsigned int upsampling_amount;

        simple_object_detector_py() {}
        simple_object_detector_py(simple_object_detector& _detector, unsigned int _upsampling_amount) :
            detector(_detector), upsampling_amount(_upsampling_amount) {}

        std::vector<dlib::rectangle> run_detector1 (boost::python::object img, const unsigned int upsampling_amount_)
        { return run_detector_with_upscale(detector, img, upsampling_amount_); }

        std::vector<dlib::rectangle> run_detector2 (boost::python::object img)
        { return run_detector_with_upscale(detector, img, upsampling_amount); }
    };
}

#endif // DLIB_SIMPLE_OBJECT_DETECTOR_PY_H__
