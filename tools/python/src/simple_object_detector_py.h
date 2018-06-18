// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SIMPLE_OBJECT_DETECTOR_PY_H__
#define DLIB_SIMPLE_OBJECT_DETECTOR_PY_H__

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry.h>
#include <dlib/image_processing/frontal_face_detector.h>

namespace py = pybind11;

namespace dlib
{
    typedef object_detector<scan_fhog_pyramid<pyramid_down<6> > > simple_object_detector;

    inline void split_rect_detections (
        std::vector<rect_detection>& rect_detections,
        std::vector<rectangle>& rectangles,
        std::vector<double>& detection_confidences,
        std::vector<unsigned long>& weight_indices
    )
    {
        rectangles.clear();
        detection_confidences.clear();
        weight_indices.clear();

        for (unsigned long i = 0; i < rect_detections.size(); ++i)
        {
            rectangles.push_back(rect_detections[i].rect);
            detection_confidences.push_back(rect_detections[i].detection_confidence);
            weight_indices.push_back(rect_detections[i].weight_index);
        }
    }


    inline std::vector<dlib::rectangle> run_detector_with_upscale1 (
        dlib::simple_object_detector& detector,
        py::array img,
        const unsigned int upsampling_amount,
        const double adjust_threshold,
        std::vector<double>& detection_confidences,
        std::vector<unsigned long>& weight_indices
    )
    {
        pyramid_down<2> pyr;

        std::vector<rectangle> rectangles;
        std::vector<rect_detection> rect_detections;

        if (is_image<unsigned char>(img))
        {
            array2d<unsigned char> temp;
            if (upsampling_amount == 0)
            {
                detector(numpy_image<unsigned char>(img), rect_detections, adjust_threshold);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);
                return rectangles;
            }
            else
            {
                pyramid_up(numpy_image<unsigned char>(img), temp, pyr);
                unsigned int levels = upsampling_amount-1;
                while (levels > 0)
                {
                    levels--;
                    pyramid_up(temp);
                }

                detector(temp, rect_detections, adjust_threshold);
                for (unsigned long i = 0; i < rect_detections.size(); ++i)
                    rect_detections[i].rect = pyr.rect_down(rect_detections[i].rect,
                                                            upsampling_amount);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);

                return rectangles;
            }
        }
        else if (is_image<rgb_pixel>(img))
        {
            array2d<rgb_pixel> temp;
            if (upsampling_amount == 0)
            {
                detector(numpy_image<rgb_pixel>(img), rect_detections, adjust_threshold);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);
                return rectangles;
            }
            else
            {
                pyramid_up(numpy_image<rgb_pixel>(img), temp, pyr);
                unsigned int levels = upsampling_amount-1;
                while (levels > 0)
                {
                    levels--;
                    pyramid_up(temp);
                }

                detector(temp, rect_detections, adjust_threshold);
                for (unsigned long i = 0; i < rect_detections.size(); ++i)
                    rect_detections[i].rect = pyr.rect_down(rect_detections[i].rect,
                                                            upsampling_amount);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);

                return rectangles;
            }
        }
        else
        {
            throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
        }
    }

    inline std::vector<dlib::rectangle> run_detectors_with_upscale1 (
        std::vector<simple_object_detector >& detectors,
        py::array img,
        const unsigned int upsampling_amount,
        const double adjust_threshold,
        std::vector<double>& detection_confidences,
        std::vector<unsigned long>& weight_indices
    )
    {
        pyramid_down<2> pyr;

        std::vector<rectangle> rectangles;
        std::vector<rect_detection> rect_detections;

        if (is_image<unsigned char>(img))
        {
            array2d<unsigned char> temp;
            if (upsampling_amount == 0)
            {
                evaluate_detectors(detectors, numpy_image<unsigned char>(img), rect_detections, adjust_threshold);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);
                return rectangles;
            }
            else
            {
                pyramid_up(numpy_image<unsigned char>(img), temp, pyr);
                unsigned int levels = upsampling_amount-1;
                while (levels > 0)
                {
                    levels--;
                    pyramid_up(temp);
                }

                evaluate_detectors(detectors, temp, rect_detections, adjust_threshold);
                for (unsigned long i = 0; i < rect_detections.size(); ++i)
                    rect_detections[i].rect = pyr.rect_down(rect_detections[i].rect,
                                                            upsampling_amount);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);

                return rectangles;
            }
        }
        else if (is_image<rgb_pixel>(img))
        {
            array2d<rgb_pixel> temp;
            if (upsampling_amount == 0)
            {
                evaluate_detectors(detectors, numpy_image<rgb_pixel>(img), rect_detections, adjust_threshold);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);
                return rectangles;
            }
            else
            {
                pyramid_up(numpy_image<rgb_pixel>(img), temp, pyr);
                unsigned int levels = upsampling_amount-1;
                while (levels > 0)
                {
                    levels--;
                    pyramid_up(temp);
                }

                evaluate_detectors(detectors, temp, rect_detections, adjust_threshold);
                for (unsigned long i = 0; i < rect_detections.size(); ++i)
                    rect_detections[i].rect = pyr.rect_down(rect_detections[i].rect,
                                                            upsampling_amount);
                split_rect_detections(rect_detections, rectangles,
                                      detection_confidences, weight_indices);

                return rectangles;
            }
        }
        else
        {
            throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
        }
    }

    inline std::vector<dlib::rectangle> run_detector_with_upscale2 (
        dlib::simple_object_detector& detector,
        py::array img,
        const unsigned int upsampling_amount

    )
    {
        std::vector<double> detection_confidences;
        std::vector<unsigned long> weight_indices;
        const double adjust_threshold = 0.0;

        return run_detector_with_upscale1(detector, img, upsampling_amount,
                                          adjust_threshold,
                                          detection_confidences, weight_indices);
    }

    inline py::tuple run_rect_detector (
        dlib::simple_object_detector& detector,
        py::array img,
        const unsigned int upsampling_amount,
        const double adjust_threshold)
    {
        py::tuple t;

        std::vector<double> detection_confidences;
        std::vector<unsigned long> weight_indices;
        std::vector<rectangle> rectangles;

        rectangles = run_detector_with_upscale1(detector, img, upsampling_amount,
                                                adjust_threshold,
                                                detection_confidences, weight_indices);

        return py::make_tuple(rectangles,
                              vector_to_python_list(detection_confidences), 
                              vector_to_python_list(weight_indices));
    }

    struct simple_object_detector_py
    {
        simple_object_detector detector;
        unsigned int upsampling_amount;

        simple_object_detector_py() {}
        simple_object_detector_py(simple_object_detector& _detector, unsigned int _upsampling_amount) :
            detector(_detector), upsampling_amount(_upsampling_amount) {}

        std::vector<dlib::rectangle> run_detector1 (py::array img,
                                                    const unsigned int upsampling_amount_)
        {
            return run_detector_with_upscale2(detector, img, upsampling_amount_);
        }

        std::vector<dlib::rectangle> run_detector2 (py::array img)
        {
            return run_detector_with_upscale2(detector, img, upsampling_amount);
        }


    };

    inline py::tuple run_multiple_rect_detectors (
        py::list& detectors,
        py::array img,
        const unsigned int upsampling_amount,
        const double adjust_threshold)
    {
        py::tuple t;

        std::vector<simple_object_detector> vector_detectors;
        const unsigned long num_detectors = len(detectors);
        // Now copy the data into dlib based objects.
        for (unsigned long i = 0; i < num_detectors; ++i)
        {
            try
            {
                vector_detectors.push_back(detectors[i].cast<simple_object_detector>());
            } catch(py::cast_error&)
            {
                vector_detectors.push_back(detectors[i].cast<simple_object_detector_py>().detector);
            }
        }

        std::vector<double> detection_confidences;
        std::vector<unsigned long> weight_indices;
        std::vector<rectangle> rectangles;

        rectangles = run_detectors_with_upscale1(vector_detectors, img, upsampling_amount,
                                                adjust_threshold,
                                                detection_confidences, weight_indices);

        return py::make_tuple(rectangles,
                              vector_to_python_list(detection_confidences),
                              vector_to_python_list(weight_indices));
    }



}

#endif // DLIB_SIMPLE_OBJECT_DETECTOR_PY_H__
