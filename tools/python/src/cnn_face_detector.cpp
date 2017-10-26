// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

class cnn_face_detection_model_v1
{

public:

    cnn_face_detection_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    std::vector<mmod_rect> detect (
        object pyimage,
        const int upsample_num_times
    )
    {
        pyramid_down<2> pyr;
        std::vector<mmod_rect> rects;

        // Copy the data into dlib based objects
        matrix<rgb_pixel> image;
        if (is_gray_python_image(pyimage))
            assign_image(image, numpy_gray_image(pyimage));
        else if (is_rgb_python_image(pyimage))
            assign_image(image, numpy_rgb_image(pyimage));
        else
            throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");

        // Upsampling the image will allow us to detect smaller faces but will cause the
        // program to use more RAM and run longer.
        unsigned int levels = upsample_num_times;
        while (levels > 0)
        {
            levels--;
            pyramid_up(image, pyr);
        }

        auto dets = net(image);

        // Scale the detection locations back to the original image size
        // if the image was upscaled.
        for (auto&& d : dets) {
            d.rect = pyr.rect_down(d.rect, upsample_num_times);
            rects.push_back(d);
        }

        return rects;
    }

    std::vector<std::vector<mmod_rect> > detect_mult (
        boost::python::list& imgs,
        const int upsample_num_times,
        const int batch_size = 128
    )
    {
        pyramid_down<2> pyr;
        std::vector<matrix<rgb_pixel> > dimgs;
        dimgs.reserve(len(imgs));

        for(int i = 0; i < len(imgs); i++)
        {
            // Copy the data into dlib based objects
            matrix<rgb_pixel> image;
            object tmp = boost::python::extract<object>(imgs[i]);
            if (is_gray_python_image(tmp))
                assign_image(image, numpy_gray_image(tmp));
            else if (is_rgb_python_image(tmp))
                assign_image(image, numpy_rgb_image(tmp));
            else
                throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");

            for(int i = 0; i < upsample_num_times; i++)
            {
                pyramid_up(image);
            }
            dimgs.push_back(image);
        }

        for(int i = 1; i < dimgs.size(); i++)
        {
            if
            (
                dimgs[i - 1].nc() != dimgs[i].nc() ||
                dimgs[i - 1].nr() != dimgs[i].nr()
            )
                throw dlib::error("Images in list must all have the same dimensions.");
            
        }        

        auto dets = net(dimgs, batch_size);
        std::vector<std::vector<mmod_rect> > all_rects;

        for(auto&& im_dets : dets)
        {
            std::vector<mmod_rect> rects;
            rects.reserve(im_dets.size());
            for (auto&& d : im_dets) {
                d.rect = pyr.rect_down(d.rect, upsample_num_times);
                rects.push_back(d);
            }
            all_rects.push_back(rects);
        }
        
        return all_rects;
    }

private:

    template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
    template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

    template <typename SUBNET> using downsampler  = relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
    template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;

    using net_type = loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

    net_type net;
};

// ----------------------------------------------------------------------------------------

void bind_cnn_face_detection()
{
    using boost::python::arg;
    {
    class_<cnn_face_detection_model_v1>("cnn_face_detection_model_v1", "This object detects human faces in an image.  The constructor loads the face detection model from a file. You can download a pre-trained model from http://dlib.net/files/mmod_human_face_detector.dat.bz2.", init<std::string>())
        .def(
            "__call__", 
            &cnn_face_detection_model_v1::detect, 
            (arg("img"), arg("upsample_num_times")=0),
            "Find faces in an image using a deep learning model.\n\
          - Upsamples the image upsample_num_times before running the face \n\
            detector."
            )
        .def(
            "__call__", 
            &cnn_face_detection_model_v1::detect_mult, 
            (arg("imgs"), arg("upsample_num_times")=0, arg("batch_size")=128), 
            "takes a list of images as input returning a 2d list of mmod rectangles"
            );
    }
    {
    typedef mmod_rect type;
    class_<type>("mmod_rectangle", "Wrapper around a rectangle object and a detection confidence score.")
        .def_readwrite("rect",   &type::rect)
        .def_readwrite("confidence", &type::detection_confidence);
    }
    {
    typedef std::vector<mmod_rect> type;
    class_<type>("mmod_rectangles", "An array of mmod rectangle objects.")
        .def(vector_indexing_suite<type>());
    }
    {
    typedef std::vector<std::vector<mmod_rect> > type;
    class_<type>("mmod_rectangless", "A 2D array of mmod rectangle objects.")
        .def(vector_indexing_suite<type>());
    } 
}
