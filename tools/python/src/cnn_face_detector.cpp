// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <boost/python/slice.hpp>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> cv;


class cnn_face_detection_model_v1
{

public:

    cnn_face_detection_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    std::vector<rectangle> cnn_face_detector (
        object pyimage,
        const int upsample_num_times
    )
    {
        pyramid_down<2> pyr;
        std::vector<rectangle> rects;

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
            rects.push_back(d.rect);
        }

        return rects;
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
        .def("cnn_face_detector", &cnn_face_detection_model_v1::cnn_face_detector, (arg("img"), arg("upsample_num_times")=0),
            "Find faces in an image using a deep learning model.\n\
          - Upsamples the image upsample_num_times before running the face \n\
            detector."
            );
    }
}
