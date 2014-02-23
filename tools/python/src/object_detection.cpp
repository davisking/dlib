// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/matrix.h>
#include <boost/python/args.hpp>
#include <dlib/geometry.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include "simple_object_detector.h"


using namespace dlib;
using namespace std;
using namespace boost::python;

template <typename T>
void resize(T& v, unsigned long n) { v.resize(n); }

// ----------------------------------------------------------------------------------------

long left(const rectangle& r) { return r.left(); }
long top(const rectangle& r) { return r.top(); }
long right(const rectangle& r) { return r.right(); }
long bottom(const rectangle& r) { return r.bottom(); }
long width(const rectangle& r) { return r.width(); }
long height(const rectangle& r) { return r.height(); }

string print_rectangle_str(const rectangle& r)
{
    std::ostringstream sout;
    sout << r;
    return sout.str();
}

string print_rectangle_repr(const rectangle& r)
{
    std::ostringstream sout;
    sout << "rectangle(" << r.left() << "," << r.top() << "," << r.right() << "," << r.bottom() << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

string print_rgb_pixel_str(const rgb_pixel& p)
{
    std::ostringstream sout;
    sout << "red: "<< (int)p.red 
         << ", green: "<< (int)p.green 
         << ", blue: "<< (int)p.blue;
    return sout.str();
}

string print_rgb_pixel_repr(const rgb_pixel& p)
{
    std::ostringstream sout;
    sout << "rgb_pixel(" << p.red << "," << p.green << "," << p.blue << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

std::vector<rectangle> run_detector (
    frontal_face_detector& detector,
    object img,
    const unsigned int upsampling_amount
)
{
    pyramid_down<2> pyr;
    array2d<unsigned char> temp;

    if (is_gray_python_image(img))
    {
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


// ----------------------------------------------------------------------------------------

struct simple_object_detector_py
{
    simple_object_detector detector;
    unsigned int upsampling_amount;
    
    std::vector<rectangle> run_detector1 (object img, const unsigned int upsampling_amount_) 
    { return ::run_detector(detector, img, upsampling_amount_); }

    std::vector<rectangle> run_detector2 (object img) 
    { return ::run_detector(detector, img, upsampling_amount); }
};

void serialize (const simple_object_detector_py& item, std::ostream& out)
{
    int version = 1;
    serialize(item.detector, out);
    serialize(version, out);
    serialize(item.upsampling_amount, out);
}

void deserialize (simple_object_detector_py& item, std::istream& in)
{
    int version = 0;
    deserialize(item.detector, in);
    deserialize(version, in);
    if (version != 1)
        throw dlib::serialization_error("Unexpected version found while deserializing a simple_object_detector.");
    deserialize(item.upsampling_amount, in);
}

// ----------------------------------------------------------------------------------------

void image_window_set_image (
    image_window& win,
    object img
)
{
    if (is_gray_python_image(img))
        return win.set_image(numpy_gray_image(img));
    else if (is_rgb_python_image(img))
        return win.set_image(numpy_rgb_image(img));
    else
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
}


void add_red_overlay_rects (
    image_window& win,
    const std::vector<rectangle>& rects
)
{
    win.add_overlay(rects, rgb_pixel(255,0,0));
}

// ----------------------------------------------------------------------------------------

boost::shared_ptr<image_window> make_image_window_from_image(object img)
{
    boost::shared_ptr<image_window> win(new image_window);
    image_window_set_image(*win, img);
    return win;
}

boost::shared_ptr<image_window> make_image_window_from_image_and_title(object img, const string& title)
{
    boost::shared_ptr<image_window> win(new image_window);
    image_window_set_image(*win, img);
    win->set_title(title);
    return win;
}

// ----------------------------------------------------------------------------------------

string print_simple_test_results(const simple_test_results& r)
{
    std::ostringstream sout;
    sout << "precision: "<<r.precision << ", recall: "<< r.recall << ", average precision: " << r.average_precision;
    return sout.str();
}

// ----------------------------------------------------------------------------------------

void bind_object_detection()
{
    using boost::python::arg;

    class_<simple_object_detector_training_options>("simple_object_detector_training_options")
        .add_property("be_verbose", &simple_object_detector_training_options::be_verbose, 
                                    &simple_object_detector_training_options::be_verbose)
        .add_property("add_left_right_image_flips", &simple_object_detector_training_options::add_left_right_image_flips, 
                                                    &simple_object_detector_training_options::add_left_right_image_flips)
        .add_property("detection_window_size", &simple_object_detector_training_options::detection_window_size,
                                               &simple_object_detector_training_options::detection_window_size)
        .add_property("num_threads", &simple_object_detector_training_options::num_threads,
                                     &simple_object_detector_training_options::num_threads);

    class_<simple_test_results>("simple_test_results")
        .add_property("precision", &simple_test_results::precision)
        .add_property("recall", &simple_test_results::recall)
        .add_property("average_precision", &simple_test_results::average_precision)
        .def("__str__", &::print_simple_test_results);


    {
    typedef rectangle type;
    class_<type>("rectangle", "This object represents a rectangular area of an image.")
        .def(init<long,long,long,long>( (arg("left"),arg("top"),arg("right"),arg("bottom")) ))
        .def("left",   &::left)
        .def("top",    &::top)
        .def("right",  &::right)
        .def("bottom", &::bottom)
        .def("width",  &::width)
        .def("height", &::height)
        .def("__str__", &::print_rectangle_str)
        .def("__repr__", &::print_rectangle_repr)
        .def_pickle(serialize_pickle<type>());
    }

    def("get_frontal_face_detector", get_frontal_face_detector, 
        "Returns the default face detector");

    def("train_simple_object_detector", train_simple_object_detector,
        (arg("dataset_filename"), arg("detector_output_filename"), arg("C"), arg("options")=simple_object_detector_training_options()),
        "whatever");

    def("test_simple_object_detector", test_simple_object_detector,
        (arg("dataset_filename"), arg("detector_filename")),
        "whatever");

    {
    typedef simple_object_detector_py type;
    class_<type>("simple_object_detector", 
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def("__init__", make_constructor(&load_object_from_file<type>),  
"Loads a simple_object_detector from a file that contains the output of the \n\
train_simple_object_detector() routine." 
            /*!
                Loads a simple_object_detector from a file that contains the output of the
                train_simple_object_detector() routine.
            !*/)
        .def("__call__", &type::run_detector1, (arg("image"), arg("upsample_num_times")),
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a list of detections.   \n\
    - Upsamples the image upsample_num_times before running the basic \n\
      detector.  If you don't know how many times you want to upsample then \n\
      don't provide a value for upsample_num_times and an appropriate \n\
      default will be used." 
            /*!
                requires
                    - image is a numpy ndarray containing either an 8bit grayscale or RGB
                      image.
                    - upsample_num_times >= 0
                ensures
                    - This function runs the object detector on the input image and returns
                      a list of detections.  
                    - Upsamples the image upsample_num_times before running the basic
                      detector.  If you don't know how many times you want to upsample then
                      don't provide a value for upsample_num_times and an appropriate
                      default will be used.
            !*/
            )
        .def("__call__", &type::run_detector2, (arg("image")),
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a list of detections.  " 
            /*!
                requires
                    - image is a numpy ndarray containing either an 8bit grayscale or RGB
                      image.
                ensures
                    - This function runs the object detector on the input image and returns
                      a list of detections.  
            !*/
            )
        .def_pickle(serialize_pickle<type>());
    }

    {
    typedef frontal_face_detector type;
    class_<type>("fhog_object_detector", 
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def("__init__", make_constructor(&load_object_from_file<type>),  
"Loads a fhog_object_detector from a file that contains a serialized  \n\
object_detector<scan_fhog_pyramid<pyramid_down<6>>> object.  " )
        .def("__call__", &::run_detector, (arg("image"), arg("upsample_num_times")=0),
"requires \n\
    - image is a numpy ndarray containing either an 8bit \n\
      grayscale or RGB image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the object detector on the input image \n\
      and returns a list of detections.   \n\
    - You can detect smaller objects by upsampling the image \n\
      before running the detector.  This function can do that \n\
      for you automatically if you set upsample_num_times to a \n\
      non-zero value.  Specifically, the image is doubled in \n\
      size upsample_num_times times.   " 
            /*!
                requires
                    - image is a numpy ndarray containing either an 8bit
                      grayscale or RGB image.
                    - upsample_num_times >= 0
                ensures
                    - This function runs the object detector on the input image
                      and returns a list of detections.  
                    - You can detect smaller objects by upsampling the image
                      before running the detector.  This function can do that
                      for you automatically if you set upsample_num_times to a
                      non-zero value.  Specifically, the image is doubled in
                      size upsample_num_times times.   
            !*/
            )
        .def_pickle(serialize_pickle<type>());
    }

    {
    typedef image_window type;
    typedef void (image_window::*set_title_funct)(const std::string&);
    typedef void (image_window::*add_overlay_funct)(const std::vector<rectangle>& r, rgb_pixel p);
    class_<type,boost::noncopyable>("image_window", 
        "This is a GUI window capable of showing images on the screen.")
        .def("__init__", make_constructor(&make_image_window_from_image), 
            "Create an image window that displays the given numpy image.")
        .def("__init__", make_constructor(&make_image_window_from_image_and_title),
            "Create an image window that displays the given numpy image and also has the given title.")
        .def("set_image", image_window_set_image, arg("image"), 
            "Make the image_window display the given image.")
        .def("set_title", (set_title_funct)&type::set_title, arg("title"),
            "Set the title of the window to the given value.")
        .def("clear_overlay", &type::clear_overlay, "Remove all overlays from the image_window.")
        .def("add_overlay", (add_overlay_funct)&type::add_overlay<rgb_pixel>, (arg("rectangles"), arg("color")),
            "Add a list of rectangles to the image_window.  They will be displayed as boxes of the given color.")
        .def("add_overlay", add_red_overlay_rects, 
            "Add a list of rectangles to the image_window.  They will be displayed as red boxes.")
        .def("wait_until_closed", &type::wait_until_closed, 
            "This function blocks until the window is closed.");
    }

    {
    typedef std::vector<rectangle> type;
    class_<type>("rectangles", "An array of rectangle objects.")
        .def(vector_indexing_suite<type>())
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def_pickle(serialize_pickle<type>());
    }

    class_<rgb_pixel>("rgb_pixel")
        .def(init<unsigned char,unsigned char,unsigned char>( (arg("red"),arg("green"),arg("blue")) ))
        .def("__str__", &print_rgb_pixel_str)
        .def("__repr__", &print_rgb_pixel_repr)
        .add_property("red", &rgb_pixel::red)
        .add_property("green", &rgb_pixel::green)
        .add_property("blue", &rgb_pixel::blue);
}

// ----------------------------------------------------------------------------------------


