// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/matrix.h>
#include <boost/python/args.hpp>
#include <dlib/geometry.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>


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

boost::shared_ptr<frontal_face_detector> load_fhog_object_detector_from_file (
    const std::string& filename
)
{
    ifstream fin(filename.c_str(), ios::binary);
    if (!fin)
        throw dlib::error("Unable to open " + filename);
    boost::shared_ptr<frontal_face_detector> detector(new frontal_face_detector());
    deserialize(*detector, fin);
    return detector;
}

// ----------------------------------------------------------------------------------------

void bind_object_detection()
{
    using boost::python::arg;

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

    {
    typedef frontal_face_detector type;
    class_<type>("fhog_object_detector", 
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def("__init__", make_constructor(&load_fhog_object_detector_from_file),  
            "Loads a fhog_object_detector from a file.")
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


