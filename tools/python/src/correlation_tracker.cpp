// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/geometry.h>
#include <boost/python/args.hpp>
#include <dlib/image_processing.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

// ----------------------------------------------------------------------------------------

void start_track (
    correlation_tracker& tracker,
    object img,
    const drectangle& bounding_box
)
{
    if (is_gray_python_image(img))
    {
        tracker.start_track(numpy_gray_image(img), bounding_box);
    }
    else if (is_rgb_python_image(img))
    {
        tracker.start_track(numpy_rgb_image(img), bounding_box);
    }
    else
    {
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
    }
}

void start_track_rec (
    correlation_tracker& tracker,
    object img,
    const rectangle& bounding_box
)
{
    drectangle dbounding_box(bounding_box);
    start_track(tracker, img, dbounding_box);
}

void update (
    correlation_tracker& tracker,
    object img
)
{
    if (is_gray_python_image(img))
    {
        tracker.update(numpy_gray_image(img));
    }
    else if (is_rgb_python_image(img))
    {
        tracker.update(numpy_rgb_image(img));
    }
    else
    {
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
    }
}

void update_guess (
    correlation_tracker& tracker,
    object img,
    const drectangle& bounding_box
)
{
    if (is_gray_python_image(img))
    {
        tracker.update(numpy_gray_image(img), bounding_box);
    }
    else if (is_rgb_python_image(img))
    {
        tracker.update(numpy_rgb_image(img), bounding_box);
    }
    else
    {
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
    }
}

void update_guess_rec (
    correlation_tracker& tracker,
    object img,
    const rectangle& bounding_box
)
{
    drectangle dbounding_box(bounding_box);
    update_guess(tracker, img, dbounding_box);
}

drectangle get_position (const correlation_tracker& tracker) { return tracker.get_position(); }

// ----------------------------------------------------------------------------------------

void bind_correlation_tracker()
{
    using boost::python::arg;
    {
    typedef correlation_tracker type;
    class_<type>("correlation_tracker", "")
        .def("start_track", &::start_track, (arg("image"), arg("bounding_box")))
        .def("start_track", &::start_track_rec, (arg("image"), arg("bounding_box")))
        .def("update", &::update, arg("image"))
        .def("update", &::update_guess, (arg("image"), arg("guess")))
        .def("update", &::update_guess_rec, (arg("image"), arg("guess")))
        .def("get_position", &::get_position);
    }
}
