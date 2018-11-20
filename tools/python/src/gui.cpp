#ifndef DLIB_NO_GUI_SUPPORT

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/geometry.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include "simple_object_detector_py.h"

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

void image_window_set_image_fhog_detector (
    image_window& win,
    const simple_object_detector& det
)
{
    win.set_image(draw_fhog(det));
}

void image_window_set_image_simple_detector_py (
    image_window& win,
    const simple_object_detector_py& det
)
{
    win.set_image(draw_fhog(det.detector));
}

// ----------------------------------------------------------------------------------------

template <typename T>
void image_window_set_image (
    image_window& win,
    const numpy_image<T>& img
)
{
    win.set_image(img);
}

void add_overlay_rect (
    image_window& win,
    const rectangle& rect,
    const rgb_pixel& color
)
{
    win.add_overlay(rect, color);
}

void add_overlay_drect (
    image_window& win,
    const drectangle& drect,
    const rgb_pixel& color
)
{
    rectangle rect(drect.left(), drect.top(), drect.right(), drect.bottom());
    win.add_overlay(rect, color);
}

void add_overlay_parts (
    image_window& win,
    const full_object_detection& detection,
    const rgb_pixel& color
)
{
    if (detection.num_parts() == 5 || detection.num_parts() == 68)
    {
        win.add_overlay(render_face_detections(detection, color));
    }
    else
    {
        std::vector<image_display::overlay_circle> tmp;
        for (unsigned long i = 0; i < detection.num_parts(); ++i)
            tmp.emplace_back(detection.part(i), 0.5, color, std::to_string(i));
        win.add_overlay(tmp);
        win.add_overlay(detection.get_rect());
    }
}

void add_overlay_line (
    image_window& win,
    const line& l,
    const rgb_pixel& color
)
{
    win.add_overlay(l,color);
}

void add_overlay_pylist (
    image_window& win,
    const py::list& objs,
    const rgb_pixel& color
)
{
    std::vector<rectangle> rects;
    for (const auto& obj : objs)
    {
        try { rects.push_back(obj.cast<rectangle>()); continue; } catch(py::cast_error&) { }
        try { rects.push_back(obj.cast<drectangle>()); continue; } catch(py::cast_error&) { }
        try { win.add_overlay(obj.cast<line>(), color); continue; } catch(py::cast_error&) { }
        add_overlay_parts(win, obj.cast<full_object_detection>(), color); 
    }
    win.add_overlay(rects, color);
}

template <typename point_type>
void add_overlay_circle (
    image_window& win,
    const point_type& c,
    const double radius,
    const rgb_pixel& color
)
{
    win.add_overlay(image_window::overlay_circle(c,radius,color));
}

template <typename T>
std::shared_ptr<image_window> make_image_window_from_image(const numpy_image<T>& img)
{
    auto win = std::make_shared<image_window>();
    image_window_set_image(*win, img);
    return win;
}

template <typename T>
std::shared_ptr<image_window> make_image_window_from_image_and_title(const numpy_image<T>& img, const string& title)
{
    auto win = std::make_shared<image_window>();
    image_window_set_image(*win, img);
    win->set_title(title);
    return win;
}

std::shared_ptr<image_window> make_image_window_from_detector(const simple_object_detector& detector)
{
    auto win = std::make_shared<image_window>();
    win->set_image(draw_fhog(detector));
    return win;
}

std::shared_ptr<image_window> make_image_window_from_detector_py(const simple_object_detector_py& detector)
{
    auto win = std::make_shared<image_window>();
    win->set_image(draw_fhog(detector.detector));
    return win;
}

std::shared_ptr<image_window> make_image_window_from_detector_and_title(const simple_object_detector& detector, const string& title)
{
    auto win = std::make_shared<image_window>();
    win->set_image(draw_fhog(detector));
    win->set_title(title);
    return win;
}

std::shared_ptr<image_window> make_image_window_from_detector_py_and_title(const simple_object_detector_py& detector, const string& title)
{
    auto win = std::make_shared<image_window>();
    win->set_image(draw_fhog(detector.detector));
    win->set_title(title);
    return win;
}

// ----------------------------------------------------------------------------------------

void bind_gui(py::module& m)
{
    {
    typedef image_window type;
    typedef void (image_window::*set_title_funct)(const std::string&);
    typedef void (image_window::*add_overlay_funct)(const std::vector<rectangle>& r, rgb_pixel p);

    const char* docs1 = "Create an image window that displays the given numpy image.";
    const char* docs2 = "Create an image window that displays the given numpy image and also has the given title.";
    const char* docs3 = "Make the image_window display the given image.";
    py::class_<type, std::shared_ptr<type>>(m, "image_window",
        "This is a GUI window capable of showing images on the screen.")
        .def(py::init())
        .def(py::init(&make_image_window_from_detector))
        .def(py::init(&make_image_window_from_detector_py))
        .def(py::init(&make_image_window_from_detector_and_title))
        .def(py::init(&make_image_window_from_detector_py_and_title))
        .def(py::init(&make_image_window_from_image<uint8_t>))
        .def(py::init(&make_image_window_from_image<uint16_t>))
        .def(py::init(&make_image_window_from_image<uint32_t>))
        .def(py::init(&make_image_window_from_image<uint64_t>))
        .def(py::init(&make_image_window_from_image<int8_t>))
        .def(py::init(&make_image_window_from_image<int16_t>))
        .def(py::init(&make_image_window_from_image<int32_t>))
        .def(py::init(&make_image_window_from_image<int64_t>))
        .def(py::init(&make_image_window_from_image<float>))
        .def(py::init(&make_image_window_from_image<double>))
        .def(py::init(&make_image_window_from_image<rgb_pixel>), docs1)
        .def(py::init(&make_image_window_from_image_and_title<uint8_t>))
        .def(py::init(&make_image_window_from_image_and_title<uint16_t>))
        .def(py::init(&make_image_window_from_image_and_title<uint32_t>))
        .def(py::init(&make_image_window_from_image_and_title<uint64_t>))
        .def(py::init(&make_image_window_from_image_and_title<int8_t>))
        .def(py::init(&make_image_window_from_image_and_title<int16_t>))
        .def(py::init(&make_image_window_from_image_and_title<int32_t>))
        .def(py::init(&make_image_window_from_image_and_title<int64_t>))
        .def(py::init(&make_image_window_from_image_and_title<float>))
        .def(py::init(&make_image_window_from_image_and_title<double>))
        .def(py::init(&make_image_window_from_image_and_title<rgb_pixel>), docs2)
        .def("set_image", image_window_set_image_simple_detector_py, py::arg("detector"),
            "Make the image_window display the given HOG detector's filters.")
        .def("set_image", image_window_set_image_fhog_detector, py::arg("detector"),
            "Make the image_window display the given HOG detector's filters.")
        .def("set_image", image_window_set_image<uint8_t>, py::arg("image"))
        .def("set_image", image_window_set_image<uint16_t>, py::arg("image"))
        .def("set_image", image_window_set_image<uint32_t>, py::arg("image"))
        .def("set_image", image_window_set_image<uint64_t>, py::arg("image"))
        .def("set_image", image_window_set_image<int8_t>, py::arg("image"))
        .def("set_image", image_window_set_image<int16_t>, py::arg("image"))
        .def("set_image", image_window_set_image<int32_t>, py::arg("image"))
        .def("set_image", image_window_set_image<int64_t>, py::arg("image"))
        .def("set_image", image_window_set_image<float>, py::arg("image"))
        .def("set_image", image_window_set_image<double>, py::arg("image"))
        .def("set_image", image_window_set_image<rgb_pixel>, py::arg("image"), docs3)
        .def("set_title", (set_title_funct)&type::set_title, py::arg("title"),
            "Set the title of the window to the given value.")
        .def("clear_overlay", &type::clear_overlay, "Remove all overlays from the image_window.")
        .def("add_overlay", (add_overlay_funct)&type::add_overlay<rgb_pixel>, py::arg("rectangles"), py::arg("color")=rgb_pixel(255, 0, 0),
            "Add a list of rectangles to the image_window. They will be displayed as red boxes by default, but the color can be passed.")
        .def("add_overlay", add_overlay_rect, py::arg("rectangle"), py::arg("color")=rgb_pixel(255, 0, 0),
            "Add a rectangle to the image_window.  It will be displayed as a red box by default, but the color can be passed.")
        .def("add_overlay", add_overlay_drect, py::arg("rectangle"), py::arg("color")=rgb_pixel(255, 0, 0),
            "Add a rectangle to the image_window.  It will be displayed as a red box by default, but the color can be passed.")
        .def("add_overlay", add_overlay_parts, py::arg("detection"), py::arg("color")=rgb_pixel(0, 0, 255),
            "Add full_object_detection parts to the image window. They will be displayed as blue lines by default, but the color can be passed.")
        .def("add_overlay", add_overlay_line, py::arg("line"), py::arg("color")=rgb_pixel(255, 0, 0),
            "Add line to the image window.")
        .def("add_overlay_circle", add_overlay_circle<point>, py::arg("center"), py::arg("radius"), py::arg("color")=rgb_pixel(255, 0, 0),
            "Add circle to the image window.")
        .def("add_overlay_circle", add_overlay_circle<dpoint>, py::arg("center"), py::arg("radius"), py::arg("color")=rgb_pixel(255, 0, 0),
            "Add circle to the image window.")
        .def("add_overlay", add_overlay_pylist, py::arg("objects"), py::arg("color")=rgb_pixel(255,0,0),
            "Adds all the overlayable objects, uses the given color.")
        .def("wait_until_closed", &type::wait_until_closed,
            "This function blocks until the window is closed.");
    }
}
#endif
