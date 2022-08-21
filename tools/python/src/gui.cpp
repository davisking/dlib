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

void wait_for_keypress_char(image_window& win, const char wait_key) {
    unsigned long key;
    bool is_printable;
    while(win.get_next_keypress(key,is_printable)) 
    {
        if (is_printable && (char)key == wait_key)
            return;
    }
}

void wait_for_keypress_other(image_window& win, const base_window::non_printable_keyboard_keys wait_key) {
    unsigned long key;
    bool is_printable;
    while(win.get_next_keypress(key,is_printable)) 
    {
        if (!is_printable && key == wait_key)
            return;
    }
}

py::object get_next_double_click(image_window& win) {
    point p;
    if (win.get_next_double_click(p))
        return py::cast(p);
    else
        return py::none();
}

py::object get_next_keypress(image_window& win, bool get_keyboard_modifiers) 
{
    unsigned long key;
    bool is_printable;
    unsigned long state;

    auto state_to_list = [&]() {
        py::list mods;
        if (state&base_window::KBD_MOD_SHIFT) mods.append(base_window::KBD_MOD_SHIFT);
        if (state&base_window::KBD_MOD_CONTROL) mods.append(base_window::KBD_MOD_CONTROL);
        if (state&base_window::KBD_MOD_ALT) mods.append(base_window::KBD_MOD_ALT);
        if (state&base_window::KBD_MOD_META) mods.append(base_window::KBD_MOD_META);
        if (state&base_window::KBD_MOD_CAPS_LOCK) mods.append(base_window::KBD_MOD_CAPS_LOCK);
        if (state&base_window::KBD_MOD_NUM_LOCK) mods.append(base_window::KBD_MOD_NUM_LOCK);
        if (state&base_window::KBD_MOD_SCROLL_LOCK) mods.append(base_window::KBD_MOD_SCROLL_LOCK);
        return mods;
    };

    if(win.get_next_keypress(key, is_printable, state))
    {
        if (is_printable)
        {
            if (get_keyboard_modifiers)
                return py::make_tuple((char)key, state_to_list());
            else
                return py::cast((char)key);
        }
        else
        {
            if (get_keyboard_modifiers)
                return py::make_tuple(static_cast<base_window::non_printable_keyboard_keys>(key), state_to_list());
            else
                return py::cast(static_cast<base_window::non_printable_keyboard_keys>(key));
        }
    }
    else
    {
        if (get_keyboard_modifiers)
            return py::make_tuple(py::none(), py::none());
        else
            return py::none();
    }
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
            "This function blocks until the window is closed.")
        .def("is_closed", &type::is_closed,
            "returns true if this window has been closed, false otherwise.  (Note that closed windows do not receive any callbacks at all.  They are also not visible on the screen.)")
        .def("get_next_double_click", get_next_double_click,
            "Blocks until the user double clicks on the image or closes the window.  Returns a dlib.point indicating the pixel the user clicked on or None if the window as closed.")
        .def("wait_for_keypress", wait_for_keypress_char, py::arg("key"),
            "Blocks until the user presses the given key or closes the window.")
        .def("wait_for_keypress", wait_for_keypress_other, py::arg("key"),
            "Blocks until the user presses the given key or closes the window.")
        .def("get_next_keypress", get_next_keypress, py::arg("get_keyboard_modifiers")=false,
"Blocks until the user presses a key on their keyboard or the window is closed. \n\
 \n\
ensures \n\
    - if (get_keyboard_modifiers==True) then \n\
        - returns a tuple of (key_pressed, keyboard_modifiers_active) \n\
    - else \n\
        - returns just the key that was pressed.   \n\
    - The returned key is either a str containing the letter that was pressed, or  \n\
      an element of the dlib.non_printable_keyboard_keys enum. \n\
    - keyboard_modifiers_active, if returned, is a list of elements of the \n\
      dlib.keyboard_mod_keys enum.  They tell you if a key like shift was being held \n\
      down or not during the button press. \n\
    - If the window is closed before the user presses a key then this function \n\
      returns with all outputs set to None." 
            /*!
                Blocks until the user presses a key on their keyboard or the window is closed.

                ensures
                    - if (get_keyboard_modifiers==True) then
                        - returns a tuple of (key_pressed, keyboard_modifiers_active)
                    - else
                        - returns just the key that was pressed.  
                    - The returned key is either a str containing the letter that was pressed, or 
                      an element of the dlib.non_printable_keyboard_keys enum.
                    - keyboard_modifiers_active, if returned, is a list of elements of the
                      dlib.keyboard_mod_keys enum.  They tell you if a key like shift was being held
                      down or not during the button press.
                    - If the window is closed before the user presses a key then this function
                      returns with all outputs set to None.
            !*/
            );
    }

    py::enum_<base_window::non_printable_keyboard_keys>(m,"non_printable_keyboard_keys")
        .value("KEY_BACKSPACE", base_window::KEY_BACKSPACE)
        .value("KEY_SHIFT", base_window::KEY_SHIFT)
        .value("KEY_CTRL", base_window::KEY_CTRL)
        .value("KEY_ALT", base_window::KEY_ALT)
        .value("KEY_PAUSE", base_window::KEY_PAUSE)
        .value("KEY_CAPS_LOCK", base_window::KEY_CAPS_LOCK)
        .value("KEY_ESC", base_window::KEY_ESC)
        .value("KEY_PAGE_UP", base_window::KEY_PAGE_UP)
        .value("KEY_PAGE_DOWN", base_window::KEY_PAGE_DOWN)
        .value("KEY_END", base_window::KEY_END)
        .value("KEY_HOME", base_window::KEY_HOME)
        .value("KEY_LEFT", base_window::KEY_LEFT)
        .value("KEY_RIGHT", base_window::KEY_RIGHT)
        .value("KEY_UP", base_window::KEY_UP)
        .value("KEY_DOWN", base_window::KEY_DOWN)
        .value("KEY_INSERT", base_window::KEY_INSERT)
        .value("KEY_DELETE", base_window::KEY_DELETE)
        .value("KEY_SCROLL_LOCK", base_window::KEY_SCROLL_LOCK)
        .value("KEY_F1", base_window::KEY_F1)
        .value("KEY_F2", base_window::KEY_F2)
        .value("KEY_F3", base_window::KEY_F3)
        .value("KEY_F4", base_window::KEY_F4)
        .value("KEY_F5", base_window::KEY_F5)
        .value("KEY_F6", base_window::KEY_F6)
        .value("KEY_F7", base_window::KEY_F7)
        .value("KEY_F8", base_window::KEY_F8)
        .value("KEY_F9", base_window::KEY_F9)
        .value("KEY_F10", base_window::KEY_F10)
        .value("KEY_F11", base_window::KEY_F11)
        .value("KEY_F12", base_window::KEY_F12)
        .export_values()
        // allow someone to compare this enum to a string since the return from get_next_keypress()
        // can be either this enum or a string and forcing the user to type check with an if is
        // maddening.
        .def("__eq__", [](base_window::non_printable_keyboard_keys, const std::string&){ return false; });

    py::enum_<base_window::keyboard_state_masks>(m,"keyboard_mod_keys")
        .value("KBD_MOD_NONE", base_window::KBD_MOD_NONE)
        .value("KBD_MOD_SHIFT", base_window::KBD_MOD_SHIFT)
        .value("KBD_MOD_CONTROL", base_window::KBD_MOD_CONTROL)
        .value("KBD_MOD_ALT", base_window::KBD_MOD_ALT)
        .value("KBD_MOD_META", base_window::KBD_MOD_META)
        .value("KBD_MOD_CAPS_LOCK", base_window::KBD_MOD_CAPS_LOCK)
        .value("KBD_MOD_NUM_LOCK", base_window::KBD_MOD_NUM_LOCK)
        .value("KBD_MOD_SCROLL_LOCK", base_window::KBD_MOD_SCROLL_LOCK)
        .export_values();

}
#endif
