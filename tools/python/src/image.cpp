#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

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
    sout << "rgb_pixel(" << (int)p.red << "," << (int)p.green << "," << (int)p.blue << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

void bind_image_classes(py::module& m)
{
    py::class_<rgb_pixel>(m, "rgb_pixel")
        .def(py::init<unsigned char,unsigned char,unsigned char>(), py::arg("red"), py::arg("green"), py::arg("blue"))
        .def("__str__", &print_rgb_pixel_str)
        .def("__repr__", &print_rgb_pixel_repr)
        .def_readwrite("red", &rgb_pixel::red)
        .def_readwrite("green", &rgb_pixel::green)
        .def_readwrite("blue", &rgb_pixel::blue);
}
