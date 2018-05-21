#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>
#include <dlib/image_processing.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_resize_image (
    const numpy_image<T>& img,
    unsigned long rows,
    unsigned long cols
)
{
    numpy_image<T> out;
    set_image_size(out, rows, cols);
    resize_image(img, out);
    return out;
}

// ----------------------------------------------------------------------------------------

template <typename T>
numpy_image<T> py_equalize_histogram (
    const numpy_image<T>& img
)
{
    numpy_image<T> out;
    equalize_histogram(img,out);
    return out;
}

// ----------------------------------------------------------------------------------------

void bind_image_classes2(py::module& m)
{

    const char* docs = "Resizes img, using bilinear interpolation, to have the indicated number of rows and columns.";


    m.def("resize_image", &py_resize_image<uint8_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<uint16_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<uint32_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<uint64_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int8_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int16_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int32_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<int64_t>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<float>, py::arg("img"), py::arg("rows"), py::arg("cols"));
    m.def("resize_image", &py_resize_image<double>, docs, py::arg("img"), py::arg("rows"), py::arg("cols"));


    docs = "Returns a histogram equalized version of img.";
    m.def("equalize_histogram", &py_equalize_histogram<uint8_t>, py::arg("img"));
    m.def("equalize_histogram", &py_equalize_histogram<uint16_t>, docs, py::arg("img"));
}


