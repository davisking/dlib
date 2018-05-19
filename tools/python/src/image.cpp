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

template <typename T>
numpy_image<unsigned char> py_threshold_image2(
    const numpy_image<T>& in_img,
    typename pixel_traits<T>::basic_pixel_type thresh
)
{
    numpy_image<unsigned char> out_img;
    threshold_image(in_img, out_img);
    return out_img;
}

template <typename T>
numpy_image<unsigned char> py_threshold_image(
    const numpy_image<T>& in_img
)
{
    numpy_image<unsigned char> out_img;
    threshold_image(in_img, out_img);
    return out_img;
}

// ----------------------------------------------------------------------------------------

template <typename T>
typename pixel_traits<T>::basic_pixel_type py_partition_pixels (
    const numpy_image<T>& img
)
{
    return partition_pixels(img);
}

template <typename T>
py::tuple py_partition_pixels2 (
    const numpy_image<T>& img,
    int num_thresholds
)
{
    DLIB_CASSERT(1 <= num_thresholds && num_thresholds <= 6);

    typename pixel_traits<T>::basic_pixel_type t1,t2,t3,t4,t5,t6;

    switch(num_thresholds)
    {
        case 1: partition_pixels(img,t1); return py::make_tuple(t1);
        case 2: partition_pixels(img,t1,t2); return py::make_tuple(t1,t2);
        case 3: partition_pixels(img,t1,t2,t3); return py::make_tuple(t1,t2,t3);
        case 4: partition_pixels(img,t1,t2,t3,t4); return py::make_tuple(t1,t2,t3,t4);
        case 5: partition_pixels(img,t1,t2,t3,t4,t5); return py::make_tuple(t1,t2,t3,t4,t5);
        case 6: partition_pixels(img,t1,t2,t3,t4,t5,t6); return py::make_tuple(t1,t2,t3,t4,t5,t6);
    }
    DLIB_CASSERT(false, "This should never happen.");
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

    const char* docs = "Thresholds img and returns the result.  Pixels in img with grayscale values >= partition_pixels(img) \n" 
              "have an output value of 255 and all others have a value of 0.";
    m.def("threshold_image", &py_threshold_image<unsigned char>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<uint16_t>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<uint32_t>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<float>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<double>, py::arg("img") );
    m.def("threshold_image", &py_threshold_image<rgb_pixel>,docs, py::arg("img") );

    docs = "Thresholds img and returns the result.  Pixels in img with grayscale values >= thresh \n"
              "have an output value of 255 and all others have a value of 0.";
    m.def("threshold_image", &py_threshold_image2<unsigned char>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<uint16_t>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<uint32_t>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<float>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<double>, py::arg("img"), py::arg("thresh") );
    m.def("threshold_image", &py_threshold_image2<rgb_pixel>,docs, py::arg("img"), py::arg("thresh") );


    docs = 
"Finds a threshold value that would be reasonable to use with \n\
threshold_image(img, threshold).  It does this by finding the threshold that \n\
partitions the pixels in img into two groups such that the sum of absolute \n\
deviations between each pixel and the mean of its group is minimized.";
    m.def("partition_pixels", &py_partition_pixels<rgb_pixel>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<unsigned char>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<uint16_t>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<uint32_t>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<float>, py::arg("img") );
    m.def("partition_pixels", &py_partition_pixels<double>,docs, py::arg("img") );

    docs = 
"This version of partition_pixels() finds multiple partitions rather than just \n\
one partition.  It does this by first partitioning the pixels just as the \n\
above partition_pixels(img) does.  Then it forms a new image with only pixels \n\
>= that first partition value and recursively partitions this new image. \n\
However, the recursion is implemented in an efficient way which is faster than \n\
explicitly forming these images and calling partition_pixels(), but the \n\
output is the same as if you did.  For example, suppose you called \n\
[t1,t2,t2] = partition_pixels(img,3).  Then we would have: \n\
   - t1 == partition_pixels(img) \n\
   - t2 == partition_pixels(an image with only pixels with values >= t1 in it) \n\
   - t3 == partition_pixels(an image with only pixels with values >= t2 in it)" ;
    m.def("partition_pixels", &py_partition_pixels2<rgb_pixel>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<unsigned char>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<uint16_t>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<uint32_t>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<float>, py::arg("img"), py::arg("num_thresholds") );
    m.def("partition_pixels", &py_partition_pixels2<double>,docs, py::arg("img"), py::arg("num_thresholds") );
}

