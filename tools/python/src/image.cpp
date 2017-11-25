#include <dlib/python.h>
#include <boost/python/args.hpp>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

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

boost::python::list get_jitter_images(object img, size_t num_jitters = 1, bool disturb_colors = false)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    dlib::rand rnd;

    // Convert the image to matrix<rgb_pixel> for processing
    matrix<rgb_pixel> img_mat;
    assign_image(img_mat, numpy_rgb_image(img));

    // The top level list (containing 1 or more images) to return to python
    boost::python::list jitter_list;

    for (int i = 0; i < num_jitters; ++i) {
        // Get a jittered crop
        matrix<rgb_pixel> crop = dlib::jitter_image(img_mat,rnd);
        // If required disturb colors of the image
        if(disturb_colors)
            dlib::disturb_colors(crop,rnd);

        // Treat the image as a list of lists
        boost::python::list img_as_list;
        size_t rows = img_mat.nr();
        size_t cols = img_mat.nc();
        
        for(size_t row=0; row<rows; row++) {
            boost::python::list row_list;
            for(size_t col=0; col<cols; col++) {
                rgb_pixel pixel = crop(row, col);
                boost::python::list item;
                
                item.append(pixel.red);
                item.append(pixel.green);
                item.append(pixel.blue);
                row_list.append(item);
            }
            img_as_list.append(row_list);
        }

        jitter_list.append(img_as_list);
    }
        
    return jitter_list;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(get_jitter_images_with_defaults, get_jitter_images, 1, 3)

// ----------------------------------------------------------------------------------------
void bind_image_classes()
{
    using boost::python::arg;

    {
    class_<rgb_pixel>("rgb_pixel")
        .def(init<unsigned char,unsigned char,unsigned char>( (arg("red"),arg("green"),arg("blue")) ))
        .def("__str__", &print_rgb_pixel_str)
        .def("__repr__", &print_rgb_pixel_repr)
        .add_property("red", &rgb_pixel::red)
        .add_property("green", &rgb_pixel::green)
        .add_property("blue", &rgb_pixel::blue);
    }

    def("jitter_image", &get_jitter_images, get_jitter_images_with_defaults(
    "Takes an image and returns a list of jittered images."
    "The returned list contains num_jitters images (default is 1)."
    "If disturb_colors is set to True, the colors of the image are disturbed (default is False)", 
    (arg("img"), arg("num_jitters"), arg("disturb_colors"))
    ));
}
