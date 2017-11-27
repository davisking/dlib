#include <dlib/python.h>
#include <boost/python/args.hpp>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

dlib::rand rnd_jitter;

using namespace dlib;
using namespace std;
using namespace boost::python;

// ----------------------------------------------------------------------------------------

boost::python::list get_jitter_images(object img, size_t num_jitters = 1, bool disturb_colors = false)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    // Convert the image to matrix<rgb_pixel> for processing
    matrix<rgb_pixel> img_mat;
    assign_image(img_mat, numpy_rgb_image(img));

    // The top level list (containing 1 or more images) to return to python
    boost::python::list jitter_list;

    size_t rows = num_rows(img_mat);
    size_t cols = num_columns(img_mat);

    // Size of the numpy array
    npy_intp dims[3] = { num_rows(img_mat), num_columns(img_mat), 3};

    for (int i = 0; i < num_jitters; ++i) {
        // Get a jittered crop
        matrix<rgb_pixel> crop = dlib::jitter_image(img_mat, rnd_jitter);
        // If required disturb colors of the image
        if(disturb_colors)
            dlib::disturb_colors(crop, rnd_jitter);
        
        PyObject *arr = PyArray_SimpleNew(3, dims, NPY_UINT8);
        npy_uint8 *outdata = (npy_uint8 *) PyArray_DATA((PyArrayObject*) arr);
        memcpy(outdata, image_data(crop), rows * width_step(crop));
                
        boost::python::handle<> handle(arr);
        // Append image to jittered image list
        jitter_list.append(object(handle));
    }
           
    return jitter_list;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(get_jitter_images_with_defaults, get_jitter_images, 1, 3)

// ----------------------------------------------------------------------------------------

void bind_numpy_returns()
{
    using boost::python::arg;
    import_array();

    def("jitter_image", &get_jitter_images, get_jitter_images_with_defaults(
    "Takes an image and returns a list of jittered images."
    "The returned list contains num_jitters images (default is 1)."
    "If disturb_colors is set to True, the colors of the image are disturbed (default is False)", 
    (arg("img"), arg("num_jitters"), arg("disturb_colors"))
    ));
}