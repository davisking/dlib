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

boost::python::list get_face_chips (
    object img,
    const std::vector<full_object_detection>& faces,
    size_t size = 150,
    float padding = 0.25
)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    if (faces.size() < 1) {
        throw dlib::error("No face were specified in the faces array.");
    }

    boost::python::list chips_list;

    std::vector<chip_details> dets;
    for (auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_rgb_image(img), dets, face_chips);

    npy_intp rows = size;
    npy_intp cols = size;

    // Size of the numpy array
    npy_intp dims[3] = { rows, cols, 3};

    for (auto& chip : face_chips) 
    {
        PyObject *arr = PyArray_SimpleNew(3, dims, NPY_UINT8);
        npy_uint8 *outdata = (npy_uint8 *) PyArray_DATA((PyArrayObject*) arr);
        memcpy(outdata, image_data(chip), rows * width_step(chip));
        boost::python::handle<> handle(arr);

        // Append image to chips list
        chips_list.append(object(handle));
    }
    return chips_list;
}

object get_face_chip (
    object img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    matrix<rgb_pixel> chip;
    extract_image_chip(numpy_rgb_image(img), get_face_chip_details(face, size, padding), chip);

    // Size of the numpy array
    npy_intp dims[3] = { num_rows(chip), num_columns(chip), 3};

    PyObject *arr = PyArray_SimpleNew(3, dims, NPY_UINT8);
    npy_uint8 *outdata = (npy_uint8 *) PyArray_DATA((PyArrayObject *) arr);
    memcpy(outdata, image_data(chip), num_rows(chip) * width_step(chip));
    boost::python::handle<> handle(arr);
    return object(handle);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(get_face_chip_with_defaults, get_face_chip, 2, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(get_face_chips_with_defaults, get_face_chips, 2, 4)


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

    def("get_face_chip", &get_face_chip, get_face_chip_with_defaults(
	"Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.", 
	(arg("img"), arg("face"), arg("size"), arg("padding"))
    ));

    def("get_face_chips", &get_face_chips, get_face_chips_with_defaults(
	"Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.",
	(arg("img"), arg("faces"), arg("size"), arg("padding"))
    ));
}