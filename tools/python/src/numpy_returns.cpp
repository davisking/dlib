#include "opaque_types.h"
#include <dlib/pixel.h>
#include <dlib/image_transforms.h>
#include <pybind11/numpy.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

namespace {

    pybind11::array_t<uint8_t> create_array_from_rgb_image(const matrix<rgb_pixel> &rgb_image)
    {
        const size_t pixel_size = sizeof(rgb_pixel);
        const size_t channel_size = sizeof(uint8_t);
        const auto rows = static_cast<const size_t>(num_rows(rgb_image));
        const auto cols = static_cast<const size_t>(num_columns(rgb_image));
        const size_t image_size = rows * cols * pixel_size;

        auto arr = new uint8_t[image_size];
        memcpy(arr, image_data(rgb_image), image_size);

        return pybind11::array_t<uint8_t>(
            {rows, cols, pixel_size},
            {cols * pixel_size, pixel_size, channel_size},
            arr,
            pybind11::capsule{
                arr, [](void *arr_p) {
                    delete[] reinterpret_cast<uint8_t *>(arr_p);
                }
            }
        );
    }

}

// ----------------------------------------------------------------------------------------

py::list get_jitter_images(py::object img, size_t num_jitters = 1, bool disturb_colors = false)
{
    static dlib::rand rnd_jitter;
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    // Convert the image to matrix<rgb_pixel> for processing
    matrix<rgb_pixel> img_mat;
    assign_image(img_mat, numpy_rgb_image(img));

    // The top level list (containing 1 or more images) to return to python
    py::list jitter_list;

    for (int i = 0; i < num_jitters; ++i) {
        // Get a jittered crop
        matrix<rgb_pixel> crop = dlib::jitter_image(img_mat, rnd_jitter);
        // If required disturb colors of the image
        if(disturb_colors)
            dlib::disturb_colors(crop, rnd_jitter);
        
        // Append image to jittered image list
        jitter_list.append(create_array_from_rgb_image(crop));
    }
           
    return jitter_list;
}

// ----------------------------------------------------------------------------------------

py::list get_face_chips (
    py::object img,
    const std::vector<full_object_detection>& faces,
    size_t size = 150,
    float padding = 0.25
)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    if (faces.empty()) {
        throw dlib::error("No face were specified in the faces array.");
    }

    py::list chips_list;

    std::vector<chip_details> dets;
    for (auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_rgb_image(img), dets, face_chips);

    for (const auto& chip : face_chips)
        chips_list.append(create_array_from_rgb_image(chip));

    return chips_list;
}

py::array_t<uint8_t> get_face_chip (
    py::object img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    matrix<rgb_pixel> chip;
    extract_image_chip(numpy_rgb_image(img), get_face_chip_details(face, size, padding), chip);

    return create_array_from_rgb_image(chip);
}

// ----------------------------------------------------------------------------------------

void bind_numpy_returns(py::module &m)
{
    m.def("jitter_image", &get_jitter_images,
    "Takes an image and returns a list of jittered images."
    "The returned list contains num_jitters images (default is 1)."
    "If disturb_colors is set to True, the colors of the image are disturbed (default is False)", 
    py::arg("img"), py::arg("num_jitters")=1, py::arg("disturb_colors")=false
    );

    m.def("get_face_chip", &get_face_chip,
	"Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.", 
	py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25
    );

    m.def("get_face_chips", &get_face_chips, 
	"Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.",
	py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25
    );
}
