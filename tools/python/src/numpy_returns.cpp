#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <pybind11/numpy.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

py::array_t<uint8_t> convert_to_numpy(matrix<rgb_pixel>&& rgb_image)
{
    const size_t dtype_size = sizeof(uint8_t);
    const auto rows = static_cast<const size_t>(num_rows(rgb_image));
    const auto cols = static_cast<const size_t>(num_columns(rgb_image));
    const size_t channels = 3;
    const size_t image_size = dtype_size * rows * cols * channels;

    unique_ptr<rgb_pixel[]> arr_ptr = rgb_image.steal_memory();
    uint8_t* arr = (uint8_t *) arr_ptr.release();

    return pybind11::array_t<uint8_t>(
        {rows, cols, channels},                                                     // shape
        {dtype_size * cols * channels, dtype_size * channels, dtype_size},          // strides
        arr,                                                                        // pointer
        pybind11::capsule{
            arr, [](void *arr_p) {
                delete[] reinterpret_cast<uint8_t *>(arr_p);
            }
        }
    );
}

// -------------------------------- Basic Image IO ----------------------------------------

py::array_t<uint8_t> load_rgb_image (const std::string &path)
{
    matrix<rgb_pixel> img;
    load_image(img, path);
    return convert_to_numpy(std::move(img));
}

bool has_ending (std::string const full_string, std::string const &ending) {
    if(full_string.length() >= ending.length()) {
        return (0 == full_string.compare(full_string.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void save_rgb_image(py::object img, const std::string &path)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    std::string lowered_path = path;
    std::transform(lowered_path.begin(), lowered_path.end(), lowered_path.begin(), ::tolower);

    if(has_ending(lowered_path, ".bmp")) {
        save_bmp(numpy_rgb_image(img), path);
    } else if(has_ending(lowered_path, ".dng")) {
        save_dng(numpy_rgb_image(img), path);
    } else if(has_ending(lowered_path, ".png")) {
        save_png(numpy_rgb_image(img), path);
    } else if(has_ending(lowered_path, ".jpg") || has_ending(lowered_path, ".jpeg")) {
        save_jpeg(numpy_rgb_image(img), path);
    } else {
        throw dlib::error("Unsupported image type, image path must end with one of [.bmp, .png, .dng, .jpg, .jpeg]");
    }
    return;
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
        
        // Convert image to Numpy array
        py::array_t<uint8_t> arr = convert_to_numpy(std::move(crop));
                
        // Append image to jittered image list
        jitter_list.append(arr);
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

    if (faces.size() < 1) {
        throw dlib::error("No face were specified in the faces array.");
    }

    py::list chips_list;

    std::vector<chip_details> dets;
    for (auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_rgb_image(img), dets, face_chips);

    for (auto& chip : face_chips) 
    {
        // Convert image to Numpy array
        py::array_t<uint8_t> arr = convert_to_numpy(std::move(chip));

        // Append image to chips list
        chips_list.append(arr);
    }
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
    return convert_to_numpy(std::move(chip));
}

// ----------------------------------------------------------------------------------------

void bind_numpy_returns(py::module &m)
{
    m.def("load_rgb_image", &load_rgb_image, 
	"Takes a path and returns a numpy array (RGB) containing the image",
	py::arg("path")
    );

    m.def("save_rgb_image", &save_rgb_image, 
	"Saves the given (RGB) image to the specified path. Determines the file type from the file extension specified in the path",
	py::arg("img"), py::arg("path")
    );

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
