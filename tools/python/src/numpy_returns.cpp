#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>
#include <dlib/image_io.h>
#include <pybind11/numpy.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;


// -------------------------------- Basic Image IO ----------------------------------------

numpy_image<rgb_pixel> load_rgb_image (const std::string &path)
{
    numpy_image<rgb_pixel> img;
    load_image(img, path);
    return img; 
}

numpy_image<unsigned char> load_grayscale_image (const std::string &path)
{
    numpy_image<unsigned char> img;
    load_image(img, path);
    return img; 
}

bool has_ending (std::string const full_string, std::string const &ending) {
    if(full_string.length() >= ending.length()) {
        return (0 == full_string.compare(full_string.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

// ----------------------------------------------------------------------------------------

template <typename T>
void save_image(numpy_image<T> img, const std::string &path)
{
    std::string lowered_path = path;
    std::transform(lowered_path.begin(), lowered_path.end(), lowered_path.begin(), ::tolower);

    if(has_ending(lowered_path, ".bmp")) {
        save_bmp(img, path);
    } else if(has_ending(lowered_path, ".dng")) {
        save_dng(img, path);
    } else if(has_ending(lowered_path, ".png")) {
        save_png(img, path);
    } else if(has_ending(lowered_path, ".jpg") || has_ending(lowered_path, ".jpeg")) {
        save_jpeg(img, path);
    } else {
        throw dlib::error("Unsupported image type, image path must end with one of [.bmp, .png, .dng, .jpg, .jpeg]");
    }
    return;
}

// ----------------------------------------------------------------------------------------

py::list get_jitter_images(numpy_image<rgb_pixel> img, size_t num_jitters = 1, bool disturb_colors = false)
{
    static dlib::rand rnd_jitter;

    // The top level list (containing 1 or more images) to return to python
    py::list jitter_list;

    for (int i = 0; i < num_jitters; ++i) {
        // Get a jittered crop
        numpy_image<rgb_pixel> crop = dlib::jitter_image(img, rnd_jitter);
        // If required disturb colors of the image
        if(disturb_colors)
            dlib::disturb_colors(crop, rnd_jitter);
        
        // Append image to jittered image list
        jitter_list.append(crop);
    }
           
    return jitter_list;
}

// ----------------------------------------------------------------------------------------

py::list get_face_chips (
    numpy_image<rgb_pixel> img,
    const std::vector<full_object_detection>& faces,
    size_t size = 150,
    float padding = 0.25
)
{

    if (faces.size() < 1) {
        throw dlib::error("No face were specified in the faces array.");
    }

    py::list chips_list;

    std::vector<chip_details> dets;
    for (const auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<numpy_image<rgb_pixel>> face_chips;
    extract_image_chips(img, dets, face_chips);

    for (const auto& chip : face_chips) 
    {
        // Append image to chips list
        chips_list.append(chip);
    }
    return chips_list;
}

numpy_image<rgb_pixel> get_face_chip (
    numpy_image<rgb_pixel> img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    numpy_image<rgb_pixel> chip;
    extract_image_chip(img, get_face_chip_details(face, size, padding), chip);
    return chip;
}

// ----------------------------------------------------------------------------------------

void bind_numpy_returns(py::module &m)
{
    m.def("load_rgb_image", &load_rgb_image, 
	"Takes a path and returns a numpy array (RGB) containing the image",
	py::arg("filename")
    );

    m.def("load_grayscale_image", &load_grayscale_image, 
	"Takes a path and returns a numpy array containing the image, as an 8bit grayscale image.",
	py::arg("filename")
    );

    m.def("save_image", &save_image<rgb_pixel>, 
	"Saves the given image to the specified path. Determines the file type from the file extension specified in the path",
	py::arg("img"), py::arg("filename")
    );
    m.def("save_image", &save_image<unsigned char>, 
	"Saves the given image to the specified path. Determines the file type from the file extension specified in the path",
	py::arg("img"), py::arg("filename")
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
