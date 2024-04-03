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

numpy_image<rgb_alpha_pixel> load_rgb_alpha_image (const std::string &path)
{
    numpy_image<rgb_alpha_pixel> img;
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
void save_image(numpy_image<T> img, const std::string &path, const float quality)
{
    std::string lowered_path = path;
    std::transform(lowered_path.begin(), lowered_path.end(), lowered_path.begin(), ::tolower);
    std::string error_message = "Unsupported image type, image path must end with one of [.bmp, .dng";
#if DLIB_PNG_SUPPORT
    error_message += ", .png";
#endif
#if DLIB_JPEG_SUPPORT
    error_message += ", .jpg, jpeg";
#endif
#if DLIB_WEBP_SUPPORT
    error_message += ", .webp";
#endif
#if DLIB_JXL_SUPPORT
    error_message += ", .jxl";
#endif
    error_message += "]";

    if(has_ending(lowered_path, ".bmp")) {
        save_bmp(img, path);
    } else if(has_ending(lowered_path, ".dng")) {
        save_dng(img, path);
#if DLIB_PNG_SUPPORT
    } else if(has_ending(lowered_path, ".png")) {
        save_png(img, path);
#endif
#if DLIB_JPEG_SUPPORT
    } else if(has_ending(lowered_path, ".jpg") || has_ending(lowered_path, ".jpeg")) {
        save_jpeg(img, path, put_in_range(0, 100, std::lround(quality)));
#endif
#if DLIB_WEBP_SUPPORT
    } else if(has_ending(lowered_path, ".webp")) {
        save_webp(img, path, std::max(0.f, quality));
#endif
#if DLIB_JXL_SUPPORT
    } else if(has_ending(lowered_path, ".jxl")) {
        save_jxl(img, path, put_in_range(0, 100, quality));
#endif
    } else {
        throw dlib::error(error_message);
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

template <typename T>
py::list get_face_chips (
    numpy_image<T> img,
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
    dlib::array<numpy_image<T>> face_chips;
    extract_image_chips(img, dets, face_chips);

    for (const auto& chip : face_chips) 
    {
        // Append image to chips list
        chips_list.append(chip);
    }
    return chips_list;
}

template <typename T>
numpy_image<T> get_face_chip (
    numpy_image<T> img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    numpy_image<T> chip;
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

    m.def("load_rgb_alpha_image", &load_rgb_alpha_image,
	"Takes a path and returns a numpy array (RGBA) containing the image",
	py::arg("filename")
    );

    m.def("load_grayscale_image", &load_grayscale_image, 
	"Takes a path and returns a numpy array containing the image, as an 8bit grayscale image.",
	py::arg("filename")
    );

    m.def("save_image", &save_image<rgb_pixel>, 
	"Saves the given image to the specified path. Determines the file type from the file extension specified in the path",
	py::arg("img"), py::arg("filename"), py::arg("quality") = 75
    );
    m.def("save_image", &save_image<rgb_alpha_pixel>,
	"Saves the given image to the specified path. Determines the file type from the file extension specified in the path",
	py::arg("img"), py::arg("filename"), py::arg("quality") = 75
    );
    m.def("save_image", &save_image<unsigned char>, 
	"Saves the given image to the specified path. Determines the file type from the file extension specified in the path",
	py::arg("img"), py::arg("filename"), py::arg("quality") = 75
    );

    m.def("jitter_image", &get_jitter_images, 
    "Takes an image and returns a list of jittered images."
    "The returned list contains num_jitters images (default is 1)."
    "If disturb_colors is set to True, the colors of the image are disturbed (default is False)", 
    py::arg("img"), py::arg("num_jitters")=1, py::arg("disturb_colors")=false
    );

    {
        const char* docs = "Takes an image and a full_object_detection that references a face in that image and returns the face as a Numpy array representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.";
        m.def("get_face_chip", &get_face_chip<uint8_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<uint16_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<uint32_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<uint64_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<int8_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<int16_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<int32_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<int64_t>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<float>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<double>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chip", &get_face_chip<rgb_pixel>, docs, py::arg("img"), py::arg("face"), py::arg("size")=150, py::arg("padding")=0.25);
    }

    {
        const char* docs = "Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of Numpy arrays representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.";
        m.def("get_face_chips", &get_face_chips<uint8_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<uint16_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<uint32_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<uint64_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<int8_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<int16_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<int32_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<int64_t>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<float>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<double>, docs, py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
        m.def("get_face_chips", &get_face_chips<rgb_pixel>, docs,py::arg("img"), py::arg("faces"), py::arg("size")=150, py::arg("padding")=0.25);
    }
}
