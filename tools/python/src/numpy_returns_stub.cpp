#include "opaque_types.h"
#include <dlib/python.h>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>

using namespace dlib;
using namespace std;
namespace py = pybind11;

// ----------------------------------------------------------------------------------------

py::list get_jitter_images(py::object img, size_t num_jitters = 1, bool disturb_colors = false)
{
    throw dlib::error("jitter_image is only supported if you compiled dlib with numpy installed!");
}

// ----------------------------------------------------------------------------------------

py::list get_face_chips (
    py::object img,
    const std::vector<full_object_detection>& faces,
    size_t size = 150,
    float padding = 0.25
)
{
    throw dlib::error("get_face_chips is only supported if you compiled dlib with numpy installed!");
}

py::object get_face_chip (
    py::object img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    throw dlib::error("get_face_chip is only supported if you compiled dlib with numpy installed!");
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
