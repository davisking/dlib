#include <dlib/python.h>
#include <boost/python/args.hpp>
#include "dlib/pixel.h"
#include <dlib/image_transforms.h>

using namespace dlib;
using namespace std;
using namespace boost::python;

// ----------------------------------------------------------------------------------------

boost::python::list get_jitter_images(object img, size_t num_jitters = 1, bool disturb_colors = false)
{
    throw dlib::error("jitter_image is only supported if you compiled dlib with numpy installed!");
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
    throw dlib::error("get_face_chips is only supported if you compiled dlib with numpy installed!");
}

object get_face_chip (
    object img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    throw dlib::error("get_face_chip is only supported if you compiled dlib with numpy installed!");
}


BOOST_PYTHON_FUNCTION_OVERLOADS(get_face_chip_with_defaults, get_face_chip, 2, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(get_face_chips_with_defaults, get_face_chips, 2, 4)

// ----------------------------------------------------------------------------------------

void bind_numpy_returns()
{
    using boost::python::arg;

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