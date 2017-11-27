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

void bind_numpy_returns()
{
    using boost::python::arg;

    def("jitter_image", &get_jitter_images, get_jitter_images_with_defaults(
    "Takes an image and returns a list of jittered images."
    "The returned list contains num_jitters images (default is 1)."
    "If disturb_colors is set to True, the colors of the image are disturbed (default is False)", 
    (arg("img"), arg("num_jitters"), arg("disturb_colors"))
    ));
}