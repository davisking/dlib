// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYTHON_CONVERSION_H__
#define DLIB_PYTHON_CONVERSION_H__

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/pixel.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;

template <typename dest_image_type>
void pyimage_to_dlib_image(py::object img, dest_image_type& image)
{
    if (is_gray_python_image(img))
        assign_image(image, numpy_gray_image(img));
    else if (is_rgb_python_image(img))
        assign_image(image, numpy_rgb_image(img));
    else
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
}

template <typename image_array, typename param_type>
void images_and_nested_params_to_dlib(
        const py::object& pyimages,
        const py::object& pyparams,
        image_array& images,
        std::vector<std::vector<param_type> >& params
)
{
    // Now copy the data into dlib based objects.
    py::iterator image_it = pyimages.begin();
    py::iterator params_it = pyparams.begin();

    for (unsigned long image_idx = 0;
         image_it != pyimages.end()
           && params_it != pyparams.end();
         ++image_it, ++params_it, ++image_idx)
    {
        for (py::iterator param_it = params_it->begin();
             param_it != params_it->end();
             ++param_it)
          params[image_idx].push_back(param_it->cast<param_type>());

        pyimage_to_dlib_image(image_it->cast<py::object>(), images[image_idx]);
    }
}

#endif // DLIB_PYTHON_CONVERSION_H__
