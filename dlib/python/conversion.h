// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_PYTHON_CONVERSION_H__
#define DLIB_PYTHON_CONVERSION_H__

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/pixel.h>
#include <dlib/python/numpy_image.h>

using namespace dlib;
using namespace std;

namespace py = pybind11;


template <typename image_array, typename param_type>
void images_and_nested_params_to_dlib(
        const py::object& pyimages,
        const py::object& pyparams,
        image_array& images,
        std::vector<std::vector<param_type>>& params
)
{
    // Now copy the data into dlib based objects.
    py::iterator image_it = pyimages.begin();
    py::iterator params_it = pyparams.begin();

    for (unsigned long image_idx = 0; image_it != pyimages.end() && params_it != pyparams.end(); ++image_it, ++params_it, ++image_idx)
    {
        for (py::iterator param_it = params_it->begin(); param_it != params_it->end(); ++param_it)
            params[image_idx].push_back(param_it->cast<param_type>());

        assign_image(images[image_idx], image_it->cast<py::array>());
    }
}

#endif // DLIB_PYTHON_CONVERSION_H__
