// Copyright (C) 2015 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <pybind11/pybind11.h>
#include <dlib/simd.h>
#include <string>

namespace py = pybind11;

void bind_matrix(py::module& m);
void bind_vector(py::module& m);
void bind_svm_c_trainer(py::module& m);
void bind_decision_functions(py::module& m);
void bind_basic_types(py::module& m);
void bind_other(py::module& m);
void bind_svm_rank_trainer(py::module& m);
void bind_cca(py::module& m);
void bind_sequence_segmenter(py::module& m);
void bind_svm_struct(py::module& m);
void bind_image_classes(py::module& m);
void bind_image_classes2(py::module& m);
void bind_image_classes3(py::module& m);
void bind_image_classes4(py::module& m);
void bind_rectangles(py::module& m);
void bind_object_detection(py::module& m);
void bind_shape_predictors(py::module& m);
void bind_correlation_tracker(py::module& m);
void bind_face_recognition(py::module& m);
void bind_cnn_face_detection(py::module& m);
void bind_global_optimization(py::module& m);
void bind_numpy_returns(py::module& m);
void bind_image_dataset_metadata(py::module& m);
void bind_line(py::module& m);

#ifndef DLIB_NO_GUI_SUPPORT
void bind_gui(py::module& m);
#endif

PYBIND11_MODULE(_dlib_pybind11, m)
{
    warn_about_unavailable_but_used_cpu_instructions();


#define DLIB_QUOTE_STRING(x) DLIB_QUOTE_STRING2(x)
#define DLIB_QUOTE_STRING2(x) #x
    m.attr("__version__") = DLIB_QUOTE_STRING(DLIB_VERSION);
    m.attr("__time_compiled__") = std::string(__DATE__) + " " + std::string(__TIME__);

#ifdef DLIB_USE_CUDA
    m.attr("DLIB_USE_CUDA") = true;
#else
    m.attr("DLIB_USE_CUDA") = false;
#endif
#ifdef DLIB_USE_BLAS 
    m.attr("DLIB_USE_BLAS") = true;
#else
    m.attr("DLIB_USE_BLAS") = false;
#endif
#ifdef DLIB_USE_LAPACK
    m.attr("DLIB_USE_LAPACK") = true;
#else
    m.attr("DLIB_USE_LAPACK") = false;
#endif
#ifdef DLIB_HAVE_AVX
    m.attr("USE_AVX_INSTRUCTIONS") = true;
#else
    m.attr("USE_AVX_INSTRUCTIONS") = false;
#endif
#ifdef DLIB_HAVE_NEON 
    m.attr("USE_NEON_INSTRUCTIONS") = true;
#else
    m.attr("USE_NEON_INSTRUCTIONS") = false;
#endif



    // Note that the order here matters.  We need to do the basic types first.  If we don't 
    // then what happens is the documentation created by sphinx will use horrible big
    // template names to refer to C++ objects rather than the python names python users
    // will expect.  For instance, if bind_basic_types() isn't called early then when
    // routines take a std::vector<double>, rather than saying dlib.array in the python
    // docs it will say "std::vector<double, std::allocator<double> >" which is awful and
    // confusing to python users.
    //
    // So when adding new things always add them to the end of the list.
    bind_matrix(m);
    bind_vector(m);
    bind_basic_types(m);
    bind_other(m);
    bind_line(m);

    bind_svm_rank_trainer(m);
    bind_decision_functions(m);
    bind_cca(m);
    bind_sequence_segmenter(m);
    bind_svm_struct(m);
    bind_rectangles(m);
    bind_image_classes(m);
    bind_image_classes2(m);
    bind_image_classes3(m);
    bind_image_classes4(m);
    bind_object_detection(m);
    bind_shape_predictors(m);
    bind_correlation_tracker(m);
    bind_face_recognition(m);
    bind_cnn_face_detection(m);
    bind_global_optimization(m);
    bind_numpy_returns(m);
    bind_svm_c_trainer(m);
#ifndef DLIB_NO_GUI_SUPPORT
    bind_gui(m);
#endif

    bind_image_dataset_metadata(m);


}
