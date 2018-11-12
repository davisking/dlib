// Copyright (C) 2015 Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "simple_object_detector.h"
#include "simple_object_detector_py.h"
#include "conversion.h"

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

string print_simple_test_results(const simple_test_results& r)
{
    std::ostringstream sout;
    sout << "precision: "<<r.precision << ", recall: "<< r.recall << ", average precision: " << r.average_precision;
    return sout.str();
}

// ----------------------------------------------------------------------------------------

inline simple_object_detector_py train_simple_object_detector_on_images_py (
    const py::list& pyimages,
    const py::list& pyboxes,
    const simple_object_detector_training_options& options
)
{
    const unsigned long num_images = py::len(pyimages);
    if (num_images != py::len(pyboxes))
        throw dlib::error("The length of the boxes list must match the length of the images list.");

    // We never have any ignore boxes for this version of the API.
    std::vector<std::vector<rectangle>> ignore(num_images), boxes(num_images);
    dlib::array<numpy_image<rgb_pixel>> images(num_images);
    images_and_nested_params_to_dlib(pyimages, pyboxes, images, boxes);

    return train_simple_object_detector_on_images("", images, boxes, ignore, options);
}

inline simple_test_results test_simple_object_detector_with_images_py (
        const py::list& pyimages,
        const py::list& pyboxes,
        simple_object_detector& detector,
        const unsigned int upsampling_amount
)
{
    const unsigned long num_images = py::len(pyimages);
    if (num_images != py::len(pyboxes))
        throw dlib::error("The length of the boxes list must match the length of the images list.");

    // We never have any ignore boxes for this version of the API.
    std::vector<std::vector<rectangle>> ignore(num_images), boxes(num_images);
    dlib::array<numpy_image<rgb_pixel>> images(num_images);
    images_and_nested_params_to_dlib(pyimages, pyboxes, images, boxes);

    return test_simple_object_detector_with_images(images, upsampling_amount, boxes, ignore, detector);
}

// ----------------------------------------------------------------------------------------

inline simple_test_results test_simple_object_detector_py_with_images_py (
        const py::list& pyimages,
        const py::list& pyboxes,
        simple_object_detector_py& detector,
        const int upsampling_amount
)
{
    // Allow users to pass an upsampling amount ELSE use the one cached on the object
    // Anything less than 0 is ignored and the cached value is used.
    unsigned int final_upsampling_amount = 0;
    if (upsampling_amount >= 0)
        final_upsampling_amount = upsampling_amount;
    else
        final_upsampling_amount = detector.upsampling_amount;

    return test_simple_object_detector_with_images_py(pyimages, pyboxes, detector.detector, final_upsampling_amount);
}

// ----------------------------------------------------------------------------------------

inline void find_candidate_object_locations_py (
    py::array pyimage,
    py::list& pyboxes,
    py::tuple pykvals,
    unsigned long min_size,
    unsigned long max_merging_iterations
)
{
    if (py::len(pykvals) != 3)
        throw dlib::error("kvals must be a tuple with three elements for start, end, num.");

    double start = pykvals[0].cast<double>();
    double end   = pykvals[1].cast<double>();
    long num     = pykvals[2].cast<long>();
    matrix_range_exp<double> kvals = linspace(start, end, num);

    std::vector<rectangle> rects;
    const long count = py::len(pyboxes);
    // Copy any rectangles in the input pyboxes into rects so that any rectangles will be
    // properly deduped in the resulting output.
    for (long i = 0; i < count; ++i)
        rects.push_back(pyboxes[i].cast<rectangle>());
    // Find candidate objects
    if (is_image<unsigned char>(pyimage))
        find_candidate_object_locations(numpy_image<unsigned char>(pyimage), rects, kvals, min_size, max_merging_iterations);
    else if (is_image<rgb_pixel>(pyimage))
        find_candidate_object_locations(numpy_image<rgb_pixel>(pyimage), rects, kvals, min_size, max_merging_iterations);
    else
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");

    // Collect boxes containing candidate objects
    std::vector<rectangle>::iterator iter;
    for (iter = rects.begin(); iter != rects.end(); ++iter)
        pyboxes.append(*iter);
}

// ----------------------------------------------------------------------------------------

std::shared_ptr<simple_object_detector_py> merge_simple_object_detectors (
    const py::list& detectors
)
{
    DLIB_CASSERT(len(detectors) > 0);
    std::vector<simple_object_detector> temp;
    for (const auto& d : detectors)
        temp.push_back(d.cast<simple_object_detector_py>().detector);

    simple_object_detector_py result;
    result.detector = simple_object_detector(temp);
    result.upsampling_amount = detectors[0].cast<simple_object_detector_py>().upsampling_amount;
    return std::make_shared<simple_object_detector_py>(result);
}

// ----------------------------------------------------------------------------------------

void bind_object_detection(py::module& m)
{
    {
    typedef simple_object_detector_training_options type;
    py::class_<type>(m, "simple_object_detector_training_options",
        "This object is a container for the options to the train_simple_object_detector() routine.")
        .def(py::init())
        .def("__str__", &::print_simple_object_detector_training_options)
        .def("__repr__", &::print_simple_object_detector_training_options)
        .def_readwrite("be_verbose", &type::be_verbose,
"If true, train_simple_object_detector() will print out a lot of information to the screen while training.")
        .def_readwrite("add_left_right_image_flips", &type::add_left_right_image_flips,
"if true, train_simple_object_detector() will assume the objects are \n\
left/right symmetric and add in left right flips of the training \n\
images.  This doubles the size of the training dataset.")
        .def_readwrite("detection_window_size", &type::detection_window_size,
                                               "The sliding window used will have about this many pixels inside it.")
        .def_readwrite("nuclear_norm_regularization_strength", &type::nuclear_norm_regularization_strength,
"This detector works by convolving a filter over a HOG feature image.  If that \n\
filter is separable then the convolution can be performed much faster.  The \n\
nuclear_norm_regularization_strength parameter encourages the machine learning \n\
algorithm to learn a separable filter.  A value of 0 disables this feature, but \n\
any non-zero value places a nuclear norm regularizer on the objective function \n\
and this encourages the learning of a separable filter.  Note that setting \n\
nuclear_norm_regularization_strength to a non-zero value can make the training \n\
process take significantly longer, so be patient when using it." 
            /*!
            This detector works by convolving a filter over a HOG feature image.  If that
            filter is separable then the convolution can be performed much faster.  The
            nuclear_norm_regularization_strength parameter encourages the machine learning
            algorithm to learn a separable filter.  A value of 0 disables this feature, but
            any non-zero value places a nuclear norm regularizer on the objective function
            and this encourages the learning of a separable filter.  Note that setting
            nuclear_norm_regularization_strength to a non-zero value can make the training
            process take significantly longer, so be patient when using it.
            !*/
                                               )
        .def_readwrite("max_runtime_seconds", &type::max_runtime_seconds,
            "Don't let the solver run for longer than this many seconds.")
        .def_readwrite("C", &type::C,
"C is the usual SVM C regularization parameter.  So it is passed to \n\
structural_object_detection_trainer::set_c().  Larger values of C \n\
will encourage the trainer to fit the data better but might lead to \n\
overfitting.  Therefore, you must determine the proper setting of \n\
this parameter experimentally.")
        .def_readwrite("epsilon", &type::epsilon,
"epsilon is the stopping epsilon.  Smaller values make the trainer's \n\
solver more accurate but might take longer to train.")
        .def_readwrite("num_threads", &type::num_threads,
"train_simple_object_detector() will use this many threads of \n\
execution.  Set this to the number of CPU cores on your machine to \n\
obtain the fastest training speed.")
        .def_readwrite("upsample_limit", &type::upsample_limit,
"train_simple_object_detector() will upsample images if needed \n\
no more than upsample_limit times. Value 0 will forbid trainer to \n\
upsample any images. If trainer is unable to fit all boxes with \n\
required upsample_limit, exception will be thrown. Higher values \n\
of upsample_limit exponentially increases memory requirements. \n\
Values higher than 2 (default) are not recommended.");
    }

    {
    typedef simple_test_results type;
    py::class_<type>(m, "simple_test_results")
        .def_readwrite("precision", &type::precision)
        .def_readwrite("recall", &type::recall)
        .def_readwrite("average_precision", &type::average_precision)
        .def("__str__", &::print_simple_test_results)
        .def("__repr__", &::print_simple_test_results);
    }

    // Here, kvals is actually the result of linspace(start, end, num) and it is different from kvals used
    // in find_candidate_object_locations(). See dlib/image_transforms/segment_image_abstract.h for more details.
    m.def("find_candidate_object_locations", find_candidate_object_locations_py, py::arg("image"), py::arg("rects"), py::arg("kvals")=py::make_tuple(50, 200, 3), py::arg("min_size")=20, py::arg("max_merging_iterations")=50,
"Returns found candidate objects\n\
requires\n\
    - image == an image object which is a numpy ndarray\n\
    - len(kvals) == 3\n\
    - kvals should be a tuple that specifies the range of k values to use.  In\n\
      particular, it should take the form (start, end, num) where num > 0. \n\
ensures\n\
    - This function takes an input image and generates a set of candidate\n\
      rectangles which are expected to bound any objects in the image.  It does\n\
      this by running a version of the segment_image() routine on the image and\n\
      then reports rectangles containing each of the segments as well as rectangles\n\
      containing unions of adjacent segments.  The basic idea is described in the\n\
      paper: \n\
          Segmentation as Selective Search for Object Recognition by Koen E. A. van de Sande, et al.\n\
      Note that this function deviates from what is described in the paper slightly. \n\
      See the code for details.\n\
    - The basic segmentation is performed kvals[2] times, each time with the k parameter\n\
      (see segment_image() and the Felzenszwalb paper for details on k) set to a different\n\
      value from the range of numbers linearly spaced between kvals[0] to kvals[1].\n\
    - When doing the basic segmentations prior to any box merging, we discard all\n\
      rectangles that have an area < min_size.  Therefore, all outputs and\n\
      subsequent merged rectangles are built out of rectangles that contain at\n\
      least min_size pixels.  Note that setting min_size to a smaller value than\n\
      you might otherwise be interested in using can be useful since it allows a\n\
      larger number of possible merged boxes to be created.\n\
    - There are max_merging_iterations rounds of neighboring blob merging.\n\
      Therefore, this parameter has some effect on the number of output rectangles\n\
      you get, with larger values of the parameter giving more output rectangles.\n\
    - This function appends the output rectangles into #rects.  This means that any\n\
      rectangles in rects before this function was called will still be in there\n\
      after it terminates.  Note further that #rects will not contain any duplicate\n\
      rectangles.  That is, for all valid i and j where i != j it will be true\n\
      that:\n\
        - #rects[i] != rects[j]");

    m.def("get_frontal_face_detector", get_frontal_face_detector,
        "Returns the default face detector");

    m.def("train_simple_object_detector", train_simple_object_detector,
        py::arg("dataset_filename"), py::arg("detector_output_filename"), py::arg("options"),
"requires \n\
    - options.C > 0 \n\
ensures \n\
    - Uses the structural_object_detection_trainer to train a \n\
      simple_object_detector based on the labeled images in the XML file \n\
      dataset_filename.  This function assumes the file dataset_filename is in the \n\
      XML format produced by dlib's save_image_dataset_metadata() routine. \n\
    - This function will apply a reasonable set of default parameters and \n\
      preprocessing techniques to the training procedure for simple_object_detector \n\
      objects.  So the point of this function is to provide you with a very easy \n\
      way to train a basic object detector.   \n\
    - The trained object detector is serialized to the file detector_output_filename.");

    m.def("train_simple_object_detector", train_simple_object_detector_on_images_py,
        py::arg("images"), py::arg("boxes"), py::arg("options"),
"requires \n\
    - options.C > 0 \n\
    - len(images) == len(boxes) \n\
    - images should be a list of numpy matrices that represent images, either RGB or grayscale. \n\
    - boxes should be a list of lists of dlib.rectangle object. \n\
ensures \n\
    - Uses the structural_object_detection_trainer to train a \n\
      simple_object_detector based on the labeled images and bounding boxes.  \n\
    - This function will apply a reasonable set of default parameters and \n\
      preprocessing techniques to the training procedure for simple_object_detector \n\
      objects.  So the point of this function is to provide you with a very easy \n\
      way to train a basic object detector.   \n\
    - The trained object detector is returned.");

    m.def("test_simple_object_detector", test_simple_object_detector,
        py::arg("dataset_filename"), py::arg("detector_filename"), py::arg("upsampling_amount")=-1,
            "ensures \n\
                - Loads an image dataset from dataset_filename.  We assume dataset_filename is \n\
                  a file using the XML format written by save_image_dataset_metadata(). \n\
                - Loads a simple_object_detector from the file detector_filename.  This means \n\
                  detector_filename should be a file produced by the train_simple_object_detector()  \n\
                  routine. \n\
                - This function tests the detector against the dataset and returns the \n\
                  precision, recall, and average precision of the detector.  In fact, The \n\
                  return value of this function is identical to that of dlib's \n\
                  test_object_detection_function() routine.  Therefore, see the documentation \n\
                  for test_object_detection_function() for a detailed definition of these \n\
                  metrics. \n\
                - if upsampling_amount>=0 then we upsample the data by upsampling_amount rather than \n\
                  use any upsampling amount that happens to be encoded in the given detector.  If upsampling_amount<0 \n\
                  then we use the upsampling amount the detector wants to use."
        );

    m.def("test_simple_object_detector", test_simple_object_detector2,
        py::arg("dataset_filename"), py::arg("detector"), py::arg("upsampling_amount")=-1,
            "ensures \n\
                - Loads an image dataset from dataset_filename.  We assume dataset_filename is \n\
                  a file using the XML format written by save_image_dataset_metadata(). \n\
                - Loads a simple_object_detector from the file detector_filename.  This means \n\
                  detector_filename should be a file produced by the train_simple_object_detector()  \n\
                  routine. \n\
                - This function tests the detector against the dataset and returns the \n\
                  precision, recall, and average precision of the detector.  In fact, The \n\
                  return value of this function is identical to that of dlib's \n\
                  test_object_detection_function() routine.  Therefore, see the documentation \n\
                  for test_object_detection_function() for a detailed definition of these \n\
                  metrics. \n\
                - if upsampling_amount>=0 then we upsample the data by upsampling_amount rather than \n\
                  use any upsampling amount that happens to be encoded in the given detector.  If upsampling_amount<0 \n\
                  then we use the upsampling amount the detector wants to use."
        );

    m.def("test_simple_object_detector", test_simple_object_detector_with_images_py,
            py::arg("images"), py::arg("boxes"), py::arg("detector"), py::arg("upsampling_amount")=0,
            "requires \n\
               - len(images) == len(boxes) \n\
               - images should be a list of numpy matrices that represent images, either RGB or grayscale. \n\
               - boxes should be a list of lists of dlib.rectangle object. \n\
               - Optionally, take the number of times to upsample the testing images (upsampling_amount >= 0). \n\
             ensures \n\
               - Loads a simple_object_detector from the file detector_filename.  This means \n\
                 detector_filename should be a file produced by the train_simple_object_detector() \n\
                 routine. \n\
               - This function tests the detector against the dataset and returns the \n\
                 precision, recall, and average precision of the detector.  In fact, The \n\
                 return value of this function is identical to that of dlib's \n\
                 test_object_detection_function() routine.  Therefore, see the documentation \n\
                 for test_object_detection_function() for a detailed definition of these \n\
                 metrics. "
    );

    m.def("test_simple_object_detector", test_simple_object_detector_py_with_images_py,
            // Please see test_simple_object_detector_py_with_images_py for the reason upsampling_amount is -1
            py::arg("images"), py::arg("boxes"), py::arg("detector"), py::arg("upsampling_amount")=-1,
            "requires \n\
               - len(images) == len(boxes) \n\
               - images should be a list of numpy matrices that represent images, either RGB or grayscale. \n\
               - boxes should be a list of lists of dlib.rectangle object. \n\
             ensures \n\
               - Loads a simple_object_detector from the file detector_filename.  This means \n\
                 detector_filename should be a file produced by the train_simple_object_detector() \n\
                 routine. \n\
               - This function tests the detector against the dataset and returns the \n\
                 precision, recall, and average precision of the detector.  In fact, The \n\
                 return value of this function is identical to that of dlib's \n\
                 test_object_detection_function() routine.  Therefore, see the documentation \n\
                 for test_object_detection_function() for a detailed definition of these \n\
                 metrics. "
    );
    {
    typedef simple_object_detector type;
    py::class_<type, std::shared_ptr<type>>(m, "fhog_object_detector",
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def(py::init(&load_object_from_file<type>),
"Loads an object detector from a file that contains the output of the \n\
train_simple_object_detector() routine or a serialized C++ object of type\n\
object_detector<scan_fhog_pyramid<pyramid_down<6>>>.")
        .def("__call__", run_detector_with_upscale2, py::arg("image"), py::arg("upsample_num_times")=0,
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a list of detections.   \n\
    - Upsamples the image upsample_num_times before running the basic \n\
      detector.")
       .def_property_readonly("detection_window_height", [](const type& item){return item.get_scanner().get_detection_window_height();})
       .def_property_readonly("detection_window_width", [](const type& item){return item.get_scanner().get_detection_window_width();})
        .def_property_readonly("num_detectors", [](const type& item){return item.num_detectors();})
       .def("run", run_rect_detector, py::arg("image"), py::arg("upsample_num_times")=0, py::arg("adjust_threshold")=0.0,
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a tuple of (list of detections, list of scores, list of weight_indices).   \n\
    - Upsamples the image upsample_num_times before running the basic \n\
      detector.")
       .def_static("run_multiple", run_multiple_rect_detectors, py::arg("detectors"),  py::arg("image"), py::arg("upsample_num_times")=0, py::arg("adjust_threshold")=0.0,
"requires \n\
    - detectors is a list of detectors. \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the list of object detectors at once on the input image and returns \n\
      a tuple of (list of detections, list of scores, list of weight_indices).   \n\
    - Upsamples the image upsample_num_times before running the basic \n\
      detector.")
           .def("save", save_simple_object_detector, py::arg("detector_output_filename"), "Save a simple_object_detector to the provided path.")
           .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef simple_object_detector_py type;
    py::class_<type, std::shared_ptr<type>>(m, "simple_object_detector",
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def(py::init(&merge_simple_object_detectors), py::arg("detectors"), 
"This version of the constructor builds a simple_object_detector from a \n\
bunch of other simple_object_detectors.  It essentially packs them together \n\
so that when you run the detector it's like calling run_multiple().  Except \n\
in this case the non-max suppression is applied to them all as a group.  So \n\
unlike run_multiple(), each detector competes in the non-max suppression. \n\
 \n\
Also, the non-max suppression settings used for this whole thing are \n\
the settings used by detectors[0].  So if you have a preference,  \n\
put the detector that uses the type of non-max suppression you like first \n\
in the list." 
            /*!
                This version of the constructor builds a simple_object_detector from a
                bunch of other simple_object_detectors.  It essentially packs them together
                so that when you run the detector it's like calling run_multiple().  Except
                in this case the non-max suppression is applied to them all as a group.  So
                unlike run_multiple(), each detector competes in the non-max suppression.

                Also, the non-max suppression settings used for this whole thing are
                the settings used by detectors[0].  So if you have a preference, 
                put the detector that uses the type of non-max suppression you like first
                in the list.
            !*/
            )
        .def(py::init(&load_object_from_file<type>),
"Loads a simple_object_detector from a file that contains the output of the \n\
train_simple_object_detector() routine.")
        .def("__call__", &type::run_detector1, py::arg("image"), py::arg("upsample_num_times"),
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a list of detections.   \n\
    - Upsamples the image upsample_num_times before running the basic \n\
      detector.  If you don't know how many times you want to upsample then \n\
      don't provide a value for upsample_num_times and an appropriate \n\
      default will be used.")
        .def_property_readonly("detection_window_height", [](const type& item){return item.detector.get_scanner().get_detection_window_height();})
        .def_property_readonly("detection_window_width", [](const type& item){return item.detector.get_scanner().get_detection_window_width();})
        .def_property_readonly("num_detectors", [](const type& item){return item.detector.num_detectors();})
        .def("__call__", &type::run_detector2, py::arg("image"),
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a list of detections.")
        .def("save", save_simple_object_detector_py, py::arg("detector_output_filename"), "Save a simple_object_detector to the provided path.")
        .def_readwrite("upsampling_amount", &type::upsampling_amount, "The detector upsamples the image this many times before running.")
        .def_static("run_multiple", run_multiple_rect_detectors, py::arg("detectors"),  py::arg("image"), py::arg("upsample_num_times")=0, py::arg("adjust_threshold")=0.0,
"requires \n\
    - detectors is a list of detectors. \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - upsample_num_times >= 0 \n\
ensures \n\
    - This function runs the list of object detectors at once on the input image and returns \n\
      a tuple of (list of detections, list of scores, list of weight_indices).   \n\
    - Upsamples the image upsample_num_times before running the basic \n\
      detector.")
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }


    m.def("num_separable_filters", [](const simple_object_detector_py& obj) { return num_separable_filters(obj.detector); },
        py::arg("detector"),
        "Returns the number of separable filters necessary to represent the HOG filters in the given detector."
    );

    m.def("threshold_filter_singular_values", [](const simple_object_detector_py& obj, double thresh) {
        auto temp = obj;
        temp.detector = threshold_filter_singular_values(obj.detector, thresh);
        return temp;
    }, py::arg("detector"), py::arg("thresh"),
"requires \n\
    - thresh >= 0 \n\
ensures \n\
    - Removes all components of the filters in the given detector that have \n\
      singular values that are smaller than the given threshold.  Therefore, this \n\
      function allows you to control how many separable filters are in a detector. \n\
      In particular, as thresh gets larger the quantity \n\
      num_separable_filters(threshold_filter_singular_values(detector,thresh)) \n\
      will generally get smaller and therefore give a faster running detector. \n\
      However, note that at some point a large enough thresh will drop too much \n\
      information from the filters and their accuracy will suffer.   \n\
    - returns the updated detector" 
    /*!
        requires
            - thresh >= 0
        ensures
            - Removes all components of the filters in the given detector that have
              singular values that are smaller than the given threshold.  Therefore, this
              function allows you to control how many separable filters are in a detector.
              In particular, as thresh gets larger the quantity
              num_separable_filters(threshold_filter_singular_values(detector,thresh))
              will generally get smaller and therefore give a faster running detector.
              However, note that at some point a large enough thresh will drop too much
              information from the filters and their accuracy will suffer.  
            - returns the updated detector
    !*/
    );
}

// ----------------------------------------------------------------------------------------
