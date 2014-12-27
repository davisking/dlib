// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <dlib/matrix.h>
#include <boost/python/args.hpp>
#include <dlib/geometry.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "indexing.h"
#include "simple_object_detector.h"
#include "simple_object_detector_py.h"
#include "conversion.h"

using namespace dlib;
using namespace std;
using namespace boost::python;

// ----------------------------------------------------------------------------------------

string print_simple_test_results(const simple_test_results& r)
{
    std::ostringstream sout;
    sout << "precision: "<<r.precision << ", recall: "<< r.recall << ", average precision: " << r.average_precision;
    return sout.str();
}

// ----------------------------------------------------------------------------------------

long left(const rectangle& r) { return r.left(); }
long top(const rectangle& r) { return r.top(); }
long right(const rectangle& r) { return r.right(); }
long bottom(const rectangle& r) { return r.bottom(); }
long width(const rectangle& r) { return r.width(); }
long height(const rectangle& r) { return r.height(); }

string print_rectangle_str(const rectangle& r)
{
    std::ostringstream sout;
    sout << r;
    return sout.str();
}

string print_rectangle_repr(const rectangle& r)
{
    std::ostringstream sout;
    sout << "rectangle(" << r.left() << "," << r.top() << "," << r.right() << "," << r.bottom() << ")";
    return sout.str();
}

// ----------------------------------------------------------------------------------------

inline simple_object_detector_py train_simple_object_detector_on_images_py (
    const boost::python::list& pyimages,
    const boost::python::list& pyboxes,
    const simple_object_detector_training_options& options
)
{
    const unsigned long num_images = len(pyimages);
    if (num_images != len(pyboxes))
        throw dlib::error("The length of the boxes list must match the length of the images list.");

    // We never have any ignore boxes for this version of the API.
    std::vector<std::vector<rectangle> > ignore(num_images), boxes(num_images);
    dlib::array<array2d<rgb_pixel> > images(num_images);
    images_and_nested_params_to_dlib(pyimages, pyboxes, images, boxes);

    return train_simple_object_detector_on_images("", images, boxes, ignore, options);
}

inline simple_test_results test_simple_object_detector_with_images_py (
        const boost::python::list& pyimages,
        const boost::python::list& pyboxes,
        simple_object_detector& detector,
        const unsigned int upsampling_amount
)
{
    const unsigned long num_images = len(pyimages);
    if (num_images != len(pyboxes))
        throw dlib::error("The length of the boxes list must match the length of the images list.");

    // We never have any ignore boxes for this version of the API.
    std::vector<std::vector<rectangle> > ignore(num_images), boxes(num_images);
    dlib::array<array2d<rgb_pixel> > images(num_images);
    images_and_nested_params_to_dlib(pyimages, pyboxes, images, boxes);

    return test_simple_object_detector_with_images(images, upsampling_amount, boxes, ignore, detector);
}

// ----------------------------------------------------------------------------------------

inline simple_test_results test_simple_object_detector_py_with_images_py (
        const boost::python::list& pyimages,
        const boost::python::list& pyboxes,
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

void bind_object_detection()
{
    using boost::python::arg;

    class_<simple_object_detector_training_options>("simple_object_detector_training_options", 
        "This object is a container for the options to the train_simple_object_detector() routine.")
        .add_property("be_verbose", &simple_object_detector_training_options::be_verbose, 
                                    &simple_object_detector_training_options::be_verbose,
"If true, train_simple_object_detector() will print out a lot of information to the screen while training.")
        .add_property("add_left_right_image_flips", &simple_object_detector_training_options::add_left_right_image_flips, 
                                                    &simple_object_detector_training_options::add_left_right_image_flips,
"if true, train_simple_object_detector() will assume the objects are \n\
left/right symmetric and add in left right flips of the training \n\
images.  This doubles the size of the training dataset.")
        .add_property("detection_window_size", &simple_object_detector_training_options::detection_window_size,
                                               &simple_object_detector_training_options::detection_window_size,
                                               "The sliding window used will have about this many pixels inside it.")
        .add_property("C", &simple_object_detector_training_options::C,
                           &simple_object_detector_training_options::C,
"C is the usual SVM C regularization parameter.  So it is passed to \n\
structural_object_detection_trainer::set_c().  Larger values of C \n\
will encourage the trainer to fit the data better but might lead to \n\
overfitting.  Therefore, you must determine the proper setting of \n\
this parameter experimentally.")
        .add_property("epsilon", &simple_object_detector_training_options::epsilon,
                                 &simple_object_detector_training_options::epsilon,
"epsilon is the stopping epsilon.  Smaller values make the trainer's \n\
solver more accurate but might take longer to train.")
        .add_property("num_threads", &simple_object_detector_training_options::num_threads,
                                     &simple_object_detector_training_options::num_threads,
"train_simple_object_detector() will use this many threads of \n\
execution.  Set this to the number of CPU cores on your machine to \n\
obtain the fastest training speed.");

    class_<simple_test_results>("simple_test_results")
        .add_property("precision", &simple_test_results::precision)
        .add_property("recall", &simple_test_results::recall)
        .add_property("average_precision", &simple_test_results::average_precision)
        .def("__str__", &::print_simple_test_results);
    {
    typedef rectangle type;
    class_<type>("rectangle", "This object represents a rectangular area of an image.")
        .def(init<long,long,long,long>( (arg("left"),arg("top"),arg("right"),arg("bottom")) ))
        .def("left",   &::left)
        .def("top",    &::top)
        .def("right",  &::right)
        .def("bottom", &::bottom)
        .def("width",  &::width)
        .def("height", &::height)
        .def("__str__", &::print_rectangle_str)
        .def("__repr__", &::print_rectangle_repr)
        .def_pickle(serialize_pickle<type>());
    }

    def("get_frontal_face_detector", get_frontal_face_detector, 
        "Returns the default face detector");

    def("train_simple_object_detector", train_simple_object_detector,
        (arg("dataset_filename"), arg("detector_output_filename"), arg("options")),
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

    def("train_simple_object_detector", train_simple_object_detector_on_images_py,
        (arg("images"), arg("boxes"), arg("options")),
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

    def("test_simple_object_detector", test_simple_object_detector,
            // Please see test_simple_object_detector for the reason upsampling_amount is -1
            (arg("dataset_filename"), arg("detector_filename"), arg("upsampling_amount")=-1),
            "requires \n\
                - Optionally, take the number of times to upsample the testing images (upsampling_amount >= 0). \n\
             ensures \n\
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
                  metrics. "
        );

    def("test_simple_object_detector", test_simple_object_detector_with_images_py,
            (arg("images"), arg("boxes"), arg("detector"), arg("upsampling_amount")=0),
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

    def("test_simple_object_detector", test_simple_object_detector_py_with_images_py,
            // Please see test_simple_object_detector_py_with_images_py for the reason upsampling_amount is -1
            (arg("images"), arg("boxes"), arg("detector"), arg("upsampling_amount")=-1),
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
    class_<type>("fhog_object_detector",
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def("__init__", make_constructor(&load_object_from_file<type>),  
"Loads an object detector from a file that contains the output of the \n\
train_simple_object_detector() routine or a serialized C++ object of type\n\
object_detector<scan_fhog_pyramid<pyramid_down<6>>>.")
        .def("__call__", run_detector_with_upscale, (arg("image"), arg("upsample_num_times")=0),
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
        .def("save", save_simple_object_detector, (arg("detector_output_filename")), "Save a simple_object_detector to the provided path.")
        .def_pickle(serialize_pickle<type>());
    }
    {
    typedef simple_object_detector_py type;
    class_<type>("simple_object_detector",
        "This object represents a sliding window histogram-of-oriented-gradients based object detector.")
        .def("__init__", make_constructor(&load_object_from_file<type>),
"Loads a simple_object_detector from a file that contains the output of the \n\
train_simple_object_detector() routine.")
        .def("__call__", &type::run_detector1, (arg("image"), arg("upsample_num_times")),
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
        .def("__call__", &type::run_detector2, (arg("image")),
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
ensures \n\
    - This function runs the object detector on the input image and returns \n\
      a list of detections.")
        .def("save", save_simple_object_detector_py, (arg("detector_output_filename")), "Save a simple_object_detector to the provided path.")
        .def_pickle(serialize_pickle<type>());
    }
    {
    typedef std::vector<rectangle> type;
    class_<type>("rectangles", "An array of rectangle objects.")
        .def(vector_indexing_suite<type>())
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def_pickle(serialize_pickle<type>());
    }
}

// ----------------------------------------------------------------------------------------
