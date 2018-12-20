// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/geometry.h>
#include <dlib/image_processing.h>
#include "shape_predictor.h"
#include "conversion.h"

using namespace dlib;
using namespace std;

namespace py = pybind11;

// ----------------------------------------------------------------------------------------

full_object_detection run_predictor (
        shape_predictor& predictor,
        py::array img,
        const rectangle& box
)
{
    if (is_image<unsigned char>(img))
    {
        return predictor(numpy_image<unsigned char>(img), box);
    }
    else if (is_image<rgb_pixel>(img))
    {
        return predictor(numpy_image<rgb_pixel>(img), box);
    }
    else
    {
        throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");
    }
}

void save_shape_predictor(const shape_predictor& predictor, const std::string& predictor_output_filename)
{
    std::ofstream fout(predictor_output_filename.c_str(), std::ios::binary);
    serialize(predictor, fout);
}

// ----------------------------------------------------------------------------------------

rectangle full_obj_det_get_rect (const full_object_detection& detection)
{ return detection.get_rect(); }

unsigned long full_obj_det_num_parts (const full_object_detection& detection)
{ return detection.num_parts(); }

point full_obj_det_part (const full_object_detection& detection, const unsigned long idx)
{
    if (idx >= detection.num_parts())
    {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        throw py::error_already_set();
    }
    return detection.part(idx);
}

std::vector<point> full_obj_det_parts (const full_object_detection& detection)
{
    const unsigned long num_parts = detection.num_parts();
    std::vector<point> parts(num_parts);
    for (unsigned long j = 0; j < num_parts; ++j)
        parts[j] = detection.part(j);
    return parts;
}

std::shared_ptr<full_object_detection> full_obj_det_init(const rectangle& rect, py::list& pyparts)
{
    const unsigned long num_parts = py::len(pyparts);
    std::vector<point> parts;
    for (const auto& item : pyparts)
        parts.push_back(item.cast<point>());

    return std::make_shared<full_object_detection>(rect, parts);
}

// ----------------------------------------------------------------------------------------

inline shape_predictor train_shape_predictor_on_images_py (
        const py::list& pyimages,
        const py::list& pydetections,
        const shape_predictor_training_options& options
)
{
    const unsigned long num_images = py::len(pyimages);
    if (num_images != py::len(pydetections))
        throw dlib::error("The length of the detections list must match the length of the images list.");

    std::vector<std::vector<full_object_detection> > detections(num_images);
    dlib::array<numpy_image<unsigned char>> images(num_images);
    images_and_nested_params_to_dlib(pyimages, pydetections, images, detections);

    return train_shape_predictor_on_images(images, detections, options);
}


inline double test_shape_predictor_with_images_py (
        const py::list& pyimages,
        const py::list& pydetections,
        const py::list& pyscales,
        const shape_predictor& predictor
)
{
    const unsigned long num_images = py::len(pyimages);
    const unsigned long num_scales = py::len(pyscales);
    if (num_images != py::len(pydetections))
        throw dlib::error("The length of the detections list must match the length of the images list.");

    if (num_scales > 0 && num_scales != num_images)
        throw dlib::error("The length of the scales list must match the length of the detections list.");

    std::vector<std::vector<full_object_detection> > detections(num_images);
    std::vector<std::vector<double> > scales;
    if (num_scales > 0)
        scales.resize(num_scales);
    dlib::array<numpy_image<unsigned char>> images(num_images);

    // Now copy the data into dlib based objects so we can call the testing routine.
    for (unsigned long i = 0; i < num_images; ++i)
    {
        const unsigned long num_boxes = py::len(pydetections[i]);
        for (py::iterator det_it = pydetections[i].begin();
             det_it != pydetections[i].end();
             ++det_it)
          detections[i].push_back(det_it->cast<full_object_detection>());

        assign_image(images[i], pyimages[i].cast<py::array>());
        if (num_scales > 0)
        {
            if (num_boxes != py::len(pyscales[i]))
                throw dlib::error("The length of the scales list must match the length of the detections list.");
            for (py::iterator scale_it = pyscales[i].begin(); scale_it != pyscales[i].end(); ++scale_it)
                scales[i].push_back(scale_it->cast<double>());
        }
    }

    return test_shape_predictor_with_images(images, detections, scales, predictor);
}

inline double test_shape_predictor_with_images_no_scales_py (
        const py::list& pyimages,
        const py::list& pydetections,
        const shape_predictor& predictor
)
{
    py::list pyscales;
    return test_shape_predictor_with_images_py(pyimages, pydetections, pyscales, predictor);
}

// ----------------------------------------------------------------------------------------

void bind_shape_predictors(py::module &m)
{
    {
    typedef full_object_detection type;
    py::class_<type, std::shared_ptr<type>>(m, "full_object_detection",
    "This object represents the location of an object in an image along with the \
    positions of each of its constituent parts.")
        .def(py::init(&full_obj_det_init),
"requires \n\
    - rect: dlib rectangle \n\
    - parts: list of dlib points")
        .def_property_readonly("rect", &full_obj_det_get_rect, "Bounding box from the underlying detector. Parts can be outside box if appropriate.")
        .def_property_readonly("num_parts", &full_obj_det_num_parts, "The number of parts of the object.")
        .def("part", &full_obj_det_part, py::arg("idx"), "A single part of the object as a dlib point.")
        .def("parts", &full_obj_det_parts, "A vector of dlib points representing all of the parts.")
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef shape_predictor_training_options type;
    py::class_<type>(m, "shape_predictor_training_options",
        "This object is a container for the options to the train_shape_predictor() routine.")
        .def(py::init())
        .def_readwrite("be_verbose", &type::be_verbose,
                      "If true, train_shape_predictor() will print out a lot of information to stdout while training.")
        .def_readwrite("cascade_depth", &type::cascade_depth,
                      "The number of cascades created to train the model with.")
        .def_readwrite("tree_depth", &type::tree_depth,
                      "The depth of the trees used in each cascade. There are pow(2, get_tree_depth()) leaves in each tree")
        .def_readwrite("num_trees_per_cascade_level", &type::num_trees_per_cascade_level,
                      "The number of trees created for each cascade.")
        .def_readwrite("nu", &type::nu,
                      "The regularization parameter.  Larger values of this parameter \
                       will cause the algorithm to fit the training data better but may also \
                       cause overfitting.  The value must be in the range (0, 1].")
        .def_readwrite("oversampling_amount", &type::oversampling_amount,
                      "The number of randomly selected initial starting points sampled for each training example")
        .def_readwrite("oversampling_translation_jitter", &type::oversampling_translation_jitter,
                      "The amount of translation jittering to apply to bounding boxes, a good value is in in the range [0 0.5].")
        .def_readwrite("feature_pool_size", &type::feature_pool_size,
                      "Number of pixels used to generate features for the random trees.")
        .def_readwrite("lambda_param", &type::lambda_param,
                      "Controls how tight the feature sampling should be. Lower values enforce closer features.")
        .def_readwrite("num_test_splits", &type::num_test_splits,
                      "Number of split features at each node to sample. The one that gives the best split is chosen.")
        .def_readwrite("landmark_relative_padding_mode", &type::landmark_relative_padding_mode,
                      "If True then features are drawn only from the box around the landmarks, otherwise they come from the bounding box and landmarks together.  See feature_pool_region_padding doc for more details.")
        .def_readwrite("feature_pool_region_padding", &type::feature_pool_region_padding,
            /*!
                  This algorithm works by comparing the relative intensity of pairs of
                  pixels in the input image.  To decide which pixels to look at, the
                  training algorithm randomly selects pixels from a box roughly centered
                  around the object of interest.  We call this box the feature pool region
                  box.  
                  
                  Each object of interest is defined by a full_object_detection, which
                  contains a bounding box and a list of landmarks.  If
                  landmark_relative_padding_mode==True then the feature pool region box is
                  the tightest box that contains the landmarks inside the
                  full_object_detection.  In this mode the full_object_detection's bounding
                  box is ignored.  Otherwise, if the padding mode is bounding_box_relative
                  then the feature pool region box is the tightest box that contains BOTH
                  the landmarks and the full_object_detection's bounding box.

                  Additionally, you can adjust the size of the feature pool padding region
                  by setting feature_pool_region_padding to some value.  If
                  feature_pool_region_padding then the feature pool region box is
                  unmodified and defined exactly as stated above. However, you can expand
                  the size of the box by setting the padding > 0 or shrink it by setting it
                  to something < 0.

                  To explain this precisely, for a padding of 0 we say that the pixels are
                  sampled from a box of size 1x1.  The padding value is added to each side
                  of the box.  So a padding of 0.5 would cause the algorithm to sample
                  pixels from a box that was 2x2, effectively multiplying the area pixels
                  are sampled from by 4.  Similarly, setting the padding to -0.2 would
                  cause it to sample from a box 0.6x0.6 in size.
            !*/
                      "Size of region within which to sample features for the feature pool. \
                      positive values increase the sampling region while negative values decrease it. E.g. padding of 0 means we \
                      sample fr")
        .def_readwrite("random_seed", &type::random_seed,
                      "The random seed used by the internal random number generator")
        .def_readwrite("num_threads", &type::num_threads,
                        "Use this many threads/CPU cores for training.")
        .def("__str__", &::print_shape_predictor_training_options)
        .def("__repr__", &::print_shape_predictor_training_options)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    typedef shape_predictor type;
    py::class_<type, std::shared_ptr<type>>(m, "shape_predictor",
"This object is a tool that takes in an image region containing some object and \
outputs a set of point locations that define the pose of the object. The classic \
example of this is human face pose prediction, where you take an image of a human \
face as input and are expected to identify the locations of important facial \
landmarks such as the corners of the mouth and eyes, tip of the nose, and so forth.")
        .def(py::init())
        .def(py::init(&load_object_from_file<type>),
"Loads a shape_predictor from a file that contains the output of the \n\
train_shape_predictor() routine.")
        .def("__call__", &run_predictor, py::arg("image"), py::arg("box"),
"requires \n\
    - image is a numpy ndarray containing either an 8bit grayscale or RGB \n\
      image. \n\
    - box is the bounding box to begin the shape prediction inside. \n\
ensures \n\
    - This function runs the shape predictor on the input image and returns \n\
      a single full_object_detection.")
        .def("save", save_shape_predictor, py::arg("predictor_output_filename"), "Save a shape_predictor to the provided path.")
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }
    {
    m.def("train_shape_predictor", train_shape_predictor_on_images_py,
        py::arg("images"), py::arg("object_detections"), py::arg("options"),
"requires \n\
    - options.lambda_param > 0 \n\
    - 0 < options.nu <= 1 \n\
    - options.feature_pool_region_padding >= 0 \n\
    - len(images) == len(object_detections) \n\
    - images should be a list of numpy matrices that represent images, either RGB or grayscale. \n\
    - object_detections should be a list of lists of dlib.full_object_detection objects. \
      Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.\n\
ensures \n\
    - Uses dlib's shape_predictor_trainer object to train a \n\
      shape_predictor based on the provided labeled images, full_object_detections, and options.\n\
    - The trained shape_predictor is returned");

    m.def("train_shape_predictor", train_shape_predictor,
        py::arg("dataset_filename"), py::arg("predictor_output_filename"), py::arg("options"),
"requires \n\
    - options.lambda_param > 0 \n\
    - 0 < options.nu <= 1 \n\
    - options.feature_pool_region_padding >= 0 \n\
ensures \n\
    - Uses dlib's shape_predictor_trainer to train a \n\
      shape_predictor based on the labeled images in the XML file \n\
      dataset_filename and the provided options.  This function assumes the file dataset_filename is in the \n\
      XML format produced by dlib's save_image_dataset_metadata() routine. \n\
    - The trained shape predictor is serialized to the file predictor_output_filename.");

    m.def("test_shape_predictor", test_shape_predictor_py,
        py::arg("dataset_filename"), py::arg("predictor_filename"),
"ensures \n\
    - Loads an image dataset from dataset_filename.  We assume dataset_filename is \n\
      a file using the XML format written by save_image_dataset_metadata(). \n\
    - Loads a shape_predictor from the file predictor_filename.  This means \n\
      predictor_filename should be a file produced by the train_shape_predictor() \n\
      routine. \n\
    - This function tests the predictor against the dataset and returns the \n\
      mean average error of the detector.  In fact, The \n\
      return value of this function is identical to that of dlib's \n\
      shape_predictor_trainer() routine.  Therefore, see the documentation \n\
      for shape_predictor_trainer() for a detailed definition of the mean average error.");

    m.def("test_shape_predictor", test_shape_predictor_with_images_no_scales_py,
            py::arg("images"), py::arg("detections"), py::arg("shape_predictor"),
"requires \n\
    - len(images) == len(object_detections) \n\
    - images should be a list of numpy matrices that represent images, either RGB or grayscale. \n\
    - object_detections should be a list of lists of dlib.full_object_detection objects. \
      Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.\n\
 ensures \n\
    - shape_predictor should be a file produced by the train_shape_predictor()  \n\
      routine. \n\
    - This function tests the predictor against the dataset and returns the \n\
      mean average error of the detector.  In fact, The \n\
      return value of this function is identical to that of dlib's \n\
      shape_predictor_trainer() routine.  Therefore, see the documentation \n\
      for shape_predictor_trainer() for a detailed definition of the mean average error.");


    m.def("test_shape_predictor", test_shape_predictor_with_images_py,
            py::arg("images"), py::arg("detections"), py::arg("scales"), py::arg("shape_predictor"),
"requires \n\
    - len(images) == len(object_detections) \n\
    - len(object_detections) == len(scales) \n\
    - for every sublist in object_detections: len(object_detections[i]) == len(scales[i]) \n\
    - scales is a list of floating point scales that each predicted part location \
      should be divided by. Useful for normalization. \n\
    - images should be a list of numpy matrices that represent images, either RGB or grayscale. \n\
    - object_detections should be a list of lists of dlib.full_object_detection objects. \
      Each dlib.full_object_detection contains the bounding box and the lists of points that make up the object parts.\n\
 ensures \n\
    - shape_predictor should be a file produced by the train_shape_predictor()  \n\
      routine. \n\
    - This function tests the predictor against the dataset and returns the \n\
      mean average error of the detector.  In fact, The \n\
      return value of this function is identical to that of dlib's \n\
      shape_predictor_trainer() routine.  Therefore, see the documentation \n\
      for shape_predictor_trainer() for a detailed definition of the mean average error.");
    }
}
