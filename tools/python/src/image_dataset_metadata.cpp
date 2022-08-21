// Copyright (C) 2018  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <iostream>

namespace pybind11
{

    // a version of bind_map that doesn't force it's own __repr__ on you.
template <typename Map, typename holder_type = std::unique_ptr<Map>, typename... Args>
class_<Map, holder_type> bind_map_no_default_repr(handle scope, const std::string &name, Args&&... args) {
    using KeyType = typename Map::key_type;
    using MappedType = typename Map::mapped_type;
    using Class_ = class_<Map, holder_type>;

    // If either type is a non-module-local bound type then make the map binding non-local as well;
    // otherwise (e.g. both types are either module-local or converting) the map will be
    // module-local.
    auto tinfo = detail::get_type_info(typeid(MappedType));
    bool local = !tinfo || tinfo->module_local;
    if (local) {
        tinfo = detail::get_type_info(typeid(KeyType));
        local = !tinfo || tinfo->module_local;
    }

    Class_ cl(scope, name.c_str(), pybind11::module_local(local), std::forward<Args>(args)...);

    cl.def(init<>());


    cl.def("__bool__",
        [](const Map &m) -> bool { return !m.empty(); },
        "Check whether the map is nonempty"
    );

    cl.def("__iter__",
           [](Map &m) { return make_key_iterator(m.begin(), m.end()); },
           keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    cl.def("items",
           [](Map &m) { return make_iterator(m.begin(), m.end()); },
           keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    cl.def("__getitem__",
        [](Map &m, const KeyType &k) -> MappedType & {
            auto it = m.find(k);
            if (it == m.end())
              throw key_error();
           return it->second;
        },
        return_value_policy::reference_internal // ref + keepalive
    );

    // Assignment provided only if the type is copyable
    detail::map_assignment<Map, Class_>(cl);

    cl.def("__delitem__",
           [](Map &m, const KeyType &k) {
               auto it = m.find(k);
               if (it == m.end())
                   throw key_error();
               return m.erase(it);
           }
    );

    cl.def("__len__", &Map::size);

    return cl;
}

}

using namespace dlib;
using namespace std;
using namespace dlib::image_dataset_metadata;

namespace py = pybind11;


dataset py_load_image_dataset_metadata(
    const std::string& filename
)
{
    dataset temp;
    load_image_dataset_metadata(temp, filename);
    return temp;
}

std::shared_ptr<std::map<std::string,point>> map_from_object(py::dict obj)
{
    auto ret = std::make_shared<std::map<std::string,point>>();
    for (const auto& v : obj)
    {
        (*ret)[v.first.cast<std::string>()] = v.second.cast<point>();
    }
    return ret;
}

// ----------------------------------------------------------------------------------------

image_dataset_metadata::dataset py_make_bounding_box_regression_training_data (
    const image_dataset_metadata::dataset& truth,
    const py::object& detections
)
{
    try
    {
        // if detections is a std::vector then call like this.
        return make_bounding_box_regression_training_data(truth, detections.cast<const std::vector<std::vector<rectangle>>&>());
    }
    catch (py::cast_error&)
    {
        // otherwise, detections should be a list of std::vectors.
        py::list dets(detections);
        std::vector<std::vector<rectangle>> temp;
        for (const auto& d : dets)
            temp.emplace_back(d.cast<const std::vector<rectangle>&>());
        return make_bounding_box_regression_training_data(truth, temp);
    }
}

// ----------------------------------------------------------------------------------------

void bind_image_dataset_metadata(py::module &m_)
{
    auto m = m_.def_submodule("image_dataset_metadata", "Routines and objects for working with dlib's image dataset metadata XML files.");

    auto datasetstr  = [](const dataset& item) { return  "dlib.dataset_dataset_metadata.dataset: images:" + to_string(item.images.size()) + ", " + item.name; };
    auto datasetrepr = [datasetstr](const dataset& item) { return "<"+datasetstr(item)+">"; };
    py::class_<dataset>(m, "dataset",
                    "This object represents a labeled set of images.  In particular, it contains the filename for each image as well as annotated boxes.")
        .def(py::init())
        .def("__str__", datasetstr)
        .def("__repr__", datasetrepr)
        .def_readwrite("images", &dataset::images)
        .def_readwrite("comment", &dataset::comment)
        .def_readwrite("name", &dataset::name);

    auto imagestr  = [](const image& item) { return  "dlib.image_dataset_metadata.image: boxes:"+to_string(item.boxes.size())+ ", " + item.filename; };
    auto imagerepr = [imagestr](const image& item) { return "<"+imagestr(item)+">"; };
    py::class_<image>(m, "image", "This object represents an annotated image.")
        .def(py::init())
        .def_readwrite("filename", &image::filename)
        .def("__str__", imagestr)
        .def("__repr__", imagerepr)
        .def_readwrite("boxes", &image::boxes);

    auto imagesstr = [imagerepr](const std::vector<image>& images) 
    { 
        std::ostringstream sout; 
        for (size_t i = 0; i < images.size(); ++i)
        {
            if (i == 0)
                sout << "[" << imagerepr(images[i]) << ",\n";
            else if (i+1 == images.size())
                sout << " " << imagerepr(images[i]) << "]";
            else
                sout << " " << imagerepr(images[i]) << ",\n";
        }
        return sout.str();
    };
    py::bind_vector<std::vector<image>>(m, "images", "An array of dlib::image_dataset_metadata::image objects.")
        .def("__str__", imagesstr)
        .def("__repr__", imagesstr);

    auto partsstr = [](const std::map<std::string,point>& item) {
        std::ostringstream sout;
        sout << "{";
        for (const auto& v : item) 
            sout << "'" << v.first << "': " << v.second << ", ";
        sout << "}";
        return sout.str();
    };
    auto partsrepr = [](const std::map<std::string,point>& item) {
        std::ostringstream sout;
        sout << "dlib.image_dataset_metadata.parts({\n";
        for (const auto& v : item) 
            sout << "'" << v.first << "': dlib.point" << v.second << ",\n";
        sout << "})";
        return sout.str();
    };

    py::bind_map_no_default_repr<std::map<std::string,point>, std::shared_ptr<std::map<std::string,point>> >(m, "parts", 
        "This object is a dictionary mapping string names to object part locations.")
        .def(py::init(&map_from_object))
        .def("__str__", partsstr)
        .def("__repr__", partsrepr);


    auto rectstr = [](const rectangle& r) {
        std::ostringstream sout;
        sout << "dlib.rectangle(" << r.left() << "," << r.top() << "," << r.right() << "," << r.bottom() << ")";
        return sout.str();
    };
    auto boxstr  = [rectstr](const box& item) { return "dlib.image_dataset_metadata.box at " + rectstr(item.rect); }; 
    auto boxrepr = [boxstr](const box& item) { return "<"+boxstr(item)+">"; };
    py::class_<box> pybox(m, "box", 
        "This object represents an annotated rectangular area of an image. \n"
        "It is typically used to mark the location of an object such as a \n"
        "person, car, etc.\n"
        "\n"
        "The main variable of interest is rect.  It gives the location of \n"
        "the box.  All the other variables are optional." ); pybox
        .def(py::init())
        .def("__str__", boxstr)
        .def("__repr__", boxrepr)
        .def_readwrite("rect",            &box::rect)
        .def_readonly("parts",           &box::parts)
        .def_readwrite("label",           &box::label)
        .def_readwrite("difficult",       &box::difficult)
        .def_readwrite("truncated",       &box::truncated)
        .def_readwrite("occluded",        &box::occluded)
        .def_readwrite("ignore",          &box::ignore)
        .def_readwrite("pose",            &box::pose)
        .def_readwrite("detection_score", &box::detection_score)
        .def_readwrite("angle",           &box::angle)
        .def_readwrite("gender",          &box::gender)
        .def_readwrite("age",             &box::age);

    auto boxesstr = [boxrepr](const std::vector<box>& boxes) 
    { 
        std::ostringstream sout; 
        for (size_t i = 0; i < boxes.size(); ++i)
        {
            if (i == 0)
                sout << "[" << boxrepr(boxes[i]) << ",\n";
            else if (i+1 == boxes.size())
                sout << " " << boxrepr(boxes[i]) << "]";
            else
                sout << " " << boxrepr(boxes[i]) << ",\n";
        }
        return sout.str();
    };
    py::bind_vector<std::vector<box>>(m, "boxes", "An array of dlib::image_dataset_metadata::box objects.")
        .def("__str__", boxesstr)
        .def("__repr__", boxesstr);

    py::enum_<gender_t>(pybox,"gender_type")
        .value("MALE", gender_t::MALE)
        .value("FEMALE", gender_t::FEMALE)
        .value("UNKNOWN", gender_t::UNKNOWN)
        .export_values();


    m.def("save_image_dataset_metadata", &save_image_dataset_metadata, py::arg("data"), py::arg("filename"),
        "Writes the contents of the meta object to a file with the given filename.  The file will be in an XML format."
        );

    m.def("load_image_dataset_metadata", &py_load_image_dataset_metadata, py::arg("filename"),
        "Attempts to interpret filename as a file containing XML formatted data as produced "
        "by the save_image_dataset_metadata() function.  The data is loaded and returned as a dlib.image_dataset_metadata.dataset object."
        );

    m_.def("make_bounding_box_regression_training_data", &py_make_bounding_box_regression_training_data, 
        py::arg("truth"), py::arg("detections"),
"requires \n\
    - len(truth.images) == len(detections) \n\
    - detections == A dlib.rectangless object or a list of dlib.rectangles. \n\
ensures \n\
    - Suppose you have an object detector that can roughly locate objects in an \n\
      image.  This means your detector draws boxes around objects, but these are \n\
      *rough* boxes in the sense that they aren't positioned super accurately.  For \n\
      instance, HOG based detectors usually have a stride of 8 pixels.  So the \n\
      positional accuracy is going to be, at best, +/-8 pixels.   \n\
       \n\
      If you want to get better positional accuracy one easy thing to do is train a \n\
      shape_predictor to give you the corners of the object.  The \n\
      make_bounding_box_regression_training_data() routine helps you do this by \n\
      creating an appropriate training dataset.  It does this by taking the dataset \n\
      you used to train your detector (the truth object), and combining that with \n\
      the output of your detector on each image in the training dataset (the \n\
      detections object).  In particular, it will create a new annotated dataset \n\
      where each object box is one of the rectangles from detections and that \n\
      object has 4 part annotations, the corners of the truth rectangle \n\
      corresponding to that detection rectangle.  You can then take the returned \n\
      dataset and train a shape_predictor on it.  The resulting shape_predictor can \n\
      then be used to do bounding box regression. \n\
    - We assume that detections[i] contains object detections corresponding to  \n\
      the image truth.images[i]." 
    /*!
        requires
            - len(truth.images) == len(detections)
            - detections == A dlib.rectangless object or a list of dlib.rectangles.
        ensures
            - Suppose you have an object detector that can roughly locate objects in an
              image.  This means your detector draws boxes around objects, but these are
              *rough* boxes in the sense that they aren't positioned super accurately.  For
              instance, HOG based detectors usually have a stride of 8 pixels.  So the
              positional accuracy is going to be, at best, +/-8 pixels.  
              
              If you want to get better positional accuracy one easy thing to do is train a
              shape_predictor to give you the corners of the object.  The
              make_bounding_box_regression_training_data() routine helps you do this by
              creating an appropriate training dataset.  It does this by taking the dataset
              you used to train your detector (the truth object), and combining that with
              the output of your detector on each image in the training dataset (the
              detections object).  In particular, it will create a new annotated dataset
              where each object box is one of the rectangles from detections and that
              object has 4 part annotations, the corners of the truth rectangle
              corresponding to that detection rectangle.  You can then take the returned
              dataset and train a shape_predictor on it.  The resulting shape_predictor can
              then be used to do bounding box regression.
            - We assume that detections[i] contains object detections corresponding to 
              the image truth.images[i].
    !*/
    );
}


