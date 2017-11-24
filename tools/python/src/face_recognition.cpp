// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include <dlib/python.h>
#include <boost/shared_ptr.hpp>
#include <dlib/matrix.h>
#include <boost/python/slice.hpp>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"
#include <dlib/image_io.h>
#include <dlib/clustering.h>


using namespace dlib;
using namespace std;
using namespace boost::python;

typedef matrix<double,0,1> cv;


class face_recognition_model_v1
{

public:

    face_recognition_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    matrix<double,0,1> compute_face_descriptor (
        object img,
        const full_object_detection& face,
        const int num_jitters
    )
    {
        std::vector<full_object_detection> faces(1, face);
        return compute_face_descriptors(img, faces, num_jitters)[0];
    }

    std::vector<matrix<double,0,1>> compute_face_descriptors (
        object img,
        const std::vector<full_object_detection>& faces,
        const int num_jitters
    )
    {
        if (!is_rgb_python_image(img))
            throw dlib::error("Unsupported image type, must be RGB image.");

        for (auto& f : faces)
        {
            if (f.num_parts() != 68 && f.num_parts() != 5)
                throw dlib::error("The full_object_detection must use the iBUG 300W 68 point face landmark style or dlib's 5 point style.");
        }


        std::vector<chip_details> dets;
        for (auto& f : faces)
            dets.push_back(get_face_chip_details(f, 150, 0.25));
        dlib::array<matrix<rgb_pixel>> face_chips;
        extract_image_chips(numpy_rgb_image(img), dets, face_chips);

        std::vector<matrix<double,0,1>> face_descriptors;
        face_descriptors.reserve(face_chips.size());

        if (num_jitters <= 1)
        {
            // extract descriptors and convert from float vectors to double vectors
            for (auto& d : net(face_chips,16))
                face_descriptors.push_back(matrix_cast<double>(d));
        }
        else
        {
            for (auto& fimg : face_chips)
                face_descriptors.push_back(matrix_cast<double>(mean(mat(net(jitter_image(fimg,num_jitters),16)))));
        }

        return face_descriptors;
    }

private:

    dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> jitter_image(
        const matrix<rgb_pixel>& img,
        const int num_jitters
    )
    {
        std::vector<matrix<rgb_pixel>> crops; 
        for (int i = 0; i < num_jitters; ++i)
            crops.push_back(dlib::jitter_image(img,rnd));
        return crops;
    }


    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

    template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
    using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

    template <int N, template <typename> class BN, int stride, typename SUBNET> 
    using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

    template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
    template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

    template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
    template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
    template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
    template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
    template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

    using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                                alevel0<
                                alevel1<
                                alevel2<
                                alevel3<
                                alevel4<
                                max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                input_rgb_image_sized<150>
                                >>>>>>>>>>>>;
    anet_type net;
};

// ----------------------------------------------------------------------------------------

boost::python::list chinese_whispers_clustering(boost::python::list descriptors, float threshold)
{
    boost::python::list clusters;

    size_t num_descriptors = len(descriptors);

    // This next bit of code creates a graph of connected objects and then uses the Chinese
    // whispers graph clustering algorithm to identify how many objects there are and which
    // objects belong to which cluster.
    std::vector<sample_pair> edges;
    std::vector<unsigned long> labels;
    for (size_t i = 0; i < num_descriptors; ++i)
    {
        for (size_t j = i; j < num_descriptors; ++j)
        {
            matrix<double,0,1>& first_descriptor = boost::python::extract<matrix<double,0,1>&>(descriptors[i]);
            matrix<double,0,1>& second_descriptor = boost::python::extract<matrix<double,0,1>&>(descriptors[j]);

            if (length(first_descriptor-second_descriptor) < threshold)
                edges.push_back(sample_pair(i,j));
        }
    }
    chinese_whispers(edges, labels);
    for (size_t i = 0; i < labels.size(); ++i)
    {
        clusters.append(labels[i]);
    }
    return clusters;
}

void save_face_chips (
    object img,
    const std::vector<full_object_detection>& faces,
    const std::string& chip_filename,
    size_t size = 150,
    float padding = 0.25
)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    int num_faces = faces.size();
    std::vector<chip_details> dets;
    for (auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_rgb_image(img), dets, face_chips);
    int i=0;
    for (auto& chip : face_chips) 
    {
        i++;
        if(num_faces > 1) 
        {
            const std::string& file_name = chip_filename + "_" + std::to_string(i) + ".jpg";
            save_jpeg(chip, file_name);
        }
        else
        {
            const std::string& file_name = chip_filename + ".jpg";
            save_jpeg(chip, file_name);
        }
    }
}

void save_face_chip (
    object img,
    const full_object_detection& face,
    const std::string& chip_filename,
    size_t size = 150,
    float padding = 0.25
)
{
    std::vector<full_object_detection> faces(1, face);
    save_face_chips(img, faces, chip_filename, size, padding);
    return;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(save_face_chip_with_defaults, save_face_chip, 3, 5)
BOOST_PYTHON_FUNCTION_OVERLOADS(save_face_chips_with_defaults, save_face_chips, 3, 5)

// ----------------------------------------------------------------------------------------

boost::python::list get_face_chips (
    object img,
    const std::vector<full_object_detection>& faces,
    size_t size = 150,
    float padding = 0.25
)
{
    if (!is_rgb_python_image(img))
        throw dlib::error("Unsupported image type, must be RGB image.");

    if (faces.size() < 1) {
        throw dlib::error("No face were specified in the faces array.");
    }

    boost::python::list chips_list;

    std::vector<chip_details> dets;
    for (auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_rgb_image(img), dets, face_chips);

    for (auto& chip : face_chips) 
    {
        boost::python::list img;
        
        for(size_t row=0; row<size; row++) {
            boost::python::list row_list;
            for(size_t col=0; col<size; col++) {
                rgb_pixel pixel = chip(row, col);
                boost::python::list item;
                
                item.append(pixel.red);
                item.append(pixel.green);
                item.append(pixel.blue);
                row_list.append(item);
            }
            img.append(row_list);
        }

        chips_list.append(img);
    }
    return chips_list;
}

boost::python::list get_face_chip (
    object img,
    const full_object_detection& face,
    size_t size = 150,
    float padding = 0.25
)
{
    std::vector<full_object_detection> faces(1, face);
    boost::python::list result = get_face_chips(img, faces, size, padding);
    size_t num_images = boost::python::len(result);
    if(num_images == 1) {
        return boost::python::extract<boost::python::list>(result[0]); 
    } else {
        throw dlib::error("No face chips found!");
    }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(get_face_chip_with_defaults, get_face_chip, 2, 4)
BOOST_PYTHON_FUNCTION_OVERLOADS(get_face_chips_with_defaults, get_face_chips, 2, 4)


// ----------------------------------------------------------------------------------------

void bind_face_recognition()
{
    using boost::python::arg;
    {
    class_<face_recognition_model_v1>("face_recognition_model_v1", "This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.  The constructor loads the face recognition model from a file. The model file is available here: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2", init<std::string>())
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptor, (arg("img"),arg("face"),arg("num_jitters")=0),
            "Takes an image and a full_object_detection that references a face in that image and converts it into a 128D face descriptor. "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor."
            )
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptors, (arg("img"),arg("faces"),arg("num_jitters")=0),
            "Takes an image and an array of full_object_detections that reference faces in that image and converts them into 128D face descriptors.  "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor."
            );
    }

    def("save_face_chip", &save_face_chip, save_face_chip_with_defaults(
	"Takes an image and a full_object_detection that references a face in that image and saves the face with the specified file name prefix.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.", 
	(arg("img"), arg("face"), arg("chip_filename"), arg("size"), arg("padding"))
    ));
    def("save_face_chips", &save_face_chips, save_face_chips_with_defaults(
	"Takes an image and a full_object_detections object that reference faces in that image and saves the faces with the specified file name prefix.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.",
	(arg("img"), arg("faces"), arg("chip_filename"), arg("size"), arg("padding"))
    ));
    def("get_face_chip", &get_face_chip, get_face_chip_with_defaults(
	"Takes an image and a full_object_detection that references a face in that image and returns the face as a list of lists representing the image.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.", 
	(arg("img"), arg("face"), arg("size"), arg("padding"))
    ));
    def("get_face_chips", &get_face_chips, get_face_chips_with_defaults(
	"Takes an image and a full_object_detections object that reference faces in that image and returns the faces as a list of list of lists representing the image.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.",
	(arg("img"), arg("faces"), arg("size"), arg("padding"))
    ));
    def("chinese_whispers_clustering", &chinese_whispers_clustering, (arg("descriptors"), arg("threshold")),
        "Takes a list of descriptors and returns a list that contains a label for each descriptor. Clustering is done using dlib::chinese_whispers."
        );

    {   
    typedef std::vector<full_object_detection> type;
    class_<type>("full_object_detections", "An array of full_object_detection objects.")
        .def(vector_indexing_suite<type>())
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def_pickle(serialize_pickle<type>());
    }
}

