// Copyright (C) 2017  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "opaque_types.h"
#include <dlib/python.h>
#include <dlib/matrix.h>
#include <dlib/geometry/vector.h>
#include <dlib/dnn.h>
#include <dlib/image_transforms.h>
#include "indexing.h"
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>


using namespace dlib;
using namespace std;

namespace py = pybind11;


typedef matrix<double,0,1> cv;

class face_recognition_model_v1
{

public:

    face_recognition_model_v1(const std::string& model_filename)
    {
        deserialize(model_filename) >> net;
    }

    matrix<double,0,1> compute_face_descriptor (
        numpy_image<rgb_pixel> img,
        const full_object_detection& face,
        const int num_jitters,
        float padding = 0.25
    )
    {
        std::vector<full_object_detection> faces(1, face);
        return compute_face_descriptors(img, faces, num_jitters, padding)[0];
    }

    matrix<double,0,1> compute_face_descriptor_from_aligned_image (
        numpy_image<rgb_pixel> img,        
        const int num_jitters
    )
    {
        std::vector<numpy_image<rgb_pixel>> images{img};        
        return batch_compute_face_descriptors_from_aligned_images(images, num_jitters)[0];                
    }

    std::vector<matrix<double,0,1>> compute_face_descriptors (
        numpy_image<rgb_pixel> img,
        const std::vector<full_object_detection>& faces,
        const int num_jitters,
        float padding = 0.25
    )
    {
        std::vector<numpy_image<rgb_pixel>> batch_img(1, img);
        std::vector<std::vector<full_object_detection>> batch_faces(1, faces);
        return batch_compute_face_descriptors(batch_img, batch_faces, num_jitters, padding)[0];
    }       

    std::vector<std::vector<matrix<double,0,1>>> batch_compute_face_descriptors (
        const std::vector<numpy_image<rgb_pixel>>& batch_imgs,
        const std::vector<std::vector<full_object_detection>>& batch_faces,
        const int num_jitters,
        float padding = 0.25
    )
    {

        if (batch_imgs.size() != batch_faces.size())
            throw dlib::error("The array of images and the array of array of locations must be of the same size");

        int total_chips = 0;
        for (const auto& faces : batch_faces)
        {
            total_chips += faces.size();
            for (const auto& f : faces)
            {
                if (f.num_parts() != 68 && f.num_parts() != 5)
                    throw dlib::error("The full_object_detection must use the iBUG 300W 68 point face landmark style or dlib's 5 point style.");
            }
        }


        dlib::array<matrix<rgb_pixel>> face_chips;
        for (int i = 0; i < batch_imgs.size(); ++i)
        {
            auto& faces = batch_faces[i];
            auto& img = batch_imgs[i];

            std::vector<chip_details> dets;
            for (const auto& f : faces)
                dets.push_back(get_face_chip_details(f, 150, padding));
            dlib::array<matrix<rgb_pixel>> this_img_face_chips;
            extract_image_chips(img, dets, this_img_face_chips);

            for (auto& chip : this_img_face_chips)
                face_chips.push_back(chip);
        }

        std::vector<std::vector<matrix<double,0,1>>> face_descriptors(batch_imgs.size());
        if (num_jitters <= 1)
        {
            // extract descriptors and convert from float vectors to double vectors
            auto descriptors = net(face_chips, 16);
            auto next = std::begin(descriptors);
            for (int i = 0; i < batch_faces.size(); ++i)
            {
                for (int j = 0; j < batch_faces[i].size(); ++j)
                {
                    face_descriptors[i].push_back(matrix_cast<double>(*next++));
                }
            }
            DLIB_ASSERT(next == std::end(descriptors));
        }
        else
        {
            // extract descriptors and convert from float vectors to double vectors
            auto fimg = std::begin(face_chips);
            for (int i = 0; i < batch_faces.size(); ++i)
            {
                for (int j = 0; j < batch_faces[i].size(); ++j)
                {
                    auto& r = mean(mat(net(jitter_image(*fimg++, num_jitters), 16)));
                    face_descriptors[i].push_back(matrix_cast<double>(r));
                }
            }
            DLIB_ASSERT(fimg == std::end(face_chips));
        }

        return face_descriptors;
    }

    std::vector<matrix<double,0,1>> batch_compute_face_descriptors_from_aligned_images (
        const std::vector<numpy_image<rgb_pixel>>& batch_imgs,        
        const int num_jitters
    )
    {
        dlib::array<matrix<rgb_pixel>> face_chips;           
        for (auto& img : batch_imgs) {

            matrix<rgb_pixel> image;
            if (is_image<unsigned char>(img))
                assign_image(image, numpy_image<unsigned char>(img));
            else if (is_image<rgb_pixel>(img))
                assign_image(image, numpy_image<rgb_pixel>(img));
            else
                throw dlib::error("Unsupported image type, must be 8bit gray or RGB image.");

            // Check for the size of the image
            if ((image.nr() != 150) || (image.nc() != 150)) {
                throw dlib::error("Unsupported image size, it should be of size 150x150. Also cropping must be done as `dlib.get_face_chip` would do it. \
                That is, centered and scaled essentially the same way.");
            }

            face_chips.push_back(image);        
        }       

        std::vector<matrix<double,0,1>> face_descriptors;
        if (num_jitters <= 1)
        {
            // extract descriptors and convert from float vectors to double vectors
            auto descriptors = net(face_chips, 16);      

            for (auto& des: descriptors) {
                face_descriptors.push_back(matrix_cast<double>(des));
            }       
        }
        else
        {
            // extract descriptors and convert from float vectors to double vectors
            for (auto& fimg : face_chips) {
                auto& r = mean(mat(net(jitter_image(fimg, num_jitters), 16)));
                face_descriptors.push_back(matrix_cast<double>(r)); 
            }
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

py::list bottom_up_clustering(py::list descriptors, const int min_num_clusters, const double max_dist)
{
    DLIB_CASSERT(min_num_clusters > 0);

    size_t num_descriptors = py::len(descriptors);
    matrix<float> dist(static_cast<long>(num_descriptors), static_cast<long>(num_descriptors));

    for (size_t i = 0; i < num_descriptors; ++i)
    {
        for (size_t j = i+1; j < num_descriptors; ++j)
        {
            const long i2 = static_cast<long>(i);
            const long j2 = static_cast<long>(j);
            matrix<double,0,1>& first_descriptor = descriptors[i].cast<matrix<double,0,1>&>();
            matrix<double,0,1>& second_descriptor = descriptors[j].cast<matrix<double,0,1>&>();
            dist(i2, j2) = dlib::length( first_descriptor- second_descriptor);
            dist(j2, i2) = dist(i2, j2);
        }
    }

    std::vector<unsigned long> labels;
    const auto num_clusters = dlib::bottom_up_cluster(dist, labels, min_num_clusters, max_dist);

    py::list clusters;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        clusters.append(labels[i]);
    }
    return clusters;
}

py::list chinese_whispers_clustering(py::list descriptors, float threshold)
{
    DLIB_CASSERT(threshold > 0);
    py::list clusters;

    size_t num_descriptors = py::len(descriptors);

    // This next bit of code creates a graph of connected objects and then uses the Chinese
    // whispers graph clustering algorithm to identify how many objects there are and which
    // objects belong to which cluster.
    std::vector<sample_pair> edges;
    std::vector<unsigned long> labels;
    for (size_t i = 0; i < num_descriptors; ++i)
    {
        for (size_t j = i; j < num_descriptors; ++j)
        {
            matrix<double,0,1>& first_descriptor = descriptors[i].cast<matrix<double,0,1>&>();
            matrix<double,0,1>& second_descriptor = descriptors[j].cast<matrix<double,0,1>&>();

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

py::list chinese_whispers_raw(py::list edges)
{
    py::list clusters;
    size_t num_edges = py::len(edges);

    std::vector<sample_pair> edges_pairs;
    std::vector<unsigned long> labels;
    for (size_t idx = 0; idx < num_edges; ++idx)
    {
        py::tuple t = edges[idx].cast<py::tuple>();
        if ((len(t) != 2) && (len(t) != 3))
        {
            PyErr_SetString( PyExc_IndexError, "Input must be a list of tuples with 2 or 3 elements.");
            throw py::error_already_set();
        }
        size_t i = t[0].cast<size_t>();
        size_t j = t[1].cast<size_t>();
        double distance = (len(t) == 3) ? t[2].cast<double>(): 1;

        edges_pairs.push_back(sample_pair(i, j, distance));
    }

    chinese_whispers(edges_pairs, labels);
    for (size_t i = 0; i < labels.size(); ++i)
    {
        clusters.append(labels[i]);
    }
    return clusters;
}

void save_face_chips (
    numpy_image<rgb_pixel> img,
    const std::vector<full_object_detection>& faces,
    const std::string& chip_filename,
    size_t size = 150,
    float padding = 0.25
)
{

    int num_faces = faces.size();
    std::vector<chip_details> dets;
    for (const auto& f : faces)
        dets.push_back(get_face_chip_details(f, size, padding));
    dlib::array<matrix<rgb_pixel>> face_chips;
    extract_image_chips(numpy_image<rgb_pixel>(img), dets, face_chips);
    int i=0;
    for (const auto& chip : face_chips) 
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
    numpy_image<rgb_pixel> img,
    const full_object_detection& face,
    const std::string& chip_filename,
    size_t size = 150,
    float padding = 0.25
)
{
    std::vector<full_object_detection> faces(1, face);
    save_face_chips(img, faces, chip_filename, size, padding);
}

void bind_face_recognition(py::module &m)
{
    {
    typedef std::vector<full_object_detection> type;
    py::bind_vector<type>(m, "full_object_detections", "An array of full_object_detection objects.")
        .def("clear", &type::clear)
        .def("resize", resize<type>)
        .def("extend", extend_vector_with_python_list<full_object_detection>)
        .def(py::pickle(&getstate<type>, &setstate<type>));
    }

    {
    py::class_<face_recognition_model_v1>(m, "face_recognition_model_v1", "This object maps human faces into 128D vectors where pictures of the same person are mapped near to each other and pictures of different people are mapped far apart.  The constructor loads the face recognition model from a file. The model file is available here: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
        .def(py::init<std::string>())
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptor,
            py::arg("img"), py::arg("face"), py::arg("num_jitters")=0, py::arg("padding")=0.25,
            "Takes an image and a full_object_detection that references a face in that image and converts it into a 128D face descriptor. "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. "
            "Optionally allows to override default padding of 0.25 around the face."
            )
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptor_from_aligned_image,
            py::arg("img"), py::arg("num_jitters")=0,
            "Takes an aligned face image of size 150x150 and converts it into a 128D face descriptor."
            "Note that the alignment should be done in the same way dlib.get_face_chip does it."
            "If num_jitters>1 then image will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. "            
            )
        .def("compute_face_descriptor", &face_recognition_model_v1::compute_face_descriptors,
            py::arg("img"), py::arg("faces"), py::arg("num_jitters")=0, py::arg("padding")=0.25,
            "Takes an image and an array of full_object_detections that reference faces in that image and converts them into 128D face descriptors.  "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. "
            "Optionally allows to override default padding of 0.25 around the face."
            )
        .def("compute_face_descriptor", &face_recognition_model_v1::batch_compute_face_descriptors,
            py::arg("batch_img"), py::arg("batch_faces"), py::arg("num_jitters")=0, py::arg("padding")=0.25,
            "Takes an array of images and an array of arrays of full_object_detections. `batch_faces[i]` must be an array of full_object_detections corresponding to the image `batch_img[i]`, "
            "referencing faces in that image. Every face will be converted into 128D face descriptors.  "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. "
            "Optionally allows to override default padding of 0.25 around the face."
            )
        .def("compute_face_descriptor", &face_recognition_model_v1::batch_compute_face_descriptors_from_aligned_images,
            py::arg("batch_img"), py::arg("num_jitters")=0,
            "Takes an array of aligned images of faces of size 150_x_150."
            "Note that the alignment should be done in the same way dlib.get_face_chip does it."
            "Every face will be converted into 128D face descriptors.  "
            "If num_jitters>1 then each face will be randomly jittered slightly num_jitters times, each run through the 128D projection, and the average used as the face descriptor. "            
            );
    }

    m.def("save_face_chip", &save_face_chip, 
	"Takes an image and a full_object_detection that references a face in that image and saves the face with the specified file name prefix.  The face will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.", 
	py::arg("img"), py::arg("face"), py::arg("chip_filename"), py::arg("size")=150, py::arg("padding")=0.25
    );
    m.def("save_face_chips", &save_face_chips, 
	"Takes an image and a full_object_detections object that reference faces in that image and saves the faces with the specified file name prefix.  The faces will be rotated upright and scaled to 150x150 pixels or with the optional specified size and padding.",
          py::arg("img"), py::arg("faces"), py::arg("chip_filename"), py::arg("size")=150, py::arg("padding")=0.25
    );
    m.def("bottom_up_clustering", &bottom_up_clustering, py::arg("descriptors"), py::arg("min_num_clusters")=1, py::arg("max_dist")=0.6,
          "Takes a list of descriptors and returns a list that contains a label for each descriptor. Clustering is done using dlib::bottom_up_cluster.");
    m.def("chinese_whispers_clustering", &chinese_whispers_clustering, py::arg("descriptors"), py::arg("threshold"),
        "Takes a list of descriptors and returns a list that contains a label for each descriptor. Clustering is done using dlib::chinese_whispers."
        );
    m.def("chinese_whispers", &chinese_whispers_raw, py::arg("edges"),
        "Given a graph with vertices represented as numbers indexed from 0, this algorithm takes a list of edges and returns back a list that contains a labels (found clusters) for each vertex. "
        "Edges are tuples with either 2 elements (integers presenting indexes of connected vertices) or 3 elements, where additional one element is float which presents distance weight of the edge). "
        "Offers direct access to dlib::chinese_whispers."
        );
}

