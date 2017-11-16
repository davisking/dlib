// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Semantic segmentation using the PASCAL VOC2012 dataset.

    In segmentation, the task is to assign each pixel of an input image
    a label - for example, 'dog'.  Then, the idea is that neighboring
    pixels having the same label can be connected together to form a
    larger region, representing a complete (or partially occluded) dog.
    So technically, segmentation can be viewed as classification of
    individual pixels (using the relevant context in the input images),
    however the goal usually is to identify meaningful regions that
    represent complete entities of interest (such as dogs).

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_semantic_segmentation_train_ex example program.
    3. Run:
       ./dnn_semantic_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_semantic_segmentation_ex example program.
    6. Run:
       ./dnn_semantic_segmentation_ex /path/to/VOC2012-or-other-images

    An alternative to steps 2-4 above is to download a pre-trained network
    from here: http://dlib.net/files/semantic_segmentation_voc2012net.dnn

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#ifndef DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
#define DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_

#include <dlib/dnn.h>

// ----------------------------------------------------------------------------------------

inline bool operator == (const dlib::rgb_pixel& a, const dlib::rgb_pixel& b)
{
    return a.red == b.red && a.green == b.green && a.blue == b.blue;
}

// ----------------------------------------------------------------------------------------

// The PASCAL VOC2012 dataset contains 20 ground-truth classes + background.  Each class
// is represented using an RGB color value.  We associate each class also to an index in the
// range [0, 20], used internally by the network.

struct Voc2012class {
    Voc2012class(uint16_t index, const dlib::rgb_pixel& rgb_label, const std::string& classlabel)
        : index(index), rgb_label(rgb_label), classlabel(classlabel)
    {}

    // The index of the class. In the PASCAL VOC 2012 dataset, indexes from 0 to 20 are valid.
    const uint16_t index = 0;

    // The corresponding RGB representation of the class.
    const dlib::rgb_pixel rgb_label;

    // The label of the class in plain text.
    const std::string classlabel;
};

namespace {
    constexpr int class_count = 21; // background + 20 classes

    const std::vector<Voc2012class> classes = {
        Voc2012class(0, dlib::rgb_pixel(0, 0, 0), ""), // background

        // The cream-colored `void' label is used in border regions and to mask difficult objects
        // (see http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html)
        Voc2012class(dlib::loss_multiclass_log_per_pixel_::label_to_ignore,
            dlib::rgb_pixel(224, 224, 192), "border"),

        Voc2012class(1,  dlib::rgb_pixel(128,   0,   0), "aeroplane"),
        Voc2012class(2,  dlib::rgb_pixel(  0, 128,   0), "bicycle"),
        Voc2012class(3,  dlib::rgb_pixel(128, 128,   0), "bird"),
        Voc2012class(4,  dlib::rgb_pixel(  0,   0, 128), "boat"),
        Voc2012class(5,  dlib::rgb_pixel(128,   0, 128), "bottle"),
        Voc2012class(6,  dlib::rgb_pixel(  0, 128, 128), "bus"),
        Voc2012class(7,  dlib::rgb_pixel(128, 128, 128), "car"),
        Voc2012class(8,  dlib::rgb_pixel( 64,   0,   0), "cat"),
        Voc2012class(9,  dlib::rgb_pixel(192,   0,   0), "chair"),
        Voc2012class(10, dlib::rgb_pixel( 64, 128,   0), "cow"),
        Voc2012class(11, dlib::rgb_pixel(192, 128,   0), "diningtable"),
        Voc2012class(12, dlib::rgb_pixel( 64,   0, 128), "dog"),
        Voc2012class(13, dlib::rgb_pixel(192,   0, 128), "horse"),
        Voc2012class(14, dlib::rgb_pixel( 64, 128, 128), "motorbike"),
        Voc2012class(15, dlib::rgb_pixel(192, 128, 128), "person"),
        Voc2012class(16, dlib::rgb_pixel(  0,  64,   0), "pottedplant"),
        Voc2012class(17, dlib::rgb_pixel(128,  64,   0), "sheep"),
        Voc2012class(18, dlib::rgb_pixel(  0, 192,   0), "sofa"),
        Voc2012class(19, dlib::rgb_pixel(128, 192,   0), "train"),
        Voc2012class(20, dlib::rgb_pixel(  0,  64, 128), "tvmonitor"),
    };
}

template <typename Predicate>
const Voc2012class& find_voc2012_class(Predicate predicate)
{
    const auto i = std::find_if(classes.begin(), classes.end(), predicate);

    if (i != classes.end())
    {
        return *i;
    }
    else
    {
        throw std::runtime_error("Unable to find a matching VOC2012 class");
    }
}

// ----------------------------------------------------------------------------------------

// Introduce the building blocks used to define the segmentation network.
// The network first does residual downsampling (similar to the dnn_imagenet_(train_)ex 
// example program), and then residual upsampling. The network could be improved e.g.
// by introducing skip connections from the input image, and/or the first layers, to the
// last layer(s).  (See Long et al., Fully Convolutional Networks for Semantic Segmentation,
// https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block = BN<dlib::con<N,3,3,1,1, dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using blockt = BN<dlib::cont<N,3,3,1,1,dlib::relu<BN<dlib::cont<N,3,3,stride,stride,SUBNET>>>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_up = dlib::add_prev2<dlib::cont<N,2,2,2,2,dlib::skip1<dlib::tag2<blockt<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, typename SUBNET> using res       = dlib::relu<residual<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using res_down  = dlib::relu<residual_down<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using res_up    = dlib::relu<residual_up<block,N,dlib::bn_con,SUBNET>>;
template <int N, typename SUBNET> using ares_up   = dlib::relu<residual_up<block,N,dlib::affine,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using level1 = res<512,res<512,res_down<512,SUBNET>>>;
template <typename SUBNET> using level2 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level3 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level4 = res<64,res<64,res<64,SUBNET>>>;

template <typename SUBNET> using alevel1 = ares<512,ares<512,ares_down<512,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel4 = ares<64,ares<64,ares<64,SUBNET>>>;

template <typename SUBNET> using level1t = res<512,res<512,res_up<512,SUBNET>>>;
template <typename SUBNET> using level2t = res<256,res<256,res_up<256,SUBNET>>>;
template <typename SUBNET> using level3t = res<128,res<128,res_up<128,SUBNET>>>;
template <typename SUBNET> using level4t = res<64,res<64,res_up<64,SUBNET>>>;

template <typename SUBNET> using alevel1t = ares<512,ares<512,ares_up<512,SUBNET>>>;
template <typename SUBNET> using alevel2t = ares<256,ares<256,ares_up<256,SUBNET>>>;
template <typename SUBNET> using alevel3t = ares<128,ares<128,ares_up<128,SUBNET>>>;
template <typename SUBNET> using alevel4t = ares<64,ares<64,ares_up<64,SUBNET>>>;

// ----------------------------------------------------------------------------------------

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::cont<class_count,7,7,2,2,
                            level4t<level3t<level2t<level1t<
                            level1<level2<level3<level4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::bn_con<dlib::con<64,7,7,2,2,
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::cont<class_count,7,7,2,2,
                            alevel4t<alevel3t<alevel2t<alevel1t<
                            alevel1<alevel2<alevel3<alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<64,7,7,2,2,
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
