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
// The network first does downsampling, and then upsampling, using
// DenseNet-style blocks. In addition, U-net style skip connections are
// employed.

template <int N, template <typename> class BN, typename SUBNET>
using dense_layer = BN<dlib::relu<dlib::con<N,3,3,1,1,SUBNET>>>;

template <int N, template <typename> class BN, typename SUBNET>
using dense_block = dlib::concat_prev1<dense_layer<N,BN,
                    dlib::concat_prev2<dense_layer<N,BN,
                    dlib::concat_prev3<dense_layer<N,BN,
                    dlib::concat_prev4<dense_layer<N,BN,
                    dlib::tag4<dlib::tag3<dlib::tag2<dlib::tag1<dense_layer<N,BN,SUBNET>>>>>>>>>>>>>;

template <int N, template <typename> class BN, typename SUBNET>
using transition_down = BN<dlib::relu<dlib::con<N,1,1,1,1,dlib::max_pool<3,3,2,2,SUBNET>>>>;

template <int N, template <typename> class BN, typename SUBNET> 
using transition_up = dlib::cont<N,3,3,2,2,SUBNET>;

template <int N, typename SUBNET> using dense     = dense_block<N,dlib::bn_con,SUBNET>;
template <int N, typename SUBNET> using adense    = dense_block<N,dlib::affine,SUBNET>;
template <int N, typename SUBNET> using down	  = transition_down<N,dlib::bn_con,SUBNET>;
template <int N, typename SUBNET> using adown     = transition_down<N,dlib::affine,SUBNET>;
template <int N, typename SUBNET> using up        = dlib::cont<N,3,3,2,2,SUBNET>;

// ----------------------------------------------------------------------------------------

template <typename SUBNET> using dense64 = dense<64,SUBNET>;
template <typename SUBNET> using dense48 = dense<48,SUBNET>;
template <typename SUBNET> using dense32 = dense<32,SUBNET>;
template <typename SUBNET> using dense16  = dense<16,SUBNET>;
template <typename SUBNET> using adense64 = dense<64,SUBNET>;
template <typename SUBNET> using adense48 = dense<48,SUBNET>;
template <typename SUBNET> using adense32 = dense<32,SUBNET>;
template <typename SUBNET> using adense16  = dense<16,SUBNET>;


template <typename SUBNET> using level1 = dlib::repeat<2,dense64,down<64,SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<2,dense48,down<48,SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<2,dense32,down<32,SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<2,dense16,down<16,SUBNET>>;

template <typename SUBNET> using alevel1 = dlib::repeat<2,adense64,adown<64,SUBNET>>;
template <typename SUBNET> using alevel2 = dlib::repeat<2,adense48,adown<48,SUBNET>>;
template <typename SUBNET> using alevel3 = dlib::repeat<2,adense32,adown<32,SUBNET>>;
template <typename SUBNET> using alevel4 = dlib::repeat<2,adense16,adown<16,SUBNET>>;

template <typename SUBNET> using level1t = dlib::repeat<2,dense64,up<64,SUBNET>>;
template <typename SUBNET> using level2t = dlib::repeat<2,dense48,up<48,SUBNET>>;
template <typename SUBNET> using level3t = dlib::repeat<2,dense32,up<32,SUBNET>>;
template <typename SUBNET> using level4t = dlib::repeat<2,dense16,up<16,SUBNET>>;

template <typename SUBNET> using alevel1t = dlib::repeat<2,adense64,up<64,SUBNET>>;
template <typename SUBNET> using alevel2t = dlib::repeat<2,adense48,up<48,SUBNET>>;
template <typename SUBNET> using alevel3t = dlib::repeat<2,adense32,up<32,SUBNET>>;
template <typename SUBNET> using alevel4t = dlib::repeat<2,adense16,up<16,SUBNET>>;

// ----------------------------------------------------------------------------------------

// training network type
using net_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::con<class_count,1,1,1,1,
                            dlib::concat_prev9<dlib::cont<class_count,7,7,2,2,
                            dlib::concat_prev8<level4t<
                            dlib::concat_prev7<level3t<
                            dlib::concat_prev6<level2t<
                            dlib::concat_prev5<level1t<
                            level1<dlib::tag5<
                            level2<dlib::tag6<
                            level3<dlib::tag7<
                            level4<dlib::tag8<
                            dlib::relu<dlib::bn_con<dlib::con<64,7,7,2,2,dlib::tag9<
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::con<class_count,1,1,1,1,
                            dlib::concat_prev9<dlib::cont<class_count,7,7,2,2,
                            dlib::concat_prev8<alevel4t<
                            dlib::concat_prev7<alevel3t<
                            dlib::concat_prev6<alevel2t<
                            dlib::concat_prev5<alevel1t<
                            alevel1<dlib::tag5<
                            alevel2<dlib::tag6<
                            alevel3<dlib::tag7<
                            alevel4<dlib::tag8<
                            dlib::relu<dlib::affine<dlib::con<64,7,7,2,2,dlib::tag9<
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
