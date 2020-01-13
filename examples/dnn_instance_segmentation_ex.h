// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    Instance segmentation using the PASCAL VOC2012 dataset.

    Instance segmentation sort-of combines object detection with semantic
    segmentation. While each dog, for example, is detected separately,
    the output is not only a bounding-box but a more accurate, per-pixel
    mask.

    For introductions to object detection and semantic segmentation, you
    can have a look at dnn_mmod_ex.cpp and dnn_semantic_segmentation.h,
    respectively.

    Instructions how to run the example:
    1. Download the PASCAL VOC2012 data, and untar it somewhere.
       http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    2. Build the dnn_instance_segmentation_train_ex example program.
    3. Run:
       ./dnn_instance_segmentation_train_ex /path/to/VOC2012
    4. Wait while the network is being trained.
    5. Build the dnn_instance_segmentation_ex example program.
    6. Run:
       ./dnn_instance_segmentation_ex /path/to/VOC2012-or-other-images

    An alternative to steps 2-4 above is to download a pre-trained network
    from here: http://dlib.net/files/instance_segmentation_voc2012net_v2.dnn

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#ifndef DLIB_DNn_INSTANCE_SEGMENTATION_EX_H_
#define DLIB_DNn_INSTANCE_SEGMENTATION_EX_H_

#include <dlib/dnn.h>

// ----------------------------------------------------------------------------------------

namespace {
    // Segmentation will be performed using patches having this size.
    constexpr int seg_dim = 227;
}

dlib::rectangle get_cropping_rect(const dlib::rectangle& rectangle)
{
    DLIB_ASSERT(!rectangle.is_empty());

    const auto center_point = dlib::center(rectangle);
    const auto max_dim = std::max(rectangle.width(), rectangle.height());
    const auto d = static_cast<long>(std::round(max_dim / 2.0 * 1.5)); // add +50%

    return dlib::rectangle(
        center_point.x() - d,
        center_point.y() - d,
        center_point.x() + d,
        center_point.y() + d
    );
}

// ----------------------------------------------------------------------------------------

// The object detection network.
// Adapted from dnn_mmod_train_find_cars_ex.cpp and friends.

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using bdownsampler = dlib::relu<dlib::bn_con<con5d<128,dlib::relu<dlib::bn_con<con5d<128,dlib::relu<dlib::bn_con<con5d<32,SUBNET>>>>>>>>>;
template <typename SUBNET> using adownsampler = dlib::relu<dlib::affine<con5d<128,dlib::relu<dlib::affine<con5d<128,dlib::relu<dlib::affine<con5d<32,SUBNET>>>>>>>>>;

template <typename SUBNET> using brcon5 = dlib::relu<dlib::bn_con<con5<256,SUBNET>>>;
template <typename SUBNET> using arcon5 = dlib::relu<dlib::affine<con5<256,SUBNET>>>;

using det_bnet_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,brcon5<brcon5<brcon5<bdownsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;
using det_anet_type = dlib::loss_mmod<dlib::con<1,9,9,1,1,arcon5<arcon5<arcon5<adownsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

// The segmentation network.
// For the time being, this is very much copy-paste from dnn_semantic_segmentation.h, although the network is made narrower (smaller feature maps).

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

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

template <typename SUBNET> using res16 = res<16,SUBNET>;
template <typename SUBNET> using res24 = res<24,SUBNET>;
template <typename SUBNET> using res32 = res<32,SUBNET>;
template <typename SUBNET> using res48 = res<48,SUBNET>;
template <typename SUBNET> using ares16 = ares<16,SUBNET>;
template <typename SUBNET> using ares24 = ares<24,SUBNET>;
template <typename SUBNET> using ares32 = ares<32,SUBNET>;
template <typename SUBNET> using ares48 = ares<48,SUBNET>;

template <typename SUBNET> using level1 = dlib::repeat<2,res16,res<16,SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<2,res24,res_down<24,SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<2,res32,res_down<32,SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<2,res48,res_down<48,SUBNET>>;

template <typename SUBNET> using alevel1 = dlib::repeat<2,ares16,ares<16,SUBNET>>;
template <typename SUBNET> using alevel2 = dlib::repeat<2,ares24,ares_down<24,SUBNET>>;
template <typename SUBNET> using alevel3 = dlib::repeat<2,ares32,ares_down<32,SUBNET>>;
template <typename SUBNET> using alevel4 = dlib::repeat<2,ares48,ares_down<48,SUBNET>>;

template <typename SUBNET> using level1t = dlib::repeat<2,res16,res_up<16,SUBNET>>;
template <typename SUBNET> using level2t = dlib::repeat<2,res24,res_up<24,SUBNET>>;
template <typename SUBNET> using level3t = dlib::repeat<2,res32,res_up<32,SUBNET>>;
template <typename SUBNET> using level4t = dlib::repeat<2,res48,res_up<48,SUBNET>>;

template <typename SUBNET> using alevel1t = dlib::repeat<2,ares16,ares_up<16,SUBNET>>;
template <typename SUBNET> using alevel2t = dlib::repeat<2,ares24,ares_up<24,SUBNET>>;
template <typename SUBNET> using alevel3t = dlib::repeat<2,ares32,ares_up<32,SUBNET>>;
template <typename SUBNET> using alevel4t = dlib::repeat<2,ares48,ares_up<48,SUBNET>>;

// ----------------------------------------------------------------------------------------

template <
    template<typename> class TAGGED,
    template<typename> class PREV_RESIZED,
    typename SUBNET
>
using resize_and_concat = dlib::add_layer<
                          dlib::concat_<TAGGED,PREV_RESIZED>,
                          PREV_RESIZED<dlib::resize_prev_to_tagged<TAGGED,SUBNET>>>;

template <typename SUBNET> using utag1 = dlib::add_tag_layer<2100+1,SUBNET>;
template <typename SUBNET> using utag2 = dlib::add_tag_layer<2100+2,SUBNET>;
template <typename SUBNET> using utag3 = dlib::add_tag_layer<2100+3,SUBNET>;
template <typename SUBNET> using utag4 = dlib::add_tag_layer<2100+4,SUBNET>;

template <typename SUBNET> using utag1_ = dlib::add_tag_layer<2110+1,SUBNET>;
template <typename SUBNET> using utag2_ = dlib::add_tag_layer<2110+2,SUBNET>;
template <typename SUBNET> using utag3_ = dlib::add_tag_layer<2110+3,SUBNET>;
template <typename SUBNET> using utag4_ = dlib::add_tag_layer<2110+4,SUBNET>;

template <typename SUBNET> using concat_utag1 = resize_and_concat<utag1,utag1_,SUBNET>;
template <typename SUBNET> using concat_utag2 = resize_and_concat<utag2,utag2_,SUBNET>;
template <typename SUBNET> using concat_utag3 = resize_and_concat<utag3,utag3_,SUBNET>;
template <typename SUBNET> using concat_utag4 = resize_and_concat<utag4,utag4_,SUBNET>;

// ----------------------------------------------------------------------------------------

static const char* instance_segmentation_net_filename = "instance_segmentation_voc2012net_v2.dnn";

// ----------------------------------------------------------------------------------------

// training network type
using seg_bnet_type = dlib::loss_binary_log_per_pixel<
                              dlib::cont<1,1,1,1,1,
                              dlib::relu<dlib::bn_con<dlib::cont<16,7,7,2,2,
                              concat_utag1<level1t<
                              concat_utag2<level2t<
                              concat_utag3<level3t<
                              concat_utag4<level4t<
                              level4<utag4<
                              level3<utag3<
                              level2<utag2<
                              level1<dlib::max_pool<3,3,2,2,utag1<
                              dlib::relu<dlib::bn_con<dlib::con<16,7,7,2,2,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using seg_anet_type = dlib::loss_binary_log_per_pixel<
                              dlib::cont<1,1,1,1,1,
                              dlib::relu<dlib::affine<dlib::cont<16,7,7,2,2,
                              concat_utag1<alevel1t<
                              concat_utag2<alevel2t<
                              concat_utag3<alevel3t<
                              concat_utag4<alevel4t<
                              alevel4<utag4<
                              alevel3<utag3<
                              alevel2<utag2<
                              alevel1<dlib::max_pool<3,3,2,2,utag1<
                              dlib::relu<dlib::affine<dlib::con<16,7,7,2,2,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // DLIB_DNn_INSTANCE_SEGMENTATION_EX_H_
