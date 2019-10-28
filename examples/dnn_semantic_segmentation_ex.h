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
    from here: http://dlib.net/files/semantic_segmentation_voc2012net_v2.dnn

    It would be a good idea to become familiar with dlib's DNN tooling before reading this
    example.  So you should read dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp
    before reading this example program.
*/

#ifndef DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
#define DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_

#include <dlib/dnn.h>
#include "pascal_voc_2012.h"

// ----------------------------------------------------------------------------------------

// Introduce the building blocks used to define the segmentation network.
// The network first does residual downsampling (similar to the dnn_imagenet_(train_)ex
// example program), and then residual upsampling. In addition, U-Net style skip
// connections are used, so that not every simple detail needs to reprented on the low
// levels. (See Ronneberger et al. (2015), U-Net: Convolutional Networks for Biomedical
// Image Segmentation, https://arxiv.org/pdf/1505.04597.pdf)

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

template <typename SUBNET> using res64 = res<64,SUBNET>;
template <typename SUBNET> using res128 = res<128,SUBNET>;
template <typename SUBNET> using res256 = res<256,SUBNET>;
template <typename SUBNET> using res512 = res<512,SUBNET>;
template <typename SUBNET> using ares64 = ares<64,SUBNET>;
template <typename SUBNET> using ares128 = ares<128,SUBNET>;
template <typename SUBNET> using ares256 = ares<256,SUBNET>;
template <typename SUBNET> using ares512 = ares<512,SUBNET>;

template <typename SUBNET> using level1 = dlib::repeat<2,res64,res<64,SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<2,res128,res_down<128,SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<2,res256,res_down<256,SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<2,res512,res_down<512,SUBNET>>;

template <typename SUBNET> using alevel1 = dlib::repeat<2,ares64,ares<64,SUBNET>>;
template <typename SUBNET> using alevel2 = dlib::repeat<2,ares128,ares_down<128,SUBNET>>;
template <typename SUBNET> using alevel3 = dlib::repeat<2,ares256,ares_down<256,SUBNET>>;
template <typename SUBNET> using alevel4 = dlib::repeat<2,ares512,ares_down<512,SUBNET>>;

template <typename SUBNET> using level1t = dlib::repeat<2,res64,res_up<64,SUBNET>>;
template <typename SUBNET> using level2t = dlib::repeat<2,res128,res_up<128,SUBNET>>;
template <typename SUBNET> using level3t = dlib::repeat<2,res256,res_up<256,SUBNET>>;
template <typename SUBNET> using level4t = dlib::repeat<2,res512,res_up<512,SUBNET>>;

template <typename SUBNET> using alevel1t = dlib::repeat<2,ares64,ares_up<64,SUBNET>>;
template <typename SUBNET> using alevel2t = dlib::repeat<2,ares128,ares_up<128,SUBNET>>;
template <typename SUBNET> using alevel3t = dlib::repeat<2,ares256,ares_up<256,SUBNET>>;
template <typename SUBNET> using alevel4t = dlib::repeat<2,ares512,ares_up<512,SUBNET>>;

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

static const char* semantic_segmentation_net_filename = "semantic_segmentation_voc2012net_v2.dnn";

// ----------------------------------------------------------------------------------------

// training network type
using bnet_type = dlib::loss_multiclass_log_per_pixel<
                              dlib::cont<class_count,1,1,1,1,
                              dlib::relu<dlib::bn_con<dlib::cont<64,7,7,2,2,
                              concat_utag1<level1t<
                              concat_utag2<level2t<
                              concat_utag3<level3t<
                              concat_utag4<level4t<
                              level4<utag4<
                              level3<utag3<
                              level2<utag2<
                              level1<dlib::max_pool<3,3,2,2,utag1<
                              dlib::relu<dlib::bn_con<dlib::con<64,7,7,2,2,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using anet_type = dlib::loss_multiclass_log_per_pixel<
                              dlib::cont<class_count,1,1,1,1,
                              dlib::relu<dlib::affine<dlib::cont<64,7,7,2,2,
                              concat_utag1<alevel1t<
                              concat_utag2<alevel2t<
                              concat_utag3<alevel3t<
                              concat_utag4<alevel4t<
                              alevel4<utag4<
                              alevel3<utag3<
                              alevel2<utag2<
                              alevel1<dlib::max_pool<3,3,2,2,utag1<
                              dlib::relu<dlib::affine<dlib::con<64,7,7,2,2,
                              dlib::input<dlib::matrix<dlib::rgb_pixel>>
                              >>>>>>>>>>>>>>>>>>>>>>>>>;

// ----------------------------------------------------------------------------------------

#endif // DLIB_DNn_SEMANTIC_SEGMENTATION_EX_H_
