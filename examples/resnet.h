#pragma once

#include <dlib/dnn.h>

// BATCHNORM must be bn_con or affine layer
template<template<typename> class BATCHNORM>
struct resnet
{
    // the resnet basic block, where REG is bn_con or affine
    template<long num_filters, template<typename> class BN, int stride, typename SUBNET>
    using basicblock = BN<dlib::con<num_filters, 3, 3, 1, 1,
                  dlib::relu<BN<dlib::con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

    // the resnet bottleneck block
    template<long num_filters, template<typename> class BN, int stride, typename SUBNET>
    using bottleneck = BN<dlib::con<4 * num_filters, 1, 1, 1, 1,
                  dlib::relu<BN<dlib::con<num_filters, 3, 3, stride, stride,
                  dlib::relu<BN<dlib::con<num_filters, 1, 1, 1, 1, SUBNET>>>>>>>>;

    // the resnet residual
    template<
        template<long, template<typename> class, int, typename> class BLOCK, // basicblock or bottleneck
        long num_filters,
        template<typename> class BN, // bn_con or affine
        typename SUBNET
    > // adds the block to the result of tag1 (the subnet)
    using residual = dlib::add_prev1<BLOCK<num_filters, BN, 1, dlib::tag1<SUBNET>>>;

    // a resnet residual that does subsampling on both paths
    template<
        template<long, template<typename> class, int, typename> class BLOCK, // basicblock or bottleneck
        long num_filters,
        template<typename> class BN, // bn_con or affine
        typename SUBNET
    >
    using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2,
                          dlib::skip1<dlib::tag2<BLOCK<num_filters, BN, 2,
                          dlib::tag1<SUBNET>>>>>>;

    // residual block with optional downsampling and custom regularization (bn_con or affine)
    template<
        template<template<long, template<typename> class, int, typename> class, long, template<typename>class, typename> class RESIDUAL,
        template<long, template<typename> class, int, typename> class BLOCK,
        long num_filters,
        template<typename> class BN, // bn_con or affine
        typename SUBNET
    >
    using residual_block = dlib::relu<RESIDUAL<BLOCK, num_filters, BN, SUBNET>>;

    template<long num_filters, typename SUBNET>
    using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, BATCHNORM, SUBNET>;
    template<long num_filters, typename SUBNET>
    using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, BATCHNORM, SUBNET>;

    // some definitions to allow the use of the repeat layer
    template<typename SUBNET> using resbasicblock_512 = residual_block<residual, basicblock, 512, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbasicblock_256 = residual_block<residual, basicblock, 256, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbasicblock_128 = residual_block<residual, basicblock, 128, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbasicblock_64  = residual_block<residual, basicblock,  64, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbottleneck_512 = residual_block<residual, bottleneck, 512, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbottleneck_256 = residual_block<residual, bottleneck, 256, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbottleneck_128 = residual_block<residual, bottleneck, 128, BATCHNORM, SUBNET>;
    template<typename SUBNET> using resbottleneck_64  = residual_block<residual, bottleneck,  64, BATCHNORM, SUBNET>;

    // common processing for standard resnet inputs
    template<template<typename> class BN, typename INPUT>
    using input_processing = dlib::max_pool<3, 3, 2, 2, dlib::relu<BN<dlib::con<64, 7, 7, 2, 2, INPUT>>>>;

    // the resnet backbone with basicblocks
    template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
    using backbone_basicblock =
        dlib::repeat<nb_512, resbasicblock_512, resbasicblock_down<512,
        dlib::repeat<nb_256, resbasicblock_256, resbasicblock_down<256,
        dlib::repeat<nb_128, resbasicblock_128, resbasicblock_down<128,
        dlib::repeat<nb_64,  resbasicblock_64, input_processing<BATCHNORM, INPUT>>>>>>>>;

    // the resnet backbone with bottlenecks
    template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
    using backbone_bottleneck =
        dlib::repeat<nb_512, resbottleneck_512, resbottleneck_down<512,
        dlib::repeat<nb_256, resbottleneck_256, resbottleneck_down<256,
        dlib::repeat<nb_128, resbottleneck_128, resbottleneck_down<128,
        dlib::repeat<nb_64,  resbottleneck_64, input_processing<BATCHNORM, INPUT>>>>>>>>;

    // the backbones for the classic architectures
    template<typename INPUT> using backbone_18  = backbone_basicblock<1, 1, 1, 2, INPUT>;
    template<typename INPUT> using backbone_34  = backbone_basicblock<2, 5, 3, 3, INPUT>;
    template<typename INPUT> using backbone_50  = backbone_bottleneck<2, 5, 3, 3, INPUT>;
    template<typename INPUT> using backbone_101 = backbone_bottleneck<2, 22, 3, 3, INPUT>;
    template<typename INPUT> using backbone_152 = backbone_bottleneck<2, 35, 7, 3, INPUT>;

    // the typical classifier models
    using l18  = dlib::loss_multiclass_log<dlib::fc<1000, dlib::avg_pool_everything<backbone_18<dlib::input_rgb_image>>>>;
    using l34  = dlib::loss_multiclass_log<dlib::fc<1000, dlib::avg_pool_everything<backbone_34<dlib::input_rgb_image>>>>;
    using l50  = dlib::loss_multiclass_log<dlib::fc<1000, dlib::avg_pool_everything<backbone_50<dlib::input_rgb_image>>>>;
    using l101 = dlib::loss_multiclass_log<dlib::fc<1000, dlib::avg_pool_everything<backbone_101<dlib::input_rgb_image>>>>;
    using l152 = dlib::loss_multiclass_log<dlib::fc<1000, dlib::avg_pool_everything<backbone_152<dlib::input_rgb_image>>>>;
};
