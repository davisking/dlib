#ifndef ResNet_H
#define ResNet_H

#include <dlib/dnn.h>

namespace resnet
{
    using namespace dlib;
    // BN is bn_con or affine layer
    template<template<typename> class BN>
    struct def
    {
        // the resnet basic block, where BN is bn_con or affine
        template<long num_filters, int stride, typename SUBNET>
        using basicblock = BN<con<num_filters, 3, 3, 1, 1,
                relu<BN<con<num_filters, 3, 3, stride, stride, SUBNET>>>>>;

        // the resnet bottleneck block
        template<long num_filters, int stride, typename SUBNET>
        using bottleneck = BN<con<4 * num_filters, 1, 1, 1, 1,
                relu<BN<con<num_filters, 3, 3, stride, stride,
                relu<BN<con<num_filters, 1, 1, 1, 1, SUBNET>>>>>>>>;

        // the resnet residual, where BLOCK is either basicblock or bottleneck
        template<template<long, int, typename> class BLOCK, long num_filters, typename SUBNET>
        using residual = add_prev1<BLOCK<num_filters, 1, tag1<SUBNET>>>;

        // a resnet residual that does subsampling on both paths
        template<template<long, int, typename> class BLOCK, long num_filters, typename SUBNET>
        using residual_down = add_prev2<avg_pool<2, 2, 2, 2,
                              skip1<tag2<BLOCK<num_filters, 2,
                              tag1<SUBNET>>>>>>;

        // residual block with optional downsampling
        template<
            template<template<long, int, typename> class, long, typename> class RESIDUAL,
            template<long, int, typename> class BLOCK,
            long num_filters,
            typename SUBNET
        >
        using residual_block = relu<RESIDUAL<BLOCK, num_filters, SUBNET>>;

        template<long num_filters, typename SUBNET>
        using resbasicblock_down = residual_block<residual_down, basicblock, num_filters, SUBNET>;
        template<long num_filters, typename SUBNET>
        using resbottleneck_down = residual_block<residual_down, bottleneck, num_filters, SUBNET>;

        // some definitions to allow the use of the repeat layer
        template<typename SUBNET> using resbasicblock_512 = residual_block<residual, basicblock, 512, SUBNET>;
        template<typename SUBNET> using resbasicblock_256 = residual_block<residual, basicblock, 256, SUBNET>;
        template<typename SUBNET> using resbasicblock_128 = residual_block<residual, basicblock, 128, SUBNET>;
        template<typename SUBNET> using resbasicblock_64  = residual_block<residual, basicblock,  64, SUBNET>;
        template<typename SUBNET> using resbottleneck_512 = residual_block<residual, bottleneck, 512, SUBNET>;
        template<typename SUBNET> using resbottleneck_256 = residual_block<residual, bottleneck, 256, SUBNET>;
        template<typename SUBNET> using resbottleneck_128 = residual_block<residual, bottleneck, 128, SUBNET>;
        template<typename SUBNET> using resbottleneck_64  = residual_block<residual, bottleneck,  64, SUBNET>;

        // common processing for standard resnet inputs
        template<typename INPUT>
        using input_processing = max_pool<3, 3, 2, 2, relu<BN<con<64, 7, 7, 2, 2, INPUT>>>>;

        // the resnet backbone with basicblocks
        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone_basicblock =
            repeat<nb_512, resbasicblock_512, resbasicblock_down<512,
            repeat<nb_256, resbasicblock_256, resbasicblock_down<256,
            repeat<nb_128, resbasicblock_128, resbasicblock_down<128,
            repeat<nb_64,  resbasicblock_64, input_processing<INPUT>>>>>>>>;

        // the resnet backbone with bottlenecks
        template<long nb_512, long nb_256, long nb_128, long nb_64, typename INPUT>
        using backbone_bottleneck =
            repeat<nb_512, resbottleneck_512, resbottleneck_down<512,
            repeat<nb_256, resbottleneck_256, resbottleneck_down<256,
            repeat<nb_128, resbottleneck_128, resbottleneck_down<128,
            repeat<nb_64,  resbottleneck_64, input_processing<INPUT>>>>>>>>;

        // the backbones for the classic architectures
        template<typename INPUT> using backbone_18  = backbone_basicblock<1, 1, 1, 2, INPUT>;
        template<typename INPUT> using backbone_34  = backbone_basicblock<2, 5, 3, 3, INPUT>;
        template<typename INPUT> using backbone_50  = backbone_bottleneck<2, 5, 3, 3, INPUT>;
        template<typename INPUT> using backbone_101 = backbone_bottleneck<2, 22, 3, 3, INPUT>;
        template<typename INPUT> using backbone_152 = backbone_bottleneck<2, 35, 7, 3, INPUT>;

        // the typical classifier models
        using n18  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_18<input_rgb_image>>>>;
        using n34  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_34<input_rgb_image>>>>;
        using n50  = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_50<input_rgb_image>>>>;
        using n101 = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_101<input_rgb_image>>>>;
        using n152 = loss_multiclass_log<fc<1000, avg_pool_everything<backbone_152<input_rgb_image>>>>;
    };

    using train_18 = def<bn_con>::n18;
    using train_34 = def<bn_con>::n34;
    using train_50 = def<bn_con>::n50;
    using train_101 = def<bn_con>::n101;
    using train_152 = def<bn_con>::n152;

    using infer_18 = def<affine>::n18;
    using infer_34 = def<affine>::n34;
    using infer_50 = def<affine>::n50;
    using infer_101 = def<affine>::n101;
    using infer_152 = def<affine>::n152;
}

#endif // ResNet_H
