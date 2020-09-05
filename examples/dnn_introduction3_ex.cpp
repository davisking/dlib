// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  I'm assuming you have already read the dnn_introduction_ex.cpp and
    the dnn_introduction2_ex.cpp examples.  So in this example program I'm going to go
    over a transfer learning example, which includes:
        - Defining a layer visitor to modify the some network parameters for fine-tuning
        - Using pretrained layers of a network for another task
*/

#include <dlib/dnn.h>
#include <iostream>

// This header file includes a generic definition of the most common ResNet architectures
#include "resnet.h"

using namespace std;
using namespace dlib;

// In this simple example we will show how to load a pretrained network and use it for a
// different task.  In particular, we will load a ResNet50 trained on ImageNet, adjust
// some of its parameters and use it as a pretrained backbone for some metric learning
// task.

// Let's start by defining a network that will use the ResNet50 backbone from resnet.h
namespace model
{
    template<template<typename> class BN>
    using net_type = loss_metric<
        fc_no_bias<128,
        avg_pool_everything<
        typename resnet::def<BN>::template backbone_50<
        input_rgb_image
        >>>>;

    using train = net_type<bn_con>;
    using infer = net_type<affine>;
}

// Next, we define a layer visitor that will modify the weight decay of a network.  The
// main interest of this class is to show how one can define custom visitors that modify
// some network parameters.
class visitor_weight_decay_multiplier
{
public:

    visitor_weight_decay_multiplier(double new_weight_decay_multiplier_) :
        new_weight_decay_multiplier(new_weight_decay_multiplier_) {}

    template <typename layer>
    void operator()(layer& l) const
    {
        set_weight_decay_multiplier(l, new_weight_decay_multiplier);
    }

private:

    double new_weight_decay_multiplier;
};


int main() try
{
    // Let's instantiate our network in train mode.
    model::train net;

    // We create a new scope so that resources from the loaded network are freed
    // automatically when leaving the scope.
    {
        // Now, let's define the classic ResNet50 network and load the pretrained model on
        // ImageNet.
        resnet::train_50 resnet50;
        std::vector<string> labels;
        deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50 >> labels;

        // For transfer learning, we are only interested in the ResNet50's backbone, which
        // lays below the loss and the fc layers, so we can extract it as:
        auto backbone = std::move(resnet50.subnet().subnet());

        // We can now assign ResNet50's backbone to our network skipping the different
        // layers, in our case, the loss layer and the fc layer:
        net.subnet().subnet() = backbone;

        // An alternative way to use the pretrained network on a different
        // network is to extract the relevant part of the network (we remove
        // loss and fc layers), stack the new layers on top of it and assign
        // the network.
        using net_type = loss_metric<fc_no_bias<128, decltype(backbone)>>;
        net_type net2;
        net2.subnet().subnet() = backbone;
    }

    // We can use the visit_layers function to modify the weight decay of the entire
    // network:
    visit_computational_layers(net, visitor_weight_decay_multiplier(0.001));

    // We can also use predefined visitors to affect the learning rate of the whole
    // network.
    set_all_learning_rate_multipliers(net, 0.5);

    // Modifying the learning rates of a network is a common practice for fine tuning, for
    // this reason it is already provided. However, it is implemented internally using a
    // visitor that is very similar to the one defined in this example.

    // Usually, we want to freeze the network, except for the top layers:
    visit_computational_layers(net.subnet().subnet(), visitor_weight_decay_multiplier(0));
    set_all_learning_rate_multipliers(net.subnet().subnet(), 0);

    // Alternatively, we can use the visit_layers_range to modify only a specific set of
    // layers:
    visit_computational_layers_range<0, 2>(net, visitor_weight_decay_multiplier(1));

    // Sometimes we might want to set the learning rate differently throughout the network.
    // Here we show how to use adjust the learning rate at the different ResNet50's
    // convolutional blocks:
    set_learning_rate_multipliers_range<  0,   2>(net, 1);
    set_learning_rate_multipliers_range<  2,  38>(net, 0.1);
    set_learning_rate_multipliers_range< 38, 107>(net, 0.01);
    set_learning_rate_multipliers_range<107, 154>(net, 0.001);
    set_learning_rate_multipliers_range<154, 193>(net, 0.0001);

    // Finally, we can check the results by printing the network.  But before, if we
    // forward an image through the network, we will see tensors shape at every layer.
    matrix<rgb_pixel> image(224, 224);
    assign_all_pixels(image, rgb_pixel(0, 0, 0));
    std::vector<matrix<rgb_pixel>> minibatch(1, image);
    resizable_tensor input;
    net.to_tensor(minibatch.begin(), minibatch.end(), input);
    net.forward(input);
    cout << net << endl;
    cout << "input size=(" <<
       "num:" << input.num_samples() << ", " <<
       "k:" << input.k() << ", " <<
       "nr:" << input.nr() << ", "
       "nc:" << input.nc() << ")" << endl;

    // We can also print the number of parameters of the network:
    cout << "number of network parameters: " << count_parameters(net) << endl;

    // From this point on, we can fine-tune the new network using this pretrained backbone
    // on another task, such as the one showed in dnn_metric_learning_on_images_ex.cpp.

    return EXIT_SUCCESS;
}
catch (const serialization_error& e)
{
    cout << e.what() << endl;
    cout << "You need to download a copy of the file resnet50_1000_imagenet_classifier.dnn" << endl;
    cout << "available at http://dlib.net/files/resnet50_1000_imagenet_classifier.dnn.bz2" << endl;
    cout << endl;
    return EXIT_FAILURE;
}
catch (const exception& e)
{
    cout << e.what() << endl;
    return EXIT_FAILURE;
}
