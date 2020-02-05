// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  I'm assuming you have already read the dnn_introduction_ex.cpp and
    the dnn_introduction2_ex.cpp examples.  So in this example program I'm going to go
    over a transfer learning example, which includes:
        - Defining a layer visitor to modify the learning rate for fine-tuning
        - Using pretrained layers of a network for another task
*/

#include <dlib/dnn.h>
#include <iostream>

// This header file includes a generic definition of the most common ResNet architectures
#include "resnet.h"

using namespace std;
using namespace dlib;

// In this simple example we will show how to load a pretrained network
// and use it for a different task.  In particular, we will load a ResNet50
// trained on ImageNet and use it as a pretrained backbone for some metric
// learning task

// Let's start by defining a network that will use the ResNet50 backbone from
// resnet.h
namespace model
{
    template<template<typename> class BN>
    using net_type = loss_metric<
        fc_no_bias<128,
        avg_pool_everything<
        typename resnet<BN>::template backbone_50<
        input_rgb_image
        >>>>;

    using train = net_type<bn_con>;
    using infer = net_type<affine>;
}

// Next, we define a layer visitor that will modify the learning rate of a network.
// In particular, we will modify the learning rate and the bias learning rate multipliers
class visitor_learning_rate_multiplier
{
public:

    visitor_learning_rate_multiplier(double new_learning_rate_multiplier_) :
        new_learning_rate_multiplier(new_learning_rate_multiplier_) {}

    template <typename T>
    void set_new_learning_rate_multiplier(T& l) const
    {
        set_learning_rate_multiplier(l, new_learning_rate_multiplier);
        set_bias_learning_rate_multiplier(l, new_learning_rate_multiplier);
    }

    template<typename input_layer_type>
    void operator()(size_t , input_layer_type& )  const
    {
        // ignore other layers
    }

    template <typename T, typename U, typename E>
    void operator()(size_t , add_layer<T,U,E>& l)  const
    {
        set_new_learning_rate_multiplier(l.layer_details());
    }

private:

    double new_learning_rate_multiplier;
};


int main() try
{
    // Let's instantiate our network in train mode
    model::train net;

    // We create a new scope so that resources from the loaded network are freed
    // automatically when leaving the scope.
    {
        // Now, let's define the classic ResNet50 network and load the pretrained model on ImageNet
        resnet<bn_con>::n50 resnet50;
        std::vector<string> labels;
        deserialize("resnet50_1000_imagenet_classifier.dnn") >> resnet50 >> labels;
        // For transfer learning, we are only interested in the ResNet50's backbone, which lays
        // below the loss and the fc layers, so we can extract it as:
        auto backbone = std::move(resnet50.subnet().subnet());
        // We can now assign ResNet50's backbone to our network skipping the different layers,
        // in our case, the loss layer and the fc layer:
        net.subnet().subnet() = backbone;

        // An alternative way to use the pretrained network on a different network is
        // to extract the relevant part of the network (we remove loss and fc layers),
        // stack the new layers on top of it and assign the network.
        using net_type = loss_metric<fc_no_bias<128, decltype(backbone)>>;
        net_type net2;
        net2.subnet().subnet() = backbone;
    }

    // We can use the visit_layers function to modify the learning rate of the whole network:
    visit_layers(net, visitor_learning_rate_multiplier(0.01));

    // Usually, we want to freeze the network, except for the top layers:
    visit_layers(net.subnet().subnet(), visitor_learning_rate_multiplier(0));

    // However, sometimes we might want to adjust the learning rate differently thoughout the
    // network.  Here we show how to use the visit_layers_range to adjust the learning rate
    // different at the different ResNet50's convolutional blocks:
    visit_layers_range<  0,   2>(net, visitor_learning_rate_multiplier(1));
    visit_layers_range<  2,  38>(net, visitor_learning_rate_multiplier(0.1));
    visit_layers_range< 38, 107>(net, visitor_learning_rate_multiplier(0.01));
    visit_layers_range<107, 154>(net, visitor_learning_rate_multiplier(0.001));
    visit_layers_range<154, 193>(net, visitor_learning_rate_multiplier(0.0001));

    // Finally, we can check the results by printing the network.  But before, if we forward
    // an image through the network, we will see tensors shape at every layer.
    matrix<rgb_pixel> image(224, 224);
    assign_all_pixels(image, rgb_pixel(0, 0, 0));
    std::vector<matrix<rgb_pixel>> minibatch(1, image);
    resizable_tensor input;
    net.to_tensor(minibatch.begin(), minibatch.end(), input);
    net.subnet().forward(input);
    cout << net << endl;
    cout << "input size=(" <<
       "num:" << input.num_samples() << ", " <<
       "k:" << input.k() << ", " <<
       "nr:" << input.nr() << ", "
       "nc:" << input.nc() << ")" << endl;

    // We can also print the number of parameters of the network
    cout << "number of network parameters: " << count_parameters(net) << endl;

    // From this point on, we can finetune the new network using this pretrained backbone on another task,
    // such as the one showed in dnn_metric_learning_on_images_ex.cpp.

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
