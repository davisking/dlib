// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  I'm assuming you have already read the dnn_mnist_ex.cpp
    example.  So in this example program I'm going to go over a number of more
    advanced parts of the API, including:
        - Training on large datasets that don't fit in memory 
        - Defining large networks
        - Accessing and configuring layers in a network
*/


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// Let's start by showing how you can conveniently define large networks.  The
// most important tool for doing this are C++'s alias templates.  These let us
// define new layer types that are combinations of a bunch of other layers.
// These will form the building blocks for more complex networks.

// So let's begin by defining the building block of a residual network (see
// Figure 2 in Deep Residual Learning for Image Recognition by He, Zhang, Ren,
// and Sun).  You can see a few things in this statement.  The most obvious is
// that we have combined a bunch of layers into the name "base_res".  You can
// also see the use of the tag1 layer.  This layer doesn't do any computation.
// It exists solely so other layers can refer to it.  In this case, the
// add_prev1 layer looks for the tag1 layer and will take the tag1 output and
// add it to the input of the add_prev1 layer.  This combination allows us to
// implement skip and residual style networks.  
template <int stride, typename SUBNET> 
using base_res  = relu<add_prev1<bn_con<con<8,3,3,1,1,relu<bn_con<con<8,3,3,stride,stride,tag1<SUBNET>>>>>>>>;

// Let's also define the same block but with all the batch normalization layers
// replaced with affine transform layers.  We will use this type of construction
// when testing our networks.
template <int stride, typename SUBNET> 
using base_ares = relu<add_prev1<affine<con<8,3,3,1,1,relu<affine<con<8,3,3,stride,stride,tag1<SUBNET>>>>>>>>;

// And of course we can define more alias templates based on previously defined
// alias templates.  The _down versions downsample the inputs by a factor of 2
// while the res and ares layer types don't.
template <typename SUBNET> using res       = base_res<1,SUBNET>;
template <typename SUBNET> using res_down  = base_res<2,SUBNET>;
template <typename SUBNET> using ares      = base_ares<1,SUBNET>;
template <typename SUBNET> using ares_down = base_ares<2,SUBNET>;



// Now that we have these convenient aliases, we can define a residual network
// without a lot of typing.  Note the use of a repeat layer.  This special layer
// type allows us to type repeat<9,res<SUBNET>> instead of
// res<res<res<res<res<res<res<res<res<SUBNET>>>>>>>>>.
const unsigned long number_of_classes = 10;
using net_type = loss_multiclass_log<fc<number_of_classes,
                            avg_pool<11,11,11,11,
                            res<res<res<res_down<
                            repeat<9,res, // repeat this layer 9 times
                            res_down<
                            res<
                            input<matrix<unsigned char>>
                            >>>>>>>>>>;


// And finally, let's define a residual network building block that uses
// parametric ReLU units instead of regular ReLU.
template <typename SUBNET> 
using pres  = prelu<add_prev1<bn_con<con<8,3,3,1,1,prelu<bn_con<con<8,3,3,1,1,tag1<SUBNET>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return 1;
    }

    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long> training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long> testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    // dlib uses cuDNN under the covers.  One of the features of cuDNN is the
    // option to use slower methods that use less RAM or faster methods that use
    // a lot of RAM.  If you find that you run out of RAM on your graphics card
    // then you can call this function and we will request the slower but more
    // RAM frugal cuDNN algorithms.
    set_dnn_prefer_smallest_algorithms();


    // Create a network as defined above.  This network will produce 10 outputs
    // because that's how we defined net_type.  However, fc layers can have the
    // number of outputs they produce changed at runtime.  
    net_type net;
    // So if you wanted to use the same network but override the number of
    // outputs at runtime you can do so like this:
    net_type net2(num_fc_outputs(15));

    // Now, let's imagine we wanted to replace some of the relu layers with
    // prelu layers.  We might do it like this:
    using net_type2 = loss_multiclass_log<fc<number_of_classes,
                                avg_pool<11,11,11,11,
                                pres<res<res<res_down< // 2 prelu layers here
                                tag4<repeat<9,pres,    // 9 groups, each containing 2 prelu layers  
                                res_down<
                                res<
                                input<matrix<unsigned char>>
                                >>>>>>>>>>>;

    // prelu layers have a floating point parameter.  If you want to set it to
    // something other than its default value you can do so like this:
    net_type2 pnet(prelu_(0.2),  
                   prelu_(0.2),
                   repeat_group(prelu_(0.3),prelu_(0.4)) // Initialize all the prelu instances in the repeat 
                                                         // layer.  repeat_group() is needed to group the 
                                                         // things that are part of repeat's block.
                   );
    // As you can see, a network will greedily assign things given to its
    // constructor to the layers inside itself.  The assignment is done in the
    // order the layers are defined, but it will skip layers where the
    // assignment doesn't make sense.  

    // The API shown above lets you modify layers at construction time.  But
    // what about after that?  There are a number of ways to access layers
    // inside a net object.

    // You can access sub layers of the network like this to get their output
    // tensors.  The following 3 statements are all equivalent and access the
    // same layer's output.
    pnet.subnet().subnet().subnet().get_output();
    layer<3>(pnet).get_output();
    layer<prelu>(pnet).get_output(); 
    // Similarly, to get access to the prelu_ object that defines the layer's
    // behavior we can say:
    pnet.subnet().subnet().subnet().layer_details();
    // or 
    layer<prelu>(pnet).layer_details(); 
    // So for example, to print the prelu parameter:
    cout << "first prelu layer's initial param value: "
         << pnet.subnet().subnet().subnet().layer_details().get_initial_param_value() << endl;

    // From this it should be clear that layer() is a general tool for accessing
    // sub layers.  It makes repeated calls to subnet() so you don't have to.
    // One of it's most important uses is to access tagged layers.  For example,
    // to access the first tag1 layer we can say:
    layer<tag1>(pnet);
    // To further illustrate the use of layer(), let's loop over the repeated
    // prelu layers and print out their parameters.  But first, let's grab a
    // reference to the repeat layer.  Since we tagged the repeat layer we can
    // access it using the layer() method.  layer<tag4>(pnet) returns the tag4
    // layer, but we want the repeat layer right after it so we can give an
    // integer as the second argument and it will jump that many layers down the
    // network.  In our case we need to jump just 1 layer down to get to repeat. 
    auto&& repeat_layer = layer<tag4,1>(pnet);
    for (size_t i = 0; i < repeat_layer.num_repetitions(); ++i)
    {
        // The repeat layer just instantiates the network block a bunch of
        // times.  get_repeated_layer() allows us to grab each of these
        // instances.
        auto&& repeated_layer = repeat_layer.get_repeated_layer(i);
        // Now that we have the i-th layer inside our repeat layer we can look
        // at its properties.  Recall that we repeated the "pres" network block,
        // which is itself a network with a bunch of layers.  So we can again
        // use layer() to jump to the prelu layers we are interested in like so:
        prelu_ prelu1 = layer<prelu>(repeated_layer).layer_details();
        prelu_ prelu2 = layer<prelu>(repeated_layer.subnet()).layer_details();
        cout << "first prelu layer parameter value: "<< prelu1.get_initial_param_value() << endl;;
        cout << "second prelu layer parameter value: "<< prelu2.get_initial_param_value() << endl;;
    }

    


    // Ok, so that's enough talk about defining networks.  Let's talk about
    // training networks!

    // The dnn_trainer will use SGD by default, but you can tell it to use
    // different solvers like adam.  
    dnn_trainer<net_type,adam> trainer(net,adam(0.001));
    trainer.be_verbose();
    trainer.set_synchronization_file("mnist_resnet_sync", std::chrono::seconds(100));
    // While the trainer is running it keeps an eye on the training error.  If
    // it looks like the error hasn't decreased for the last 2000 iterations it
    // will automatically reduce the step size by 0.1.  You can change these
    // default parameters to some other values by calling these functions.  Or
    // disable them entirely by setting the shrink amount to 1.
    trainer.set_iterations_without_progress_threshold(2000);
    trainer.set_step_size_shrink_amount(0.1);


    // Now, what if your training dataset is so big it doesn't fit in RAM?  You
    // make mini-batches yourself, any way you like, and you send them to the
    // trainer by repeatedly calling trainer.train_one_step(). 
    //
    // For example, the loop below stream MNIST data to out trainer.
    std::vector<matrix<unsigned char>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels; 
    dlib::rand rnd(time(0));
    // Loop until the trainer's automatic shrinking has shrunk the step size by
    // 1e-3.  For the default shrinks amount of 0.1 this means stop after it
    // shrinks it 3 times.
    while(trainer.get_step_size() >= 1e-3)
    {
        mini_batch_samples.clear();
        mini_batch_labels.clear();

        // make a 128 image mini-batch
        while(mini_batch_samples.size() < 128)
        {
            auto idx = rnd.get_random_32bit_number()%training_images.size();
            mini_batch_samples.push_back(training_images[idx]);
            mini_batch_labels.push_back(training_labels[idx]);
        }

        trainer.train_one_step(mini_batch_samples, mini_batch_labels);
    }

    // When you call train_one_step(), the trainer will do its processing in a
    // separate thread.  This allows the main thread to work on loading data
    // while the trainer is busy executing the mini-batches in parallel.
    // However, this also means we need to wait for any mini-batches that are
    // still executing to stop before we mess with the net object.  Calling
    // get_net() performs the necessary synchronization.
    trainer.get_net();


    net.clean();
    serialize("mnist_res_network.dat") << net;


    // Now we have a trained network.  However, it has batch normalization
    // layers in it.  As is customary, we should replace these with simple
    // affine layers before we use the network.  This can be accomplished by
    // making a network type which is identical to net_type but with the batch
    // normalization layers replaced with affine.  For example:
    using test_net_type = loss_multiclass_log<fc<number_of_classes,
                                avg_pool<11,11,11,11,
                                ares<ares<ares<ares_down<
                                repeat<9,res,
                                ares_down<
                                ares<
                                input<matrix<unsigned char>>
                                >>>>>>>>>>;
    // Then we can simply assign our trained net to our testing net.
    test_net_type tnet = net;
    // Or if you only had a file with your trained network you could deserialize
    // it directly into your testing network.  
    deserialize("mnist_res_network.dat") >> tnet;


    // And finally, we can run the testing network over our data.

    std::vector<unsigned long> predicted_labels = tnet(training_images);
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < training_images.size(); ++i)
    {
        if (predicted_labels[i] == training_labels[i])
            ++num_right;
        else
            ++num_wrong;
        
    }
    cout << "training num_right: " << num_right << endl;
    cout << "training num_wrong: " << num_wrong << endl;
    cout << "training accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

    predicted_labels = tnet(testing_images);
    num_right = 0;
    num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        if (predicted_labels[i] == testing_labels[i])
            ++num_right;
        else
            ++num_wrong;
        
    }
    cout << "testing num_right: " << num_right << endl;
    cout << "testing num_wrong: " << num_wrong << endl;
    cout << "testing accuracy:  " << num_right/(double)(num_right+num_wrong) << endl;

}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

