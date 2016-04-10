

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

template <int stride, typename SUBNET> 
using base_res  = relu<add_prev1<    bn_con<con<8,3,3,1,1,relu<    bn_con<con<8,3,3,stride,stride,tag1<SUBNET>>>>>>>>;

template <int stride, typename SUBNET> 
using base_ares = relu<add_prev1<affine<con<8,3,3,1,1,relu<affine<con<8,3,3,stride,stride,tag1<SUBNET>>>>>>>>;

template <typename SUBNET> using res       = base_res<1,SUBNET>;
template <typename SUBNET> using res_down  = base_res<2,SUBNET>;
template <typename SUBNET> using ares      = base_ares<1,SUBNET>;
template <typename SUBNET> using ares_down = base_ares<2,SUBNET>;

template <typename SUBNET> 
using pres  = prelu<add_prev1<    bn_con<con<8,3,3,1,1,prelu<    bn_con<con<8,3,3,1,1,tag1<SUBNET>>>>>>>>;

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv) try
{
    if (argc != 2)
    {
        cout << "give MNIST data folder!" << endl;
        return 1;
    }

    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long> training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long> testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    set_dnn_prefer_smallest_algorithms();

    const unsigned long number_of_classes = 10;
    typedef loss_multiclass_log<fc<number_of_classes,FC_HAS_BIAS,
                                avg_pool<11,11,11,11,
                                res<res<res<res_down<
                                repeat<9,res, // repeat this layer 9 times
                                res_down<
                                res<
                                input<matrix<unsigned char>
                                >>>>>>>>>>> net_type;


    net_type net;


    // If you wanted to use the same network but override the number of outputs at runtime
    // you can do so like this:
    net_type net2(num_fc_outputs(15));

    // Let's imagine we wanted to replace some of the relu layers with prelu layers.  We
    // might do it like this:
    typedef loss_multiclass_log<fc<number_of_classes,FC_HAS_BIAS,
                                avg_pool<11,11,11,11,
                                pres<res<res<res_down< // 2 prelu layers here
                                tag4<repeat<9,pres,    // 9 groups, each containing 2 prelu layers  
                                res_down<
                                res<
                                input<matrix<unsigned char>
                                >>>>>>>>>>>> net_type2;

    // prelu layers have a floating point parameter.  If you want to set it to something
    // other than its default value you can do so like this:
    net_type2 pnet(prelu_(0.2),  
                   prelu_(0.2),
                   repeat_group(prelu_(0.3),prelu_(0.4)) // Initialize all the prelu instances in the repeat 
                                                       // layer.  repeat_group() is needed to group the things 
                                                       // that are part of repeat's block.
                   );
    // As you can see, a network will greedily assign things given to its constructor to
    // the layers inside itself.  The assignment is done in the order the layers are
    // defined but it will skip layers where the assignment doesn't make sense.  


    // You can access sub layers of the network like this:
    net.subnet().subnet().get_output();
    layer<2>(net).get_output();
    layer<relu>(net).get_output();
    layer<tag1>(net).get_output();
    // To further illustrate the use of layer(), let's loop over the repeated layers and
    // print out their parameters.  But first, let's grab a reference to the repeat layer.
    // Since we tagged the repeat layer we can access it using the layer() method.
    // layer<tag4>(pnet) returns the tag4 layer, but we want the repeat layer so we can
    // give an integer as the second argument and it will jump that many layers down the
    // network.  In our case we need to jump just 1 layer down to get to repeat. 
    auto&& repeat_layer = layer<tag4,1>(pnet);
    for (size_t i = 0; i < repeat_layer.num_repetitions(); ++i)
    {
        // The repeat layer just instantiates the network block a bunch of times as a
        // network object.  get_repeated_layer() allows us to grab each of these instances.
        auto&& repeated_layer = repeat_layer.get_repeated_layer(i);
        // Now that we have the i-th layer inside our repeat layer we can look at its
        // properties.  Recall that we repeated the "pres" network block, which is itself a
        // network with a bunch of layers.  So we can again use layer() to jump to the
        // prelu layers we are interested in like so:
        prelu_ prelu1 = layer<prelu>(repeated_layer).layer_details();
        prelu_ prelu2 = layer<prelu>(repeated_layer.subnet()).layer_details();
        cout << "first prelu layer parameter value: "<< prelu1.get_initial_param_value() << endl;;
        cout << "second prelu layer parameter value: "<< prelu2.get_initial_param_value() << endl;;
    }

    


    dnn_trainer<net_type,adam> trainer(net,adam(0.001));
    trainer.be_verbose();
    trainer.set_synchronization_file("mnist_resnet_sync", std::chrono::seconds(100));
    std::vector<matrix<unsigned char>> mini_batch_samples;
    std::vector<unsigned long> mini_batch_labels; 
    dlib::rand rnd;
    //trainer.train(training_images, training_labels);
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
    // wait for threaded processing to stop.
    trainer.get_net();

    net.clean();
    serialize("mnist_res_network.dat") << net;



    typedef loss_multiclass_log<fc<number_of_classes,FC_HAS_BIAS,
                                avg_pool<11,11,11,11,
                                ares<ares<ares<ares_down<
                                repeat<9,res,
                                ares_down<
                                ares<
                                input<matrix<unsigned char>
                                >>>>>>>>>>> test_net_type;
    test_net_type tnet = net;
    // or you could deserialize the saved network
    deserialize("mnist_res_network.dat") >> tnet;


    // Run the net on all the data to get predictions
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

