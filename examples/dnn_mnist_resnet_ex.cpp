

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

template <typename T> using res = relu<add_prev1<bn<con<relu<bn<con<tag1<T>>>>>>>>;

std::tuple<relu_,add_prev1_,bn_,con_,relu_,bn_,con_> res_ (
    unsigned long outputs,
    unsigned long stride = 1
) 
{
    return std::make_tuple(relu_(),
                           add_prev1_(),
                           bn_(CONV_MODE),
                           con_(outputs,3,3,stride,stride),
                           relu_(),
                           bn_(CONV_MODE),
                           con_(outputs,3,3,stride,stride));
}

template <typename T> using ares = relu<add_prev1<affine<con<relu<affine<con<tag1<T>>>>>>>>;

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



    typedef loss_multiclass_log<fc<avg_pool<
                                res<res<res<res<
                                repeat<10,res,
                                res<
                                res<
                                input<matrix<unsigned char>
                                >>>>>>>>>>> net_type;


    const unsigned long number_of_classes = 10;
    net_type net(fc_(number_of_classes),
                 avg_pool_(10,10,10,10),
                 res_(8),res_(8),res_(8),res_(8,2),
                 res_(8), // repeated 10 times
                 res_(8,2),
                 res_(8)
                );


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

    // You can access sub layers of the network like this:
    net.subnet().subnet().get_output();
    layer<2>(net).get_output();
    layer<avg_pool>(net).get_output();

    net.clean();
    serialize("mnist_res_network.dat") << net;



    typedef loss_multiclass_log<fc<avg_pool<
                                ares<ares<ares<ares<
                                repeat<10,ares,
                                ares<
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

