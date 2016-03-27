
/*

    Train the venerable LeNet from 
        LeCun, Yann, et al. "Gradient-based learning applied to document recognition."
        Proceedings of the IEEE 86.11 (1998): 2278-2324.
    on MNIST
*/


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;
 
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


    typedef loss_multiclass_log<fc<relu<fc<relu<fc<max_pool<relu<con<max_pool<relu<con<
                               input<matrix<unsigned char>>>>>>>>>>>>>> net_type;

    net_type net(fc_(10),
                 relu_(),
                 fc_(84),
                 relu_(),
                 fc_(120),
                 max_pool_(2,2,2,2),
                 relu_(),
                 con_(16,5,5),
                 max_pool_(2,2,2,2),
                 relu_(),
                 con_(6,5,5));

    dnn_trainer<net_type> trainer(net,sgd(0.1));
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
    trainer.train(training_images, training_labels);

    net.clean();
    serialize("mnist_network.dat") << net;

    // Run the net on all the data to get predictions
    std::vector<unsigned long> predicted_labels = net(training_images);
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

    predicted_labels = net(testing_images);
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

