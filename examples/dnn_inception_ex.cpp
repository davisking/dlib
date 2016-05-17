// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  I'm assuming you have already read the dnn_mnist_ex.cpp
    example.  So in this example program I'm going to go over a number of more
    advanced parts of the API, including:
        - Using grp layer for constructing inception layer

    Inception layer is a kind of NN architecture for running sevelar convolution types
    on the same input area and joining all convolution results into one output.
    For further reading refer http://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
*/


#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>
#include <tuple>

using namespace std;
using namespace dlib;

// Here we define inception module as described in GoogLeNet specification. The depth of each sublayer can be changed
template<typename SUBNET>
using inception = grp<std::tuple<con<8,1,1,1,1, group_input>,
                                 con<8,3,3,1,1, con<8,1,1,1,1, group_input>>,
                                 con<8,5,5,1,1, con<8,1,1,1,1, group_input>>,
                                 con<8,1,1,1,1, max_pool<3,3,1,1, group_input>>>,
                      SUBNET>;

int main(int argc, char** argv) try
{
    // This example is going to run on the MNIST dataset.  
    if (argc != 2)
    {
        cout << "This example needs the MNIST dataset to run!" << endl;
        cout << "You can get MNIST from http://yann.lecun.com/exdb/mnist/" << endl;
        cout << "Download the 4 files that comprise the dataset, decompress them, and" << endl;
        cout << "put them in a folder.  Then give that folder as input to this program." << endl;
        return 1;
    }


    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long>         training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long>         testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    // Create a the same network as in dnn_mnist_ex, but use inception layer insteam of convolution
    // in the middle
    using net_type = loss_multiclass_log<
            fc<10,
                    relu<fc<84,
                            relu<fc<120,
                                    max_pool<2,2,2,2,relu<inception<
                                            max_pool<2,2,2,2,relu<con<6,5,5,1,1,
                                                    input<matrix<unsigned char>>
                                            >>>>>>>>>>>>;


    // Create a network as defined above.  This network will produce 10 outputs
    // because that's how we defined net_type.  However, fc layers can have the
    // number of outputs they produce changed at runtime.
    net_type net;

    // the following training process is the same as in dnn_mnist_ex sample

    // And then train it using the MNIST data.  The code below uses mini-batch stochastic
    // gradient descent with an initial learning rate of 0.01 to accomplish this.
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    // Since DNN training can take a long time, we can ask the trainer to save its state to
    // a file named "mnist_sync" every 20 seconds.  This way, if we kill this program and
    // start it again it will begin where it left off rather than restarting the training
    // from scratch.  This is because, when the program restarts, this call to
    // set_synchronization_file() will automatically reload the settings from mnist_sync if
    // the file exists.
    trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
    // Finally, this line begins training.  By default, it runs SGD with our specified
    // learning rate until the loss stops decreasing.  Then it reduces the learning rate by
    // a factor of 10 and continues running until the loss stops decreasing again.  It will
    // keep doing this until the learning rate has dropped below the min learning rate
    // defined above or the maximum number of epochs as been executed (defaulted to 10000). 
    trainer.train(training_images, training_labels);

    // At this point our net object should have learned how to classify MNIST images.  But
    // before we try it out let's save it to disk.  Note that, since the trainer has been
    // running images through the network, net will have a bunch of state in it related to
    // the last batch of images it processed (e.g. outputs from each layer).  Since we
    // don't care about saving that kind of stuff to disk we can tell the network to forget
    // about that kind of transient data so that our file will be smaller.  We do this by
    // "cleaning" the network before saving it.
    net.clean();
    serialize("mnist_network.dat") << net;
    // Now if we later wanted to recall the network from disk we can simply say:
    // deserialize("mnist_network.dat") >> net;


    // Now let's run the training images through the network.  This statement runs all the
    // images through it and asks the loss layer to convert the network's raw output into
    // labels.  In our case, these labels are the numbers between 0 and 9.
    std::vector<unsigned long> predicted_labels = net(training_images);
    int num_right = 0;
    int num_wrong = 0;
    // And then let's see if it classified them correctly.
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

    // Let's also see if the network can correctly classify the testing images.  Since
    // MNIST is an easy dataset, we should see at least 99% accuracy.
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

