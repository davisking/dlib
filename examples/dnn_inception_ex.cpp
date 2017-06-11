// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  I'm assuming you have already read the introductory
    dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp examples.  In this
    example we are going to show how to create inception networks. 

    An inception network is composed of inception blocks of the form:

               input from SUBNET
              /        |        \
             /         |         \
          block1    block2  ... blockN 
             \         |         /
              \        |        /
          concatenate tensors from blocks
                       |
                    output
                 
    That is, an inception blocks runs a number of smaller networks (e.g. block1,
    block2) and then concatenates their results.  For further reading refer to:
    Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of
    the IEEE Conference on Computer Vision and Pattern Recognition. 2015.
*/

#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;

// Inception layer has some different convolutions inside.  Here we define
// blocks as convolutions with different kernel size that we will use in
// inception layer block.
template <typename SUBNET> using block_a1 = relu<con<10,1,1,1,1,SUBNET>>;
template <typename SUBNET> using block_a2 = relu<con<10,3,3,1,1,relu<con<16,1,1,1,1,SUBNET>>>>;
template <typename SUBNET> using block_a3 = relu<con<10,5,5,1,1,relu<con<16,1,1,1,1,SUBNET>>>>;
template <typename SUBNET> using block_a4 = relu<con<10,1,1,1,1,max_pool<3,3,1,1,SUBNET>>>;

// Here is inception layer definition. It uses different blocks to process input
// and returns combined output.  Dlib includes a number of these inceptionN
// layer types which are themselves created using concat layers.  
template <typename SUBNET> using incept_a = inception4<block_a1,block_a2,block_a3,block_a4, SUBNET>;

// Network can have inception layers of different structure.  It will work
// properly so long as all the sub-blocks inside a particular inception block
// output tensors with the same number of rows and columns.
template <typename SUBNET> using block_b1 = relu<con<4,1,1,1,1,SUBNET>>;
template <typename SUBNET> using block_b2 = relu<con<4,3,3,1,1,SUBNET>>;
template <typename SUBNET> using block_b3 = relu<con<4,1,1,1,1,max_pool<3,3,1,1,SUBNET>>>;
template <typename SUBNET> using incept_b = inception3<block_b1,block_b2,block_b3,SUBNET>;

// Now we can define a simple network for classifying MNIST digits.  We will
// train and test this network in the code below.
using net_type = loss_multiclass_log<
        fc<10,
        relu<fc<32,
        max_pool<2,2,2,2,incept_b<
        max_pool<2,2,2,2,incept_a<
        input<matrix<unsigned char>>
        >>>>>>>>;

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


    // Make an instance of our inception network.
    net_type net;
    cout << "The net has " << net.num_layers << " layers in it." << endl;
    cout << net << endl;


    cout << "Traning NN..." << endl;
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    trainer.set_synchronization_file("inception_sync", std::chrono::seconds(20));
    // Train the network.  This might take a few minutes...
    trainer.train(training_images, training_labels);

    // At this point our net object should have learned how to classify MNIST images.  But
    // before we try it out let's save it to disk.  Note that, since the trainer has been
    // running images through the network, net will have a bunch of state in it related to
    // the last batch of images it processed (e.g. outputs from each layer).  Since we
    // don't care about saving that kind of stuff to disk we can tell the network to forget
    // about that kind of transient data so that our file will be smaller.  We do this by
    // "cleaning" the network before saving it.
    net.clean();
    serialize("mnist_network_inception.dat") << net;
    // Now if we later wanted to recall the network from disk we can simply say:
    // deserialize("mnist_network_inception.dat") >> net;


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

    // Let's also see if the network can correctly classify the testing images.
    // Since MNIST is an easy dataset, we should see 99% accuracy.
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

