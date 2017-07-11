// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will show how to use the loss_metric layer to do
    metric learning.  

    The main reason you might want to use this kind of algorithm is because you
    would like to use a k-nearest neighbor classifier or similar algorithm, but
    you don't know a good way to calculate the distance between two things.  A
    popular example would be face recognition.  There are a whole lot of papers
    that train some kind of deep metric learning algorithm that embeds face
    images in some vector space where images of the same person are close to each
    other and images of different people are far apart.  Then in that vector
    space it's very easy to do face recognition with some kind of k-nearest
    neighbor classifier.  

    To keep this example as simple as possible we won't do face recognition.
    Instead, we will create a very simple network and use it to learn a mapping
    from 8D vectors to 2D vectors such that vectors with the same class labels
    are near each other.  If you want to see a more complex example that learns
    the kind of network you would use for something like face recognition read
    the dnn_metric_learning_on_images_ex.cpp example.

    You should also have read the examples that introduce the dlib DNN API before 
    continuing.  These are dnn_introduction_ex.cpp and dnn_introduction2_ex.cpp.
*/


#include <dlib/dnn.h>
#include <iostream>

using namespace std;
using namespace dlib;


int main() try
{
    // The API for doing metric learning is very similar to the API for
    // multi-class classification.  In fact, the inputs are the same, a bunch of
    // labeled objects.  So here we create our dataset.  We make up some simple
    // vectors and label them with the integers 1,2,3,4.  The specific values of
    // the integer labels don't matter.
    std::vector<matrix<double,0,1>> samples;
    std::vector<unsigned long> labels;

    // class 1 training vectors
    samples.push_back({1,0,0,0,0,0,0,0}); labels.push_back(1);
    samples.push_back({0,1,0,0,0,0,0,0}); labels.push_back(1);

    // class 2 training vectors
    samples.push_back({0,0,1,0,0,0,0,0}); labels.push_back(2);
    samples.push_back({0,0,0,1,0,0,0,0}); labels.push_back(2);

    // class 3 training vectors
    samples.push_back({0,0,0,0,1,0,0,0}); labels.push_back(3);
    samples.push_back({0,0,0,0,0,1,0,0}); labels.push_back(3);

    // class 4 training vectors
    samples.push_back({0,0,0,0,0,0,1,0}); labels.push_back(4);
    samples.push_back({0,0,0,0,0,0,0,1}); labels.push_back(4);


    // Make a network that simply learns a linear mapping from 8D vectors to 2D
    // vectors.
    using net_type = loss_metric<fc<2,input<matrix<double,0,1>>>>; 
    net_type net;
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);

    // It should be emphasized out that it's really important that each mini-batch contain
    // multiple instances of each class of object.  This is because the metric learning
    // algorithm needs to consider pairs of objects that should be close as well as pairs
    // of objects that should be far apart during each training step.  Here we just keep
    // training on the same small batch so this constraint is trivially satisfied.
    while(trainer.get_learning_rate() >= 1e-4)
        trainer.train_one_step(samples, labels);

    // Wait for training threads to stop
    trainer.get_net();
    cout << "done training" << endl;


    // Run all the samples through the network to get their 2D vector embeddings.
    std::vector<matrix<float,0,1>> embedded = net(samples);

    // Print the embedding for each sample to the screen.  If you look at the
    // outputs carefully you should notice that they are grouped together in 2D
    // space according to their label.
    for (size_t i = 0; i < embedded.size(); ++i)
        cout << "label: " << labels[i] << "\t" << trans(embedded[i]);

    // Now, check if the embedding puts things with the same labels near each other and
    // things with different labels far apart.
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < embedded.size(); ++i)
    {
        for (size_t j = i+1; j < embedded.size(); ++j)
        {
            if (labels[i] == labels[j])
            {
                // The loss_metric layer will cause things with the same label to be less
                // than net.loss_details().get_distance_threshold() distance from each
                // other.  So we can use that distance value as our testing threshold for
                // "being near to each other".
                if (length(embedded[i]-embedded[j]) < net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
            else
            {
                if (length(embedded[i]-embedded[j]) >= net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
        }
    }

    cout << "num_right: "<< num_right << endl;
    cout << "num_wrong: "<< num_wrong << endl;
}
catch(std::exception& e)
{
    cout << e.what() << endl;
}

