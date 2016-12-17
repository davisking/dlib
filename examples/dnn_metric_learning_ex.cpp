// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the
    dlib C++ Library.  In it, we will show how to use the loss_metric layer to do
    metric learning.  


*/


#include <dlib/dnn.h>
#include <iostream>

using namespace std;
using namespace dlib;


int main() try
{
    using net_type = loss_metric<fc<2,input<matrix<double,0,1>>>>; 

    net_type net;
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.1);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    trainer.set_iterations_without_progress_threshold(100);



    std::vector<matrix<double,0,1>> samples;
    std::vector<unsigned long> labels;

    samples.push_back({1,0,0,0,0,0,0,0}); labels.push_back(1);
    samples.push_back({0,1,0,0,0,0,0,0}); labels.push_back(1);

    samples.push_back({0,0,1,0,0,0,0,0}); labels.push_back(2);
    samples.push_back({0,0,0,1,0,0,0,0}); labels.push_back(2);

    samples.push_back({0,0,0,0,1,0,0,0}); labels.push_back(3);
    samples.push_back({0,0,0,0,0,1,0,0}); labels.push_back(3);

    samples.push_back({0,0,0,0,0,0,1,0}); labels.push_back(4);
    samples.push_back({0,0,0,0,0,0,0,1}); labels.push_back(4);

    trainer.train(samples, labels);


    auto embedded = net(samples);

    for (size_t i = 0; i < embedded.size(); ++i)
        cout << "label: " << labels[i] << "\t" << trans(embedded[i]);

    // now count how many pairs are correctly classified.
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < embedded.size(); ++i)
    {
        for (size_t j = i+1; j < embedded.size(); ++j)
        {
            if (labels[i] == labels[j])
            {
                if (length(embedded[i]-embedded[j]) < net.loss_details().get_distance_threshold())
                    ++num_right;
                else
                    ++num_wrong;
            }
            else
            {
                if (length(embedded[i]-embedded[j]) < net.loss_details().get_distance_threshold())
                    ++num_wrong;
                else
                    ++num_right;
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

