/*
    This is an example illustrating the use of the kkmeans object 
    from the dlib C++ Library.

    The kkmeans object is an implementation of a kernelized k-means clustering 
    algorithm.  It is implemented by using the kcentroid object to represent 
    each center found by the usual k-means clustering algorithm.  

    So this object allows you to perform non-linear clustering in the same way 
    a svm classifier finds non-linear decision surfaces.  
    
    This example will make points from 3 classes and perform kernelized k-means 
    clustering on those points.

    The classes are as follows:
        - points very close to the origin
        - points on the circle of radius 10 around the origin
        - points that are on a circle of radius 4 but not around the origin at all
*/

#include <iostream>
#include <vector>

#include "dlib/svm.h"
#include "dlib/rand.h"

using namespace std;
using namespace dlib;

int main()
{
    // Here we declare that our samples will be 2 dimensional column vectors.  
    typedef matrix<double,2,1> sample_type;

    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the kcentroid object.  The first argument to the constructor
    // is the kernel we wish to use.  The second is a parameter that determines the numerical 
    // accuracy with which the object will perform part of the learning algorithm.  Generally
    // smaller values give better results but cause the algorithm to run slower.  You just have
    // to play with it to decide what balance of speed and accuracy is right for your problem.
    // Here we have set it to 0.01.
    kcentroid<kernel_type> kc(kernel_type(0.1),0.01);

    // Now we make an instance of the kkmeans object and tell it to use kcentroid objects
    // that are configured with the parameters from the kc object we defined above.
    kkmeans<kernel_type> test(kc);

    std::vector<sample_type> samples;
    std::vector<sample_type> initial_centers;

    sample_type m;

    dlib::rand::float_1a rnd;

    // we will make 50 points from each class
    const long num = 50;

    // make some samples near the origin
    double radius = 0.5;
    for (long i = 0; i < num; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));

        // add this sample to our set of samples we will run k-means 
        samples.push_back(m);
    }

    // make some samples in a circle around the origin but far away
    radius = 10.0;
    for (long i = 0; i < num; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));

        // add this sample to our set of samples we will run k-means 
        samples.push_back(m);
    }

    // make some samples in a circle around the point (25,25) 
    radius = 4.0;
    for (long i = 0; i < num; ++i)
    {
        double sign = 1;
        if (rnd.get_random_double() < 0.5)
            sign = -1;
        m(0) = 2*radius*rnd.get_random_double()-radius;
        m(1) = sign*sqrt(radius*radius - m(0)*m(0));

        // translate this point away from the origin
        m(0) += 25;
        m(1) += 25;

        // add this sample to our set of samples we will run k-means 
        samples.push_back(m);
    }

    // tell the kkmeans object we made that we want to run k-means with k set to 3. 
    // (i.e. we want 3 clusters)
    test.set_number_of_centers(3);

    // You need to pick some initial centers for the k-means algorithm.  So here
    // we will use the dlib::pick_initial_centers() function which tries to find
    // n points that are far apart (basically).  
    pick_initial_centers(3, initial_centers, samples, test.get_kernel());

    // now run the k-means algorithm on our set of samples.  
    test.train(samples,initial_centers);

    // now loop over all our samples and print out their predicted class.  In this example
    // all points are correctly identified.
    for (unsigned long i = 0; i < samples.size()/3; ++i)
    {
        cout << test(samples[i]) << " ";
        cout << test(samples[i+num]) << " ";
        cout << test(samples[i+2*num]) << "\n";
    }

}


