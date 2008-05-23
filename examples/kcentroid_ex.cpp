/*
    This is an example illustrating the use of the kcentroid object 
    from the dlib C++ Library.

    The kcentroid object is an implementation of an algorithm that recursively
    computes the centroid (i.e. average) of a set of points.  The interesting
    thing about dlib::kcentroid is that it does so in a kernel induced feature
    space.  This means that you can use it as a non-linear one-class classifier.
    So you might use it to perform online novelty detection.  
    
    This example will train an instance of it on points from the sinc function.

*/

#include <iostream>
#include <vector>

#include "dlib/svm.h"

using namespace std;
using namespace dlib;

// Here is the sinc function we will be trying to learn with the krls
// object.
double sinc(double x)
{
    if (x == 0)
        return 1;
    return sin(x)/x;
}

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
    kcentroid<kernel_type> test(kernel_type(0.1),0.01);

    // now we train our object on a few samples of the sinc function.
    sample_type m;
    for (double x = -15; x <= 8; x += 1)
    {
        m(0) = x;
        m(1) = sinc(x);
        test.train(m);
    }


    // lets output the distance from the centroid to some points that are from the sinc function.
    // These numbers should all be similar
    cout << "Points that are on the sinc function:\n";
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << test(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << test(m) << endl;
    m(0) = -0;   m(1) = sinc(m(0)); cout << "   " << test(m) << endl;
    m(0) = -0.5; m(1) = sinc(m(0)); cout << "   " << test(m) << endl;
    m(0) = -4.1; m(1) = sinc(m(0)); cout << "   " << test(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0)); cout << "   " << test(m) << endl;
    m(0) = -0.5; m(1) = sinc(m(0)); cout << "   " << test(m) << endl;

    // lets output the distance from the centroid to some points that are NOT from the sinc function.
    // These numbers should all be bigger than previous set of numbers.  In fact, if you computed the
    // standard deviation of the above set of numbers you would note that these following numbers
    // are many standard deviations away from them which indicates that they are highly unlike
    // the set of points from above.
    cout << "Points that are NOT on the sinc function:\n";
    m(0) = -1.5; m(1) = sinc(m(0))+4;   cout << "   " << test(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0))+3;   cout << "   " << test(m) << endl;
    m(0) = -0;   m(1) = -sinc(m(0));    cout << "   " << test(m) << endl;
    m(0) = -0.5; m(1) = -sinc(m(0));    cout << "   " << test(m) << endl;
    m(0) = -4.1; m(1) = sinc(m(0))+2;   cout << "   " << test(m) << endl;
    m(0) = -1.5; m(1) = sinc(m(0))+0.9; cout << "   " << test(m) << endl;
    m(0) = -0.5; m(1) = sinc(m(0))+1;   cout << "   " << test(m) << endl;
}


