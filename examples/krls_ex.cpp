/*
    This is an example illustrating the use of the krls object 
    from the dlib C++ Library.

    The krls object allows you to perform online regression.  This
    example will train an instance of it on the sinc function.

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
    // Here we declare that our samples will be 1 dimensional column vectors.  The reason for
    // using a matrix here is that in general you can use N dimensional vectors as inputs to the
    // krls object.  But here we only have 1 dimension to make the example simple.
    typedef matrix<double,1,1> sample_type;

    // Now we are making a typedef for the kind of kernel we want to use.  I picked the
    // radial basis kernel because it only has one parameter and generally gives good
    // results without much fiddling.
    typedef radial_basis_kernel<sample_type> kernel_type;

    // Here we declare an instance of the krls object.  The first argument to the constructor
    // is the kernel we wish to use.  The second is a parameter that determines the numerical 
    // accuracy with which the object will perform part of the regression algorithm.  Generally
    // smaller values give better results but cause the algorithm to run slower.  You just have
    // to play with it to decide what balance of speed and accuracy is right for your problem.
    // Here we have set it to 0.001.
    krls<kernel_type> test(kernel_type(0.1),0.001);

    // now we train our object on a few samples of the sinc function.
    sample_type m;
    for (double x = -10; x <= 4; x += 1)
    {
        m(0) = x;
        test.train(m, sinc(x));
    }

    // now we output the value of the sinc function for a few test points as well as the 
    // value predicted by krls object.
    m(0) = 2.5; cout << sinc(m(0)) << "   " << test(m) << endl;
    m(0) = 0.1; cout << sinc(m(0)) << "   " << test(m) << endl;
    m(0) = -4;  cout << sinc(m(0)) << "   " << test(m) << endl;
    m(0) = 5.0; cout << sinc(m(0)) << "   " << test(m) << endl;

    // The output is as follows:
    // 0.239389   0.238808
    // 0.998334   0.997779
    // -0.189201   -0.189754
    // -0.191785   -0.1979

    // The first column is the true value of the sinc function and the second
    // column is the output from the krls estimate.  

}


