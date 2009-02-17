// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use the optimization 
    routines from the dlib C++ Library.

    The library provides implementations of the conjugate gradient and 
    quasi-newton BFGS optimization algorithms.  Both of these algorithms 
    allow you to find the minimum of a function of many input variables.  
    This example walks though a few of the ways you might put these 
    routines to use.

*/


#include "dlib/optimization.h"
#include <iostream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// Here we just make a typedef for a variable length column vector of doubles.  
typedef matrix<double,0,1> column_vector;

// ----------------------------------------------------------------------------------------
// Below we create a few functions.  When you get down into main() you will see that
// we can use the optimization algorithms to find the minimums of these functions.
// ----------------------------------------------------------------------------------------

double rosen ( const column_vector& m)
/*
    This function computes what is known as Rosenbrock's function.  It is 
    a function of two input variables and has a global minimum at (1,1).
    So when we use this function to test out the optimization algorithms
    we will see that the minimum found is indeed at the point (1,1). 
*/
{
    const double x = m(0); 
    const double y = m(1);

    // compute Rosenbrock's function and return the result
    return 100.0*pow(y - x*x,2) + pow(1 - x,2);
}

// This is a helper function used while optimizing the rosen() function.  
const column_vector rosen_derivative ( const column_vector& m)
/*!
    ensures
        - returns the gradient vector for the rosen function
!*/
{
    const double x = m(0);
    const double y = m(1);

    // make us a column vector of length 2
    column_vector res(2);

    // now compute the gradient vector
    res(0) = -400*x*(y-x*x) - 2*(1-x); // derivative of rosen() with respect to x
    res(1) = 200*(y-x*x);              // derivative of rosen() with respect to y
    return res;
}

// ----------------------------------------------------------------------------------------

class test_function
{
    /*
        This object is an example of what is known as a "function object" in C++.
        It is simply an object with an overloaded operator().  This means it can 
        be used in a way that is similar to a normal C function.  The interesting
        thing about this sort of function is that it can have state.  
        
        In this example, our test_function object contains a column_vector 
        as its state and it computes the mean squared error between this 
        stored column_vector and the arguments to its operator() function.
        This is a very simple function.  However, in general you could compute
        any function you wanted here.  An example of a typical use would be 
        to find the parameters to some regression function that minimized 
        the mean squared error on a set of data.  In this case the arguments
        to the operator() function would be the parameters of your regression
        function and you would use those parameters to loop over all your data
        samples, compute the output of the regression function given those 
        parameters, and finally return a measure of the error.   The dlib 
        optimization functions would then be used to find the parameters that 
        minimized the error.
    */
public:

    test_function (
        const column_vector& input
    )
    {
        target = input;
    }

    double operator() ( const column_vector& arg) const
    {
        // return the mean squared error between the target vector and the input vector
        return mean(squared(target-arg));
    }

private:
    column_vector target;
};

// ----------------------------------------------------------------------------------------

int main()
{
    // make a column vector of length 2
    column_vector starting_point;
    starting_point.set_size(2);

    cout << "Find the minimum of the rosen function()" << endl;

    // Set the starting point to (4,8).  This is the point the optimization algorithm
    // will start out from and it will slowly move it closer and closer to the
    // function's minimum point
    starting_point = 4, 8;
    // Now we use the quasi newton algorithm to find the minimum point.  The first argument
    // to this routine is the function we wish to minimize, the second is the 
    // derivative of that function, the third is the starting point, and the last is
    // an acceptable minimum value of the rosen() function.  That is, if the algorithm
    // finds any inputs to rosen() that gives an output value <= -1 then it will
    // stop immediately.  Usually you supply a number smaller than the actual 
    // global minimum.  So since the smallest output of the rosen function is 0 
    // we just put -1 here which effectively causes this last argument to be disregarded.
    find_min_quasi_newton(&rosen, &rosen_derivative, starting_point, -1);
    // Once the function ends the starting_point vector will contain the optimum point 
    // of (1,1).
    cout << starting_point << endl;


    // Now lets try doing it again with a different starting point and the version
    // of the quasi newton algorithm that doesn't require you to supply 
    // a derivative function.  This version will compute a numerical approximation
    // of the derivative since we didn't supply one to it.
    starting_point = -94, 5.2;
    find_min_quasi_newton2(&rosen, starting_point, -1);
    // Again the correct minimum point is found and stored in starting_point
    cout << starting_point << endl;


    // Here we repeat the same thing as above but this time using the conjugate 
    // gradient algorithm.  As a rule of thumb, the quasi newton algorithm is 
    // a better algorithm.  However, it uses O(N^2) memory where N is the size
    // of the starting_point vector.  The conjugate gradient algorithm however
    // uses only O(N) memory.  So if you have a function of a huge number
    // of variables the conjugate gradient algorithm is often a better choice.
    starting_point = 4, 8;
    find_min_conjugate_gradient(&rosen, &rosen_derivative, starting_point, -1);
    cout << starting_point << endl;

    starting_point = -94, 5.2;
    find_min_conjugate_gradient2(&rosen, starting_point, -1);
    cout << starting_point << endl;





    // Now lets look at using the test_function object with the optimization 
    // functions.  
    cout << "\nFind the minimum of the test_function" << endl;

    column_vector target;
    target.set_size(4);
    starting_point.set_size(4);

    // This variable will be used as the target of the test_function.   So,
    // our simple test_function object will have a global minimum at the
    // point given by the target.  We will then use the optimization 
    // routines to find this minimum value.
    target = 3, 5, 1, 7;

    // set the starting point far from the global minimum
    starting_point = 1,2,3,4;
    find_min_quasi_newton2(test_function(target), starting_point, -1);
    // At this point the correct value of (3,6,1,7) should be found and stored in starting_point
    cout << starting_point << endl;

    // Now lets try it again with the conjugate gradient algorithm.
    starting_point = -4,5,99,3;
    find_min_conjugate_gradient2(test_function(target), starting_point, -1);
    cout << starting_point << endl;

}

