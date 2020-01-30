// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use the general purpose non-linear
    optimization routines from the dlib C++ Library.

    The library provides implementations of many popular algorithms such as L-BFGS
    and BOBYQA.  These algorithms allow you to find the minimum or maximum of a
    function of many input variables.  This example walks though a few of the ways
    you might put these routines to use.

*/


#include <dlib/optimization.h>
#include <dlib/global_optimization.h>
#include <iostream>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

// In dlib, most of the general purpose solvers optimize functions that take a
// column vector as input and return a double.  So here we make a typedef for a
// variable length column vector of doubles.  This is the type we will use to
// represent the input to our objective functions which we will be minimizing.
typedef matrix<double,0,1> column_vector;

// ----------------------------------------------------------------------------------------
// Below we create a few functions.  When you get down into main() you will see that
// we can use the optimization algorithms to find the minimums of these functions.
// ----------------------------------------------------------------------------------------

double rosen (const column_vector& m)
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
const column_vector rosen_derivative (const column_vector& m)
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

// This function computes the Hessian matrix for the rosen() fuction.  This is
// the matrix of second derivatives.
matrix<double> rosen_hessian (const column_vector& m)
{
    const double x = m(0);
    const double y = m(1);

    matrix<double> res(2,2);

    // now compute the second derivatives 
    res(0,0) = 1200*x*x - 400*y + 2; // second derivative with respect to x
    res(1,0) = res(0,1) = -400*x;   // derivative with respect to x and y
    res(1,1) = 200;                 // second derivative with respect to y
    return res;
}

// ----------------------------------------------------------------------------------------

class rosen_model 
{
    /*!
        This object is a "function model" which can be used with the
        find_min_trust_region() routine.  
    !*/

public:
    typedef ::column_vector column_vector;
    typedef matrix<double> general_matrix;

    double operator() (
        const column_vector& x
    ) const { return rosen(x); }

    void get_derivative_and_hessian (
        const column_vector& x,
        column_vector& der,
        general_matrix& hess
    ) const
    {
        der = rosen_derivative(x);
        hess = rosen_hessian(x);
    }
};

// ----------------------------------------------------------------------------------------

int main() try
{
    // Set the starting point to (4,8).  This is the point the optimization algorithm
    // will start out from and it will move it closer and closer to the function's 
    // minimum point.   So generally you want to try and compute a good guess that is
    // somewhat near the actual optimum value.
    column_vector starting_point = {4, 8};

    // The first example below finds the minimum of the rosen() function and uses the
    // analytical derivative computed by rosen_derivative().  Since it is very easy to
    // make a mistake while coding a function like rosen_derivative() it is a good idea
    // to compare your derivative function against a numerical approximation and see if
    // the results are similar.  If they are very different then you probably made a 
    // mistake.  So the first thing we do is compare the results at a test point: 
    cout << "Difference between analytic derivative and numerical approximation of derivative: " 
         << length(derivative(rosen)(starting_point) - rosen_derivative(starting_point)) << endl;


    cout << "Find the minimum of the rosen function()" << endl;
    // Now we use the find_min() function to find the minimum point.  The first argument
    // to this routine is the search strategy we want to use.  The second argument is the 
    // stopping strategy.  Below I'm using the objective_delta_stop_strategy which just 
    // says that the search should stop when the change in the function being optimized 
    // is small enough.

    // The other arguments to find_min() are the function to be minimized, its derivative, 
    // then the starting point, and the last is an acceptable minimum value of the rosen() 
    // function.  That is, if the algorithm finds any inputs to rosen() that gives an output 
    // value <= -1 then it will stop immediately.  Usually you supply a number smaller than 
    // the actual global minimum.  So since the smallest output of the rosen function is 0 
    // we just put -1 here which effectively causes this last argument to be disregarded.

    find_min(bfgs_search_strategy(),  // Use BFGS search algorithm
             objective_delta_stop_strategy(1e-7), // Stop when the change in rosen() is less than 1e-7
             rosen, rosen_derivative, starting_point, -1);
    // Once the function ends the starting_point vector will contain the optimum point 
    // of (1,1).
    cout << "rosen solution:\n" << starting_point << endl;


    // Now let's try doing it again with a different starting point and the version
    // of find_min() that doesn't require you to supply a derivative function.  
    // This version will compute a numerical approximation of the derivative since 
    // we didn't supply one to it.
    starting_point = {-94, 5.2};
    find_min_using_approximate_derivatives(bfgs_search_strategy(),
                                           objective_delta_stop_strategy(1e-7),
                                           rosen, starting_point, -1);
    // Again the correct minimum point is found and stored in starting_point
    cout << "rosen solution:\n" << starting_point << endl;


    // Here we repeat the same thing as above but this time using the L-BFGS 
    // algorithm.  L-BFGS is very similar to the BFGS algorithm, however, BFGS 
    // uses O(N^2) memory where N is the size of the starting_point vector.  
    // The L-BFGS algorithm however uses only O(N) memory.  So if you have a 
    // function of a huge number of variables the L-BFGS algorithm is probably 
    // a better choice.
    starting_point = {0.8, 1.3};
    find_min(lbfgs_search_strategy(10),  // The 10 here is basically a measure of how much memory L-BFGS will use.
             objective_delta_stop_strategy(1e-7).be_verbose(),  // Adding be_verbose() causes a message to be 
                                                                // printed for each iteration of optimization.
             rosen, rosen_derivative, starting_point, -1);

    cout << endl << "rosen solution: \n" << starting_point << endl;

    starting_point = {-94, 5.2};
    find_min_using_approximate_derivatives(lbfgs_search_strategy(10),
                                           objective_delta_stop_strategy(1e-7),
                                           rosen, starting_point, -1);
    cout << "rosen solution: \n"<< starting_point << endl;




    // dlib also supports solving functions subject to bounds constraints on
    // the variables.  So for example, if you wanted to find the minimizer
    // of the rosen function where both input variables were in the range
    // 0.1 to 0.8 you would do it like this:
    starting_point = {0.1, 0.1}; // Start with a valid point inside the constraint box.
    find_min_box_constrained(lbfgs_search_strategy(10),  
                             objective_delta_stop_strategy(1e-9),  
                             rosen, rosen_derivative, starting_point, 0.1, 0.8);
    // Here we put the same [0.1 0.8] range constraint on each variable, however, you
    // can put different bounds on each variable by passing in column vectors of
    // constraints for the last two arguments rather than scalars.  

    cout << endl << "constrained rosen solution: \n" << starting_point << endl;

    // You can also use an approximate derivative like so:
    starting_point = {0.1, 0.1}; 
    find_min_box_constrained(bfgs_search_strategy(),  
                             objective_delta_stop_strategy(1e-9),  
                             rosen, derivative(rosen), starting_point, 0.1, 0.8);
    cout << endl << "constrained rosen solution: \n" << starting_point << endl;




    // In many cases, it is useful if we also provide second derivative information
    // to the optimizers.  Two examples of how we can do that are shown below.  
    starting_point = {0.8, 1.3};
    find_min(newton_search_strategy(rosen_hessian),
             objective_delta_stop_strategy(1e-7),
             rosen,
             rosen_derivative,
             starting_point,
             -1);
    cout << "rosen solution: \n"<< starting_point << endl;

    // We can also use find_min_trust_region(), which is also a method which uses
    // second derivatives.  For some kinds of non-convex function it may be more
    // reliable than using a newton_search_strategy with find_min().
    starting_point = {0.8, 1.3};
    find_min_trust_region(objective_delta_stop_strategy(1e-7),
                          rosen_model(), 
                          starting_point, 
                          10 // initial trust region radius
    );
    cout << "rosen solution: \n"<< starting_point << endl;





    // Next, let's try the BOBYQA algorithm.  This is a technique specially
    // designed to minimize a function in the absence of derivative information.  
    // Generally speaking, it is the method of choice if derivatives are not available
    // and the function you are optimizing is smooth and has only one local optima.  As
    // an example, consider the be_like_target function defined below:
    column_vector target = {3, 5, 1, 7};
    auto be_like_target = [&](const column_vector& x) {
        return mean(squared(x-target));
    };
    starting_point = {-4,5,99,3};
    find_min_bobyqa(be_like_target, 
                    starting_point, 
                    9,    // number of interpolation points
                    uniform_matrix<double>(4,1, -1e100),  // lower bound constraint
                    uniform_matrix<double>(4,1, 1e100),   // upper bound constraint
                    10,    // initial trust region radius
                    1e-6,  // stopping trust region radius
                    100    // max number of objective function evaluations
    );
    cout << "be_like_target solution:\n" << starting_point << endl;





    // Finally, let's try the find_min_global() routine.  Like find_min_bobyqa(),
    // this technique is specially designed to minimize a function in the absence
    // of derivative information.  However, it is also designed to handle
    // functions with many local optima.  Where BOBYQA would get stuck at the
    // nearest local optima, find_min_global() won't.  find_min_global() uses a
    // global optimization method based on a combination of non-parametric global
    // function modeling and BOBYQA style quadratic trust region modeling to
    // efficiently find a global minimizer.  It usually does a good job with a
    // relatively small number of calls to the function being optimized.  
    // 
    // You also don't have to give it a starting point or set any parameters,
    // other than defining bounds constraints.  This makes it the method of
    // choice for derivative free optimization in the presence of multiple local
    // optima.  Its API also allows you to define functions that take a
    // column_vector as shown above or to explicitly use named doubles as
    // arguments, which we do here.
    auto complex_holder_table = [](double x0, double x1)
    {
        // This function is a version of the well known Holder table test
        // function, which is a function containing a bunch of local optima.
        // Here we make it even more difficult by adding more local optima
        // and also a bunch of discontinuities. 

        // add discontinuities
        double sign = 1;
        for (double j = -4; j < 9; j += 0.5)
        {
            if (j < x0 && x0 < j+0.5) 
                x0 += sign*0.25;
            sign *= -1;
        }
        // Holder table function tilted towards 10,10 and with additional
        // high frequency terms to add more local optima.
        return -( std::abs(sin(x0)*cos(x1)*exp(std::abs(1-std::sqrt(x0*x0+x1*x1)/pi))) -(x0+x1)/10 - sin(x0*10)*cos(x1*10));
    };

    // To optimize this difficult function all we need to do is call
    // find_min_global()
    auto result = find_min_global(complex_holder_table, 
                                  {-10,-10}, // lower bounds
                                  {10,10}, // upper bounds
                                  std::chrono::milliseconds(500) // run this long
                                  );

    cout.precision(9);
    // These cout statements will show that find_min_global() found the
    // globally optimal solution to 9 digits of precision:
    cout << "complex holder table function solution y (should be -21.9210397): " << result.y << endl;
    cout << "complex holder table function solution x:\n" << result.x << endl;
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

