// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the parallel for loop tools from the dlib
    C++ Library.

    Normally, a for loop executes the body of the loop in a serial manner.  This means
    that, for example, if it takes 1 second to execute the body of the loop and the body
    needs to execute 10 times then it will take 10 seconds to execute the entire loop.
    However, on modern multi-core computers we have the opportunity to speed this up by
    executing multiple steps of a for loop in parallel.  This example program will walk you
    though a few examples showing how to do just that.  
*/


#include <dlib/threads.h>
#include <dlib/misc_api.h>  // for dlib::sleep
#include <vector>
#include <iostream>

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

void print(const std::vector<int>& vect)
{
    for (unsigned long i = 0; i < vect.size(); ++i)
    {
        cout << vect[i] << endl;
    }
    cout << "\n**************************************\n";
}

// ----------------------------------------------------------------------------------------

void example_using_regular_non_parallel_loops();
void example_using_lambda_functions();

// ----------------------------------------------------------------------------------------

int main()
{
    // We have 2 examples, each contained in a separate function.  Both examples perform
    // exactly the same computation, however, the second does so using parallel for loops.
    // The first example is here to show you what we are doing in terms of classical
    // non-parallel for loops.  The other example will illustrate how to parallelize the
    // for loops in C++11. 

    example_using_regular_non_parallel_loops();
    example_using_lambda_functions();
}

// ----------------------------------------------------------------------------------------

void example_using_regular_non_parallel_loops()
{
    cout << "\nExample using regular non-parallel for loops\n" << endl;

    std::vector<int> vect;

    // put 10 elements into vect which are all equal to -1
    vect.assign(10, -1);

    // Now set each element equal to its index value.  We put a sleep call in here so that
    // when we run the same thing with a parallel for loop later on you will be able to
    // observe the speedup. 
    for (unsigned long i = 0; i < vect.size(); ++i)
    {
        vect[i] = i;
        dlib::sleep(1000); // sleep for 1 second
    }
    print(vect);



    // Assign only part of the elements in vect.
    vect.assign(10, -1);
    for (unsigned long i = 1; i < 5; ++i)
    {
        vect[i] = i;
        dlib::sleep(1000);
    }
    print(vect);



    // Sum all element sin vect.
    int sum = 0;
    vect.assign(10, 2);
    for (unsigned long i = 0; i < vect.size(); ++i)
    {
        dlib::sleep(1000);
        sum += vect[i];
    }

    cout << "sum: "<< sum << endl;
}

// ----------------------------------------------------------------------------------------

void example_using_lambda_functions()
{
    cout << "\nExample using parallel for loops\n" << endl;

    std::vector<int> vect;

    vect.assign(10, -1);
    parallel_for(0, vect.size(), [&](long i){
        // The i variable is the loop counter as in a normal for loop.  So we simply need
        // to place the body of the for loop right here and we get the same behavior.  The
        // range for the for loop is determined by the 1nd and 2rd arguments to
        // parallel_for().  This way of calling parallel_for() will use a number of threads
        // that is appropriate for your hardware.  See the parallel_for() documentation for
        // other options.
        vect[i] = i;
        dlib::sleep(1000);
    });
    print(vect);


    // Assign only part of the elements in vect.
    vect.assign(10, -1);
    parallel_for(1, 5, [&](long i){
        vect[i] = i;
        dlib::sleep(1000);
    });
    print(vect);


    // Note that things become a little more complex if the loop bodies are not totally
    // independent.  In the first two cases each iteration of the loop touched different
    // memory locations, so we didn't need to use any kind of thread synchronization.
    // However, in the summing loop we need to add some synchronization to protect the sum
    // variable.  This is easily accomplished by creating a mutex and locking it before
    // adding to sum.  More generally, you must ensure that the bodies of your parallel for
    // loops are thread safe using whatever means is appropriate for your code.  Since a
    // parallel for loop is implemented using threads, all the usual techniques for
    // ensuring thread safety can be used. 
    int sum = 0;
    dlib::mutex m;
    vect.assign(10, 2);
    parallel_for(0, vect.size(), [&](long i){
        // The sleep statements still execute in parallel.  
        dlib::sleep(1000);

        // Lock the m mutex.  The auto_mutex will automatically unlock at the closing }.
        // This will ensure only one thread can execute the sum += vect[i] statement at
        // a time.
        auto_mutex lock(m);
        sum += vect[i];
    });

    cout << "sum: "<< sum << endl;
}

// ----------------------------------------------------------------------------------------

