// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

    This is an example illustrating the use of the parallel for loop
    tools from the dlib C++ Library.


*/


#include <dlib/threads.h>
#include <dlib/misc_api.h>  // for dlib::sleep
#include <vector>
#include <iostream>

using namespace dlib;
using namespace std;

struct function_object
{
    function_object( std::vector<int>& vect ) : vect1(vect) {}

    std::vector<int>& vect1;

    void operator() (long i) const
    {
        vect1[i] = i;
        dlib::sleep(100);
    }
};

int main()
{

    const unsigned long num_threads = 4;
    
    std::vector<int> vect1(10);
    parallel_for(num_threads, 0, vect1.size(), function_object(vect1));

    for (unsigned long i = 0; i < vect1.size(); ++i)
        cout << vect1[i] << endl;
    cout << "\n**************************************\n";

    vect1.assign(10, -1);
    parallel_for(num_threads, 1, 5, function_object(vect1));
    for (unsigned long i = 0; i < vect1.size(); ++i)
        cout << vect1[i] << endl;
    cout << "\n**************************************\n";


// uncomment this line if your compiler supports the new C++0x lambda functions
#define COMPILER_SUPPORTS_CPP0X_LAMBDA_FUNCTIONS
#ifdef COMPILER_SUPPORTS_CPP0X_LAMBDA_FUNCTIONS

    std::vector<int> vect2(10);
    parallel_for(num_threads, 0, vect2.size(), [&](long i){
        vect2[i] = i;
        dlib::sleep(100);
    });

    for (unsigned long i = 0; i < vect2.size(); ++i)
        cout << vect2[i] << endl;

#endif

}





