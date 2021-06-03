// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This mex file will create a MATLAB function called example_mex_class.  If you call it
    with no arguments it will output the MATLAB .m code to create a MATLAB wrapper class.
    Paste that code into a .m file.  Then you will be able to work with this C++ class
    directly in MATLAB.
*/

#include <iostream>
#include <dlib/matrix.h>


using namespace std;
using namespace dlib;

class example_class 
{
public:

    // The class must have a default constructor.  It's also the only kind of constructor
    // you can call from MATLAB.
    example_class()
    {
        xx.set_size(3,2);
        xx = 1;
    }

    // The rest of the member functions that you want to bind have to return void and
    // generally have the same syntax limitations as regular mex funcitons.
    void do_stuff(const matrix_colmajor& x)
    {
        cout << "in do_stuff" << endl;
        cout << x << endl;
        xx = x;
    }

    void do_other_stuff(int x)
    {
        cout << "in do_other_stuff" << endl;
        cout << "x: " << x << endl;
    }

    void print_state()
    {
        cout << xx << endl;
    }

    // saveobj() and load_obj() are special functions. If you provide these then you will
    // be able to save() and load() your objects using MATLAB's built in object
    // serialization.
    void saveobj(matrix_colmajor& state)
    {
        // save this object's state to state.
        state = xx;
    }
    void load_obj(const matrix_colmajor& state)
    {
        xx = state;
    }

private:
    matrix_colmajor xx;
};

// Just tell the mex wrapper the name of your class and list the methods you want to bind.
#define MEX_CLASS_NAME example_class 
#define MEX_CLASS_METHODS do_stuff, do_other_stuff, print_state, saveobj, load_obj


#include "mex_wrapper.cpp"


