// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt

#include "call_matlab.h"
#include "dlib/matrix.h"
using namespace dlib;
using namespace std;


/*
    This mex function takes a MATLAB struct, prints a few of its fields,
    and then returns a new struct.

    For example, you can call this function in MATLAB like so:
        input = {}
        input.val = 2
        input.stuff = 'some string'
        output = example_mex_struct(input)

        output.number
        output.number2
        output.sub.stuff
        output.sub.some_matrix
*/


void mex_function (
    const matlab_struct& input,
    matlab_struct& output 
) 
{
    int val = input["val"];
    string stuff = input["stuff"];

    if (input.has_field("val2")) 
    {
        string val2 = input["val2"];
        cout << "The optional val2 field was set to: " << val2 << endl;
    }

    cout << "val: "<< val << endl;
    cout << "stuff: " << stuff << endl;

    output["number"] = 999;

    output["number2"] = 1000;
    output["sub"]["stuff"] = "some other string";
    matrix<double> m = randm(2,2);
    output["sub"]["some_matrix"] = m;
}



// #including this brings in all the mex boiler plate needed by MATLAB.
#include "mex_wrapper.cpp"

