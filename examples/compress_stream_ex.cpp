/*

    This is an example illustrating the use of the compress_stream and
    cmd_line_parser components from the dlib C++ Library.  

    This example implements a simple command line compression utility.


    The output from the program when the -h option is given is:

        Usage: dclib_example (-c|-d) --in input_file --out output_file
        Options:
        -c                       Indicates that we want to compress a file.
        -d                       Indicates that we want to decompress a file.
        -h                       Display this help message.
        --in <arg>               This option takes one argument which specifies the
                                 name of the file we want to compress/decompress.
        --out <arg>              This option takes one argument which specifies the
                                 name of the output file.

*/




#include "dlib/compress_stream.h"
#include "dlib/cmd_line_parser.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

// I am making a typedef for the verson of compress_stream I want to use.
// I have selected kernel_1ec.
typedef dlib::compress_stream::kernel_1ec cs;

// Here I am making another typedef, this time for the verson of
// cmd_line_parser I want to use.  I have selected print_1a_c,
// this is the version of kernel_1a that checks all its 
// preconditions (i.e. the debugging version) and is 
// extended by print_kernel_1. 
typedef dlib::cmd_line_parser<char>::print_1a_c clp;


using namespace std;
using namespace dlib;


int main(int argc, char** argv)
{  
    try
    {
        clp parser;
        cs compressor;

        // first I will define the command line options I want
        parser.add_option("c","Indicates that we want to compress a file.");
        parser.add_option("d","Indicates that we want to decompress a file.");
        parser.add_option("in","This option takes one argument which specifies the name of the file we want to compress/decompress.",1);
        parser.add_option("out","This option takes one argument which specifies the name of the output file.",1);
        parser.add_option("h","Display this help message.");

        // now I will parse the command line
        parser.parse(argc,argv);

        // check if the -h option was given on the command line
        if (parser.option("h"))
        {
            // display all the command line options
            cout << "Usage: dclib_example (-c|-d) --in input_file --out output_file\n";
            parser.print_options(cout); // this print_options() function is really 
                                        // convenient :)
            cout << endl;
            return 0;
        }

        const clp::option_type& option_c = parser.option("c");
        const clp::option_type& option_d = parser.option("d");
        const clp::option_type& option_in = parser.option("in");
        const clp::option_type& option_out = parser.option("out");

        if ((option_c.count() != 0 && option_d.count() != 0 ) || 
            (option_c.count() == 0 && option_d.count() == 0 ) )
        {
            cout << "Error in command line:\n   You must specify either the c option or the d option.\n";
            cout << "\nTry the -h option for more information." << endl;
            return 0;
        }


        string in_file;
        string out_file;

        // check if the user told us the input file and if they did then 
        // get the file name
        if (option_in.count() == 1)
        {
            in_file = option_in.argument();
        }
        else if (option_in.count() > 1)
        {
            cout << "Error in command line:\n   You must specify only one input file.\n";
            cout << "\nTry the -h option for more information." << endl;
            return 0;
        }
        else
        {
            cout << "Error in command line:\n   You must specify an input file.\n";
            cout << "\nTry the -h option for more information." << endl;
            return 0;
        }


        // check if the user told us the output file and if they did then 
        // get the file name
        if (option_out.count() == 1)
        {
            out_file = option_out.argument();
        }
        else if (option_out.count() > 1)
        {
            cout << "Error in command line:\n   You must specify only one output file.\n";
            cout << "\nTry the -h option for more information." << endl;
            return 0;
        }
        else
        {
            cout << "Error in command line:\n   You must specify an output file.\n";
            cout << "\nTry the -h option for more information." << endl;
            return 0;
        }


        ifstream fin(in_file.c_str(),ios::binary);
        ofstream fout(out_file.c_str(),ios::binary);

        if (!fin)
        {
            cout << "Error opening file " << in_file << ".\n";
            return 0;
        }

        if (!fout)
        {
            cout << "Error creating file " << out_file << ".\n";
            return 0;
        }



        // now perform the actual compression or decompression.
        if (option_c)
        {
            compressor.compress(fin,fout);
        }
        else
        {
            compressor.decompress(fin,fout);
        }


        

    }
    catch (exception& e)
    {
        // Note that this will catch any cmd_line_parse_error exceptions and print
        // the default message.   
        cout << e.what() << endl;
    }
    catch (...)
    {
        cout << "Some error occurred" << endl;
    }
}





