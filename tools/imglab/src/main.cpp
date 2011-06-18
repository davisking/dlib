
#include <iostream>
#include <fstream>

#include <dlib/cmd_line_parser.h>

using namespace std;
using namespace dlib;

int main(int argc, char** argv)
{
    try
    {
        typedef dlib::cmd_line_parser<char>::check_1a_c parser_type;

        parser_type parser;

        parser.add_option("h","Displays this information.");
        parser.add_option("c","Create an XML file named <arg> listing a set of images.",1);

        parser.parse(argc, argv);

        const char* singles[] = {"h","c"};
        parser.check_one_time_options(singles);

        if (parser.option("h"))
        {
            cout << "Options:\n";
            parser.print_options(cout);
            cout << endl;
            return EXIT_SUCCESS;
        }

        if (parser.option("c"))
        {

            return EXIT_SUCCESS;
        }

    }
    catch (exception& e)
    {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }
}

