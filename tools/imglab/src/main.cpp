
#include <iostream>
#include <fstream>
#include <string>

#include <dlib/cmd_line_parser.h>
#include <dlib/geometry.h>

#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>


using namespace std;
using namespace dlib;

// ----------------------------------------------------------------------------------------

namespace dlib
{
    namespace imglab
    {
        struct box
        {
            box() : head(-0xFFFF,-0xFFFF) {}

            rectangle rect;

            // optional fields
            std::string label;
            point head; // a value of (-0xFFFF,-0xFFFF) indicates the field not supplied

            bool has_head() const { return head != point(-0xFFFF,-0xFFFF); }
            bool has_label() const { return label.size() != 0; }
        };

        struct image
        {
            image() {}
            image(const std::string& f) : filename(f) {}

            std::string filename;
            std::vector<box> boxes;
        };

        struct image_dataset_metadata
        {
            std::vector<image> images;
            std::string comment;
            std::string name;
        };

    // ------------------------------------------------------------------------------------

        const std::string get_decoded_string();
        void create_image_metadata_stylesheet_file()
        {
            ofstream fout("image_metadata_stylesheet.xsl");
            if (!fout)
                throw dlib::error("ERROR: Unable to open image_metadata_stylesheet.xsl for writing.");

            fout << get_decoded_string();

            if (!fout)
                throw dlib::error("ERROR: Unable to write to image_metadata_stylesheet.xsl.");
        }

        void save_image_dataset_metadata (
            const image_dataset_metadata& metadata,
            const std::string& filename
        )
        {
            create_image_metadata_stylesheet_file();

            const std::vector<image>& images = metadata.images;

            ofstream fout(filename.c_str());
            if (!fout)
                throw dlib::error("ERROR: Unable to open " + filename + " for writing.");

            fout << "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
            fout << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n";
            fout << "<dataset>\n";
            fout << "<name>" << metadata.name << "</name>\n";
            fout << "<comment>" << metadata.comment << "</comment>\n";
            fout << "<images>\n";
            for (unsigned long i = 0; i < images.size(); ++i)
            {
                fout << "  <image file='" << images[i].filename << "'>\n";

                // save all the boxes
                for (unsigned long j = 0; j < images[i].boxes.size(); ++j)
                {
                    const box& b = images[i].boxes[j];
                    fout << "    <box top='" << b.rect.top() << "' "
                                 << "left='" << b.rect.left() << "' "
                                << "width='" << b.rect.width() << "' "
                               << "height='" << b.rect.height() << "'>\n";

                    if (b.has_label())
                        fout << "      <label>" << b.label << "</label>\n";
                    if (b.has_head())
                        fout << "      <head x='"<< b.head.x() <<"' y='"<< b.head.y() <<"'/>\n";

                    fout << "    </box>\n";
                                            
                }



                fout << "  </image>\n";

                if (!fout)
                    throw dlib::error("ERROR: Unable to write to " + filename + ".");
            }
            fout << "</images>";
            fout << "</dataset>";
        }

    // ------------------------------------------------------------------------------------

        void load_image_dataset_metadata (
            image_dataset_metadata& images,
            const std::string& filename
        )
        {
        }

    // ------------------------------------------------------------------------------------

        // This function returns the contents of the file 'images.xsl'
        const std::string get_decoded_string()
        {
            dlib::base64 base64_coder;
            dlib::compress_stream::kernel_1ea compressor;
            std::ostringstream sout;
            std::istringstream sin;

            // The base64 encoded data from the file 'image_metadata_stylesheet.xsl' we want to decode and return.
            sout << "PFWfgmWfCHr1DkV63lbjjeY2dCc2FbHDOVh0Kd7dkvaOfRYrOG24f0x77/5iMVq8FtE3UBxtGwSd";
            sout << "1ZHOHRSHgieNoeBv8ssJQ75RRxYtFKRY3OTPX5eKQoCN9jUaUnHnR4QZtEHgmKqXSs50Yrdd+2Ah";
            sout << "gNyarPZCiR6nvqNvCjtP2MP5FxleqNf8Fylatm2KdsXmrv5K87LYVN7i7JMkmZ++cTXYSOxDmxZi";
            sout << "OiCH8funXUdF9apDW547gCjz9HOQUI6dkz5dYUeFjfp6dFugpnaJyyprFLKq048Qk7+QiL4CNF/G";
            sout << "7e0VpBw8dMpiyRNi2fSQGSZGfIAUQKKT6+rPwQoRH2spdjsdXVWj4XQAqBX87nmqMnqjMhn/Vd1s";
            sout << "W5aoC0drwRGu3Xe3gn9vBL8hBkRXcKndVwPFMPUeADJ2Z2tc8cb9MqwkOCPBKdfQ+w24u/kvjdsQ";
            sout << "Kq20+MI2uPOSGbCebui/yMveNYUeot5b4s+ZyRIdcbURUHlQQ68BZQVj0WTw+8ohK7VUpjidF/T+";
            sout << "gawZzSnr/HrocEedQL7DyjyVuC03qoFGlwe6Q/4ynndgscVhXyVleneSpOlx/CGOlBKXzwyEpEi8";
            sout << "JV4kPhoJ21N5va7iyDgxuikbDT94tFYje9GtXQOPGPERKzxnp6dCor3a5dPLOfmVsvAdEDtr7B3e";
            sout << "WJpFbfD4XtFfllWmwxggCqyfvz/q7kMQzhsH623CLRDre9AsIRg830e6T992uKgtXx8QM4GPhsjc";
            sout << "leFCNzwDqpeORW7oJ72COj3pgGIn2/BGDWrr1oqu0ACKy7vY32VcgvyFcWbqqdEX71SQ9LcxCbet";
            sout << "iwrBJY8n//Whq55kD0hpSffmW+uup6sV8sbAI7rwnCYpfUNQHXgN+5sLot67jhO03FlafaEmrgfT";
            sout << "XKjBU+z64tCCy1uwJj8gx0CBVFhr8MPzGkKp3rAaeKzVYZJbC2TkZPL+PMjohL0SDhaXxiwX+pKo";
            sout << "VHNRqCmlnrxmB6kikuS5zN5Z6feHW6KNhhcEKuXvDawYNlppJvMt1lE0Q9oKx+JL3Atlm+V8/Q2R";
            sout << "fI5DQAaCaxMXdzJcDfNPMHOIcYlkQPIk9cdLqScrbFu9VjFpqGyGoTGOUiiJP4d3peJvEbRgZZ8d";
            sout << "SUaF4cycZh4yRSI06rrTGi3wyS5HjRFgS+giS2p0ZUi+7YAt1opbkDhcovTxZGUkuavZBCsjZw3C";
            sout << "1CNGurdvhw9mOOL9JY3Vn/wfb9t4ScwaGGrUoeyZHXMARoHYVF1ST7gGOHL2qMRMdXr+k0P3OLxO";
            sout << "4DVe0r3v93vpyqd5KgA=";


            // Put the data into the istream sin
            sin.str(sout.str());
            sout.str("");

            // Decode the base64 text into its compressed binary form
            base64_coder.decode(sin,sout);
            sin.clear();
            sin.str(sout.str());
            sout.str("");

            // Decompress the data into its original form
            compressor.decompress(sin,sout);

            // Return the decoded and decompressed data
            return sout.str();
        }


    }
}

// ----------------------------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        typedef dlib::cmd_line_parser<char>::check_1a_c parser_type;

        parser_type parser;

        parser.add_option("h","Displays this information.");
        parser.add_option("c","Create an XML file named <arg> listing a set of images.",1);
        parser.add_option("d","Include all images files in directory <arg> in the new image file list.",1);
        parser.add_option("r","Search directories recursively for images.");

        parser.parse(argc, argv);

        const char* singles[] = {"h","c","r"};
        parser.check_one_time_options(singles);
        parser.check_sub_option("c", "d");
        parser.check_sub_option("d", "r");

        if (parser.option("h"))
        {
            cout << "Options:\n";
            parser.print_options(cout);
            cout << endl;
            return EXIT_SUCCESS;
        }

        if (parser.option("c"))
        {
            using namespace dlib::imglab;

            image_dataset_metadata metadata;
            metadata.name = "imglab dataset";
            metadata.comment = "Created by imglab tool.";
            for (unsigned long i = 0; i < parser.number_of_arguments(); ++i)
            {
                metadata.images.push_back(image(parser[i]));
            }

            save_image_dataset_metadata(metadata, parser.option("c").argument());

            return EXIT_SUCCESS;
        }

    }
    catch (exception& e)
    {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }
}

