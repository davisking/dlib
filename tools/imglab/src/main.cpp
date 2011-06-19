
#include <iostream>
#include <fstream>
#include <string>

#include <dlib/cmd_line_parser.h>
#include <dlib/geometry.h>
#include <dlib/dir_nav.h>

#include <sstream>
#include <dlib/compress_stream.h>
#include <dlib/base64.h>
#include <dlib/xml_parser.h>


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
                               << "height='" << b.rect.height() << "'";

                    if (b.has_label() || b.has_head())
                    {
                        fout << ">\n";

                        if (b.has_label())
                            fout << "      <label>" << b.label << "</label>\n";
                        if (b.has_head())
                            fout << "      <head x='"<< b.head.x() <<"' y='"<< b.head.y() <<"'/>\n";

                        fout << "    </box>\n";
                    }
                    else
                    {
                        fout << "/>\n";
                    }
                }



                fout << "  </image>\n";

                if (!fout)
                    throw dlib::error("ERROR: Unable to write to " + filename + ".");
            }
            fout << "</images>\n";
            fout << "</dataset>";
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------

        class doc_handler : public document_handler
        {
            std::vector<std::string> ts;
            image temp_image;
            box temp_box;

            image_dataset_metadata& metadata;

        public:

            doc_handler(
                image_dataset_metadata& metadata_
            ):
                metadata(metadata_) 
            {}


            virtual void start_document (
            )
            {
                metadata = image_dataset_metadata();
                ts.clear();
                temp_image = image();
                temp_box = box();
            }

            virtual void end_document (
            )
            {
            }

            virtual void start_element ( 
                const unsigned long line_number,
                const std::string& name,
                const dlib::attribute_list& atts
            )
            {
                if (ts.size() == 0) 
                {
                    if (name != "dataset")
                    {
                        std::ostringstream sout;
                        sout << "Invalid XML document.  Root tag must be <dataset>.  Found <" << name << "> instead.";
                        throw dlib::error(sout.str());
                    }
                    else
                    {
                        ts.push_back(name);
                        return;
                    }
                }


                if (name == "box")
                {
                    if (atts.is_in_list("top")) temp_box.rect.top() = sa = atts["top"];
                    else throw dlib::error("<box> missing required attribute 'top'");

                    if (atts.is_in_list("left")) temp_box.rect.left() = sa = atts["left"];
                    else throw dlib::error("<box> missing required attribute 'left'");

                    if (atts.is_in_list("width")) temp_box.rect.right() = sa = atts["width"];
                    else throw dlib::error("<box> missing required attribute 'width'");

                    if (atts.is_in_list("height")) temp_box.rect.bottom() = sa = atts["height"];
                    else throw dlib::error("<box> missing required attribute 'height'");

                    temp_box.rect.bottom() += temp_box.rect.top()-1;
                    temp_box.rect.right() += temp_box.rect.left()-1;
                }
                else if (name == "head" && ts.back() == "box")
                {
                    if (atts.is_in_list("x")) temp_box.head.x() = sa = atts["x"];
                    else throw dlib::error("<head> missing required attribute 'x'");

                    if (atts.is_in_list("y")) temp_box.head.y() = sa = atts["y"];
                    else throw dlib::error("<head> missing required attribute 'y'");
                }
                else if (name == "image")
                {
                    temp_image.boxes.clear();

                    if (atts.is_in_list("file")) temp_image.filename = atts["file"];
                    else throw dlib::error("<image> missing required attribute 'file'");
                }

                ts.push_back(name);
            }

            virtual void end_element ( 
                const unsigned long line_number,
                const std::string& name
            )
            {
                ts.pop_back();
                if (ts.size() == 0)
                    return;

                if (name == "box" && ts.back() == "image")
                {
                    temp_image.boxes.push_back(temp_box);
                    temp_box = box();
                }
                else if (name == "image" && ts.back() == "images")
                {
                    metadata.images.push_back(temp_image);
                    temp_image = image();
                }
            }

            virtual void characters ( 
                const std::string& data
            )
            {
                if (ts.size() == 2 && ts[1] == "name")
                {
                    metadata.name = trim(data);
                }
                else if (ts.size() == 2 && ts[1] == "comment")
                {
                    metadata.comment = trim(data);
                }
                else if (ts.size() >= 2 && ts[ts.size()-1] == "label" && 
                                           ts[ts.size()-2] == "box")
                {
                    temp_box.label = trim(data);
                }
            }

            virtual void processing_instruction (
                const unsigned long ,
                const std::string& ,
                const std::string& 
            )
            {
            }
        };

    // ----------------------------------------------------------------------------------------

        class xml_error_handler : public error_handler
        {
        public:
            virtual void error (
                const unsigned long line_number
            )
            {
                cout << "There is a non-fatal error on line " << line_number << " in the file we are parsing." << endl;
            }

            virtual void fatal_error (
                const unsigned long line_number
            )
            {
                std::ostringstream sout;
                sout << "There is a fatal error on line " << line_number << " so parsing will now halt.";
                throw dlib::error(sout.str());
            }
        };

    // ------------------------------------------------------------------------------------

        void load_image_dataset_metadata (
            image_dataset_metadata& metadata,
            const std::string& filename
        )
        {
            xml_error_handler eh;
            doc_handler dh(metadata);

            std::ifstream fin(filename.c_str());
            if (!fin)
                throw dlib::error("ERROR: unable to open " + filename + " for reading.");

            xml_parser::kernel_1a parser;
            parser.add_document_handler(dh);
            parser.add_error_handler(eh);
            parser.parse(fin);
        }

    // ------------------------------------------------------------------------------------
    // ------------------------------------------------------------------------------------
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

std::string strip_path (
    const std::string& str,
    const std::string& prefix
)
{
    unsigned long i;
    for (i = 0; i < str.size() && i < prefix.size(); ++i)
    {
        if (str[i] != prefix[i]) 
            return str;
    }

    if (i < str.size() && (str[i] == '/' || str[i] == '\\'))
        ++i;

    return str.substr(i);
}

// ----------------------------------------------------------------------------------------

void make_empty_file (
    const std::string& filename
)
{
    ofstream fout(filename.c_str());
    if (!fout)
        throw dlib::error("ERROR: Unable to open " + filename + " for writing.");
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
        parser.add_option("r","Search directories recursively for images.");

        parser.parse(argc, argv);

        const char* singles[] = {"h","c","r"};
        parser.check_one_time_options(singles);
        parser.check_sub_option("c", "r");

        if (parser.option("h"))
        {
            cout << "Usage: imglab [options] <image files/directories or XML file list>\n";
            parser.print_options(cout);
            cout << endl << endl;
            return EXIT_SUCCESS;
        }

        if (parser.option("c"))
        {
            using namespace dlib::imglab;

            const std::string filename = parser.option("c").argument();
            // make sure the file exists so we can use the get_parent_directory() command to
            // figure out it's parent directory.
            make_empty_file(filename);
            const std::string parent_dir = get_parent_directory(file(filename)).full_name();

            unsigned long depth = 0;
            if (parser.option("r"))
                depth = 30;

            image_dataset_metadata metadata;
            metadata.name = "imglab dataset";
            metadata.comment = "Created by imglab tool.";
            for (unsigned long i = 0; i < parser.number_of_arguments(); ++i)
            {
                try
                {
                    const string temp = strip_path(file(parser[i]).full_name(), parent_dir);
                    metadata.images.push_back(image(temp));
                }
                catch (dlib::file::file_not_found&)
                {
                    // then parser[i] should be a directory

                    std::vector<file> files = get_files_in_directory_tree(parser[i], 
                                                                          match_endings(".png .PNG .jpeg .JPEG .jpg .JPG .bmp .BMP .dng .DNG"),
                                                                          depth);
                    sort(files.begin(), files.end());

                    for (unsigned long j = 0; j < files.size(); ++j)
                    {
                        metadata.images.push_back(image(strip_path(files[j].full_name(), parent_dir)));
                    }
                }
            }

            save_image_dataset_metadata(metadata, filename);

            return EXIT_SUCCESS;
        }

        if (parser.number_of_arguments() == 1)
        {
            dlib::imglab::image_dataset_metadata metadata;
            load_image_dataset_metadata(metadata, parser[0]);
            save_image_dataset_metadata(metadata, "out.xml");
        }
    }
    catch (exception& e)
    {
        cout << e.what() << endl;
        return EXIT_FAILURE;
    }
}

