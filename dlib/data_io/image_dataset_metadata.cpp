// Copyright (C) 2011  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_IMAGE_DAtASET_METADATA_CPP__
#define DLIB_IMAGE_DAtASET_METADATA_CPP__

#include "image_dataset_metadata.h"

#include <fstream>
#include <sstream>
#include "../compress_stream.h"
#include "../base64.h"
#include "../xml_parser.h"
#include "../string.h"

// ----------------------------------------------------------------------------------------

namespace dlib
{
    namespace image_dataset_metadata
    {

    // ------------------------------------------------------------------------------------

        const std::string get_decoded_string();
        void create_image_metadata_stylesheet_file(const std::string& main_filename)
        {
            std::string path;
            std::string::size_type pos = main_filename.find_last_of("/\\");
            if (pos != std::string::npos)
                path = main_filename.substr(0,pos+1);

            std::ofstream fout((path + "image_metadata_stylesheet.xsl").c_str());
            if (!fout)
                throw dlib::error("ERROR: Unable to open image_metadata_stylesheet.xsl for writing.");

            fout << get_decoded_string();

            if (!fout)
                throw dlib::error("ERROR: Unable to write to image_metadata_stylesheet.xsl.");
        }

        void save_image_dataset_metadata (
            const dataset& meta,
            const std::string& filename
        )
        {
            create_image_metadata_stylesheet_file(filename);

            const std::vector<image>& images = meta.images;

            std::ofstream fout(filename.c_str());
            if (!fout)
                throw dlib::error("ERROR: Unable to open " + filename + " for writing.");

            fout << "<?xml version='1.0' encoding='ISO-8859-1'?>\n";
            fout << "<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n";
            fout << "<dataset>\n";
            fout << "<name>" << meta.name << "</name>\n";
            fout << "<comment>" << meta.comment << "</comment>\n";
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
                    if (b.difficult)
                        fout << " difficult='" << b.difficult << "'";
                    if (b.truncated)
                        fout << " truncated='" << b.truncated << "'";
                    if (b.occluded)
                        fout << " occluded='" << b.occluded << "'";

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

            dataset& meta;

        public:

            doc_handler(
                dataset& metadata_
            ):
                meta(metadata_) 
            {}


            virtual void start_document (
            )
            {
                meta = dataset();
                ts.clear();
                temp_image = image();
                temp_box = box();
            }

            virtual void end_document (
            )
            {
            }

            virtual void start_element ( 
                const unsigned long ,
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

                    if (atts.is_in_list("difficult")) temp_box.difficult = sa = atts["difficult"];
                    if (atts.is_in_list("truncated")) temp_box.truncated = sa = atts["truncated"];
                    if (atts.is_in_list("occluded"))  temp_box.occluded  = sa = atts["occluded"];

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
                const unsigned long ,
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
                    meta.images.push_back(temp_image);
                    temp_image = image();
                }
            }

            virtual void characters ( 
                const std::string& data
            )
            {
                if (ts.size() == 2 && ts[1] == "name")
                {
                    meta.name = trim(data);
                }
                else if (ts.size() == 2 && ts[1] == "comment")
                {
                    meta.comment = trim(data);
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
                const unsigned long 
            ) { }

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
            dataset& meta,
            const std::string& filename
        )
        {
            xml_error_handler eh;
            doc_handler dh(meta);

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

#endif // DLIB_IMAGE_DAtASET_METADATA_CPP__


