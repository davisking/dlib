// Copyright (C) 2019  Paul Dreik (github@pauldreik.se)
// License: Boost Software License   See LICENSE.txt for the full license.

#include "tester.h"
#include <dlib/image_io.h>

namespace
{
    dlib::logger dlog("test.image_load");

    using namespace test;

    class image_load_tester : public tester
    {
        /*!
             Tests the image loading utilities (parsing image files).
        !*/
    public:
        image_load_tester (
        ) :
            tester (
                "test_image_load",                // the command line argument name for this test
                "Runs image files recursively found in arg through the image load routine.", // the command line argument description
                1                                   // the number of command line arguments for this test
            )
        {}

        void perform_test (
            const std::string& arg
        )
        {
            dlog << dlib::LINFO << "the argument passed to this test was " << arg;

            m_arg=arg;

           print_spinner();

           // recursively load files
           invoke_on_file_or_dir(m_arg);
        }
     private:
        void invoke_on_file_or_dir(const std::string& dir_or_file) {
            if(dlib::file_exists(dir_or_file)) {
                invoke_on_file(dir_or_file);
            } else {
            const auto files=dlib::get_files_in_directory_tree(dir_or_file,dlib::match_all{});
            for(const auto& file: files) {
                invoke_on_file(file);
            }
            }
        }
        void invoke_on_file(const std::string& filename) {
            print_spinner();
            dlog << dlib::LINFO << "attempting to parse image "<<filename;

                   dlib::array2d<dlib::rgb_pixel> img;
                   try {
                       dlib::load_image(img, filename);
                       dlog << dlib::LINFO << "successfully parsed image "<<filename;
                   } catch(std::exception& e) {
                       dlog << dlib::LERROR << "got an exception while loading image "<<filename<<": "<<e.what();
                   }
                   // get text output, so the user sees something happened
                   DLIB_TEST_MSG(true,"survived loading an image");
        }
        std::string m_arg;
    };

    // Create an instance of this object.  Doing this causes this test
    // to be automatically inserted into the testing framework whenever this cpp file
    // is linked into the project.  Note that since we are inside an unnamed-namespace 
    // we won't get any linker errors about the symbol a being defined multiple times. 
    image_load_tester a;
}



